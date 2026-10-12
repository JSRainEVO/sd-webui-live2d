"""Forge / reForge adapter. The inference engine stays in a separate process."""
from __future__ import annotations

from contextlib import closing
import time

import gradio as gr
from PIL import ImageOps
from modules import call_queue, devices, script_callbacks, sd_models
try:
    from modules import infotext_utils as copypaste
except ImportError:
    from modules import generation_parameters_copypaste as copypaste

from live2d_extension.bridge import (
    BridgeError, Cancelled, Jobs, command_lines, default_engine, engine_command,
    engine_root, new_job_directory, normalize_resolution, pack_result, save_input, selected_infotext,
)
from live2d_extension.gpu import gpu_session

GRADIO_MAJOR = int(gr.__version__.split(".")[0])
SLIDER_MAXIMUM = 4096


def js_argument(value):
    return {"js" if GRADIO_MAJOR >= 4 else "_js": value}

JOBS = Jobs()
COMPONENTS = {}
SEND_BUTTONS = {}


def capture_component(component, **kwargs):
    elem_id = kwargs.get("elem_id") or getattr(component, "elem_id", None)
    if elem_id in {"txt2img_gallery", "img2img_gallery", "generation_info_txt2img", "generation_info_img2img"}:
        COMPONENTS[elem_id] = component
    for tab in ("txt2img", "img2img"):
        if elem_id == f"{tab}_send_to_extras":
            SEND_BUTTONS[tab] = gr.Button("发送到 Live2D", elem_id=f"{tab}_send_to_live2d", size="sm")


def receive_result(filedata, info, index):
    try:
        infotext = selected_infotext(info, index)
        image = copypaste.image_from_url_text(filedata)
        if image is None:
            raise BridgeError("请先生成并选中一张角色图片。")
        return image, infotext, "图片已接收，点击“转换为 Live2D”开始。"
    except BridgeError as error:
        raise gr.Error(str(error))


def size_update(value):
    return gr.update(value=value, maximum=max(SLIDER_MAXIMUM, value))


def canvas_dimensions(width, height):
    try:
        return normalize_resolution(width), normalize_resolution(height)
    except BridgeError:
        raise BridgeError("宽度和高度必须是正整数。") from None


def dimensions_from_image(image):
    if image is None:
        raise gr.Error("请先上传图片，或从文生图 / 图生图发送一张图片。")
    width, height = canvas_dimensions(*ImageOps.exif_transpose(image).size)
    return size_update(width), size_update(height)


def swap_dimensions(width, height):
    width, height = canvas_dimensions(width, height)
    return size_update(height), size_update(width)


def prepare_job(image, engine, previous, mode, width=512, height=512):
    if mode == "convert" and image is None:
        raise gr.Error("请上传图片，或从文生图 / 图生图发送一张图片。")
    try:
        if mode == "convert":
            width, height = canvas_dimensions(width, height)
            sizes = size_update(width), size_update(height)
        else:
            sizes = gr.update(), gr.update()
        engine_root(engine)
        key = JOBS.create(previous)
    except BridgeError as error:
        raise gr.Error(str(error))
    status = f"任务已提交，画布尺寸为 {width} × {height}。" if mode == "convert" else "任务已提交。"
    return (key, gr.update(interactive=False), gr.update(interactive=False),
            gr.update(interactive=True), status, None, None, "", *sizes)


def stop_job(key):
    return "正在停止任务并释放资源……" if JOBS.cancel(key) else "当前没有运行中的任务。"


def run_job(key, image, infotext, engine, width, height, mode):
    """Hold the host's GPU queue for the whole conversion and restoration."""
    logs = []
    archive = preview = None
    status = "等待当前出图任务结束……" if mode == "convert" else "正在检测并补齐转换环境……"
    last_update = 0.0
    work_acquired = False

    def output(finished=False):
        return (status, str(preview) if preview else None, str(archive) if archive else None,
                "\n".join(logs[-120:]), gr.update(interactive=finished),
                gr.update(interactive=finished), gr.update(interactive=not finished))

    try:
        cancel = JOBS.event(key)
        root = engine_root(engine)
        if mode == "convert":
            width, height = canvas_dimensions(width, height)
        yield output()
        while not JOBS.work_lock.acquire(timeout=.3):
            if cancel.is_set():
                raise Cancelled("任务已停止")
            status = "等待其他 Live2D 任务完成……"
            if time.monotonic() - last_update > .5:
                last_update = time.monotonic()
                yield output()
        work_acquired = True
        if cancel.is_set():
            raise Cancelled("任务已停止")
        if mode == "setup":
            command = engine_command(root, "auto_setup.py", "--fix")
            with closing(command_lines(command, root, cancel)) as stream:
                for line in stream:
                    if line:
                        logs.append(line)
                    if time.monotonic() - last_update > .5:
                        last_update = time.monotonic()
                        yield output()
            status = "转换环境已就绪，可以发送图片并开始转换。首次转换可能继续下载模型权重。"
        else:
            # The host uses this same lock for txt2img/img2img and model loading.
            with call_queue.queue_lock:
                if cancel.is_set():
                    raise Cancelled("任务已停止")
                directory = new_job_directory(root)
                source = save_input(image, directory, infotext)
                status = "正在为转换释放显存……"
                yield output()
                with gpu_session(sd_models, devices) as host:
                    logs.append(f"已通过 {host} 显存管理接口暂停出图模型。")
                    command = engine_command(root, "auto_live2d.py", str(source),
                                             "--output", str(directory),
                                             "--width", str(width), "--height", str(height))
                    status = "正在检查环境并加载转换模型……"
                    with closing(command_lines(command, root, cancel)) as stream:
                        for line in stream:
                            if line:
                                logs.append(line)
                                if line.startswith("[1/3]"):
                                    status = "正在自动拆层和补全遮挡区域……"
                                elif line.startswith("[2/3]"):
                                    status = "正在检查部件、嘴部与图层顺序……"
                                elif line.startswith("[3/3]"):
                                    status = "正在自动绑模并导出 Live2D 文件……"
                            if time.monotonic() - last_update > .5:
                                last_update = time.monotonic()
                                yield output()
                    archive, preview, warnings = pack_result(directory)
                    status = "模型已导出，正在恢复 SD 出图模型……"
                    yield output()
                status = "模型已导出，可下载完整文件包。"
                if warnings:
                    status += "\n" + "\n".join(warnings)
    except Cancelled:
        status = "任务已停止，中间结果已保留。"
    except Exception as error:
        status = "任务失败：" + str(error)
        logs.append(status)
    finally:
        if work_acquired:
            JOBS.work_lock.release()
        JOBS.finish(key)
    yield output(finished=True)


def create_tab():
    with gr.Blocks() as tab:
        gr.Markdown("### 图片转 Live2D\n从文生图 / 图生图发送选中的角色图片，也可以直接上传。优先支持正面二次元半身像。")
        key = gr.State(None)
        convert_mode = gr.State("convert")
        setup_mode = gr.State("setup")
        index = gr.Number(value=0, visible=False, precision=0)
        with gr.Row():
            source = gr.Image(label="角色图片", type="pil", image_mode="RGBA", **(
                {"sources": ["upload"]} if GRADIO_MAJOR >= 4 else {"source": "upload"}))
            preview = gr.Image(label="拆层素材预览", type="filepath", interactive=False)
        infotext = gr.Textbox(label="生成参数", visible=False)
        with gr.Row():
            with gr.Column(scale=4):
                width = gr.Slider(minimum=64, maximum=SLIDER_MAXIMUM, step=64,
                                  value=512, label="宽度（像素）", elem_id="live2d_width")
                height = gr.Slider(minimum=64, maximum=SLIDER_MAXIMUM, step=64,
                                   value=512, label="高度（像素）", elem_id="live2d_height")
            with gr.Column(scale=1, min_width=130):
                use_source_size = gr.Button("按原图宽高填写")
                swap_size = gr.Button("交换宽高 ⇄")
        gr.Markdown("拖动滑条或直接输入宽、高，按 64 像素调整。角色保持比例并居中，比例不同时补透明边。尺寸越大，占用显存越多。")
        with gr.Row():
            start = gr.Button("转换为 Live2D", variant="primary")
            stop = gr.Button("停止", interactive=False)
        status = gr.Textbox(label="进度", value="选择一张图片后开始。", lines=3, interactive=False)
        download = gr.File(label="下载 Live2D 模型文件包", interactive=False)
        gr.Markdown("转换期间会顺序使用显存，完成或停止后恢复 SD 出图。素材预览不代表最终动作效果，张嘴细节需要检查。")
        with gr.Accordion("环境与运行记录", open=False):
            engine = gr.Textbox(label="转换环境目录", value=str(default_engine()))
            setup = gr.Button("检测并补齐环境")
            log = gr.Textbox(label="运行记录", lines=12, interactive=False)
        inputs = [key, source, infotext, engine, width, height]
        outputs = [status, preview, download, log, start, setup, stop]
        prepare_outputs = [key, start, setup, stop, status, preview, download, log, width, height]
        start.click(prepare_job, [source, engine, key, convert_mode, width, height], prepare_outputs, queue=False).success(
            run_job, inputs + [convert_mode], outputs)
        setup.click(prepare_job, [source, engine, key, setup_mode, width, height], prepare_outputs, queue=False).success(
            run_job, inputs + [setup_mode], outputs)
        stop.click(stop_job, [key], [status], queue=False)
        use_source_size.click(dimensions_from_image, [source], [width, height], queue=False)
        swap_size.click(swap_dimensions, [width, height], [width, height], queue=False)
        source.upload(lambda: "", outputs=[infotext], queue=False)

        # Host galleries live in separate Blocks; their events are merged when
        # WebUI renders every interface into its final root Blocks.
        for origin, button in SEND_BUTTONS.items():
            gallery = COMPONENTS.get(f"{origin}_gallery")
            info = COMPONENTS.get(f"generation_info_{origin}")
            if gallery is not None and info is not None:
                button.click(receive_result, [gallery, info, index], [source, infotext, status],
                             queue=False, **js_argument("live2d_selected_result")).success(
                                 fn=None, inputs=[], outputs=[], **js_argument("switch_to_live2d"))
    return [(tab, "Live2D", "live2d")]


def reset_components():
    COMPONENTS.clear()
    SEND_BUTTONS.clear()


script_callbacks.on_before_ui(reset_components)
script_callbacks.on_after_component(capture_component)
script_callbacks.on_ui_tabs(create_tab)
