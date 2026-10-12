**Stable Diffusion 图片转 Live2D 扩展**

在 Forge / reForge 的文生图或图生图页面生成角色后，发送到 Live2D 页面，即可自动拆层、补全遮挡、初步绑模并导出模型文件包。也可以直接上传图片。

最好使用正面的二次元半身像。脸部和五官清楚、头发遮挡较少、背景简单的图片，更容易得到可用的初版模型。侧脸、复杂姿势、多人图片和复杂背景会增加修正难度。

经验证本项目在最低8gb显存的设备上可用。

当前转换环境要求 Windows x64 和支持 CUDA 的 NVIDIA 显卡。以上说明基于现有 8GB 设备的运行结果；8GB 以下尚未验证。建议从 512 × 512 开始，较大尺寸会增加显存占用和处理时间。


**安装插件**

推荐在 Forge / reForge 的“扩展”页面打开“从网址安装”，填写以下仓库地址，然后点击安装：

https://github.com/JSRainEVO/sd-webui-live2d

安装完成后重启 WebUI，打开 Live2D 页面，在“环境与运行记录”中点击“检测并补齐环境”。


也可以手动安装：

1. 将插件压缩包解压到 Forge / reForge 的 extensions 文件夹，并将插件文件夹命名为 sd-webui-live2d。GitHub 的 Code → Download ZIP 下载结果通常带有 -main 后缀，需要去掉该后缀。

2. 确认目录结构为 extensions/sd-webui-live2d/scripts/live2d_tab.py，压缩包中的 engine 文件夹也要保留。

3. 重启 WebUI，打开 Live2D 页面，在“环境与运行记录”中点击“检测并补齐环境”。

4. 如果已经配置过本项目，可以把“转换环境目录”填写为原项目目录，复用已有环境和模型缓存。


**第一次会安装哪些环境**

安装器会在所选转换环境目录内补齐缺少的组件。插件默认目录是 engine，独立程序使用项目目录，转换使用独立 Python 环境。

1. 独立 Python 3.12、.auto_env 环境，以及需要时安装的 uv 包管理工具。

2. PyTorch 2.8.0、torchvision 0.23.0、torchaudio 2.8.0，以及对应 cu128 软件包提供的 CUDA 运行组件。这一步需要下载数 GB 文件。

3. 模型加载与量化组件：Diffusers、Transformers、Accelerate、bitsandbytes、huggingface_hub、tokenizers、safetensors 等。

4. 图像和 PSD 处理组件：Pillow、NumPy、OpenCV、psd-tools，以及 See-through 依赖清单中的科学计算、界面和辅助组件，如 SciPy、scikit-image、Matplotlib、pandas、scikit-learn、PyQt6 等。

5. 转换工具：See-through 源码，用于单图拆层和遮挡补全；PSD2Live Windows x64 便携版，用于自动绑模和导出。

运行前需要已安装 NVIDIA 显卡驱动和可从系统调用的 Git。安装器不会自动安装驱动和 Git。使用独立启动入口 auto_start.bat 时，还需要系统能够调用 Python 启动器 py。

环境与模型是两个准备步骤。“环境已就绪”表示计算环境和工具已准备好，首次转换图片时仍可能继续下载模型权重。


**第一次会下载哪些模型**

LayerDiff，约 3.76 GB。
仓库：24yearsold/seethroughv0.0.2_layerdiff3d_nf4
用途：拆分角色部件，并补全被遮挡的内容。
下载页面：https://huggingface.co/24yearsold/seethroughv0.0.2_layerdiff3d_nf4/tree/main

Marigold，约 1.93 GB。
仓库：24yearsold/seethroughv0.0.1_marigold_nf4
用途：估计深度，辅助判断部件的前后关系。
下载页面：https://huggingface.co/24yearsold/seethroughv0.0.1_marigold_nf4/tree/main

PS：在第一次在Web ui点击运行后，模型会自动下载。

两套模型合计约 5.69 GB。模型以 NF4 量化版本为主，所需编码器、VAE、分词器和配置文件包含在对应仓库中，需要保留完整目录结构。这是文件下载量和磁盘占用，运行时的显存占用另计；环境、安装缓存和生成结果也会占用磁盘。

想提前下载时，完成环境安装后，双击转换环境目录中的 download_models.bat。它会整库下载两套模型、保留子文件夹，并自动建立插件识别所需的目录，无需在网页逐个下载文件。该工具只下载文件，不加载模型、不运行图片转换。

单独使用下载工具时，将 live2d-model-downloader.zip 解压到转换环境目录，确认 download_models.bat、download_models.py 和 auto_live2d.py 位于同一层。

下载工具使用当前转换环境目录的 .model-cache/hub。两套模型有完整独立缓存时，转换会优先离线加载。Forge 继承的缓存设置也可能使文件下载到 Forge 的 models/diffusers，实际位置以日志为准。

下载中断后，整库下载工具会保留已完成文件，再次运行可继续下载。如果浏览器依靠代理访问 Hugging Face，在提示中填写同一代理的 HTTP 地址；无需代理则直接回车。


**首次需要等多久**

首次准备包含环境安装、模型下载和图片转换，等待时间通常明显长于后续使用。

环境安装可先预留约 10～30 分钟。这是估算，取决于软件包下载速度、已有组件和磁盘速度；下载慢时可能更久。

两套模型约 5.69 GB，按实际下载速度估算：

10 MB/s：模型下载约 10 分钟。

5 MB/s：模型下载约 20 分钟。

1 MB/s：模型下载约 95 分钟。

以上下载时间只按数据量计算，没有包含连接等待和重试。这里使用 MB/s，不能与宽带套餐中的 Mbps 直接等同。

如果环境和模型全部从零准备，在稳定 5～10 MB/s 的下载速度下，可先预留约 30～60 分钟。慢速连接可能需要两小时以上。这些是估算，并非每台设备的实测保证。

日志停在 Building LayerDiff3D pipeline 或 Building Marigold depth pipeline 时，可能仍在下载或加载模型。下载慢时日志会长时间不更新，请结合下载文件变化判断进展。出现任务失败或错误堆栈时，再按完整错误记录排查。


**512 × 512 图片的大概运行时间 （仅供参考）**

模型已下载完整后，以 512 × 512 处理正面的二次元半身像，现有 8GB 显存设备的 AI 阶段实测约 159 秒，即 2 分 39 秒，包含拆层、深度估计和原始 PSD 组装。

加上后续图层检查、绑模和导出，整套转换可先预留约 3～5 分钟。全流程时间是估算，图片内容、显卡、磁盘和后台占用都会影响结果。首次模型下载的时间另算。

作为参考，此前 768 处理分辨率的 AI 阶段约 306 秒。一次实际 Forge 转换中，1024 处理分辨率仅身体和头部拆层就用了约 480 秒，因此更大图片需要更长等待时间。


**使用步骤**

1. 在文生图或图生图生成角色，点选单张图片，再点击“发送到 Live2D”。也可以直接上传 PNG、JPG 或 WebP 图片。批量结果中的拼图需要先切换成单张图片。

2. 设置“宽度”和“高度”。可以拖动滑条或直接填写数值，也可以点击“按原图宽高填写”或“交换宽高”。首次建议用 512 × 512。

3. 点击“转换为 Live2D”，等待拆层、深度估计、图层检查和自动绑模完成。

4. 下载“Live2D 模型文件包”，完整解压并保留 model 文件夹内的全部文件。

宽、高控制分层 PSD 和素材预览的实际画布尺寸。角色保持原图比例，缩放后居中，画布比例不同时补透明边。尺寸按 64 像素对齐，例如 1000 × 1500 调整为 1024 × 1536。

滑条默认范围为 64～4096；“按原图宽高填写”会在需要时扩大滑条上限。内部拆层模型仍在正方形画布上计算，边长取所选宽、高中较大的值，拆层后统一去掉补边。贴图尺寸由绑模工具打包决定。

转换期间会通过 WebUI 的模型管理接口临时释放显存，并在任务结束后调用恢复接口。“停止”会终止本任务及其转换子进程，中间文件会保留。如果正在等待已有 SD 出图任务，则需要等出图任务释放任务锁后处理停止请求。

模型包包含运行时模型、贴图、动作、物理文件、导出的 CMO3 工程、自动分层 PSD、生成参数和转换检查报告。任务文件保存在转换环境的 outputs/webui/时间-任务编号 中。


**查看生成的模型**

下载模型包后完整解压，保留 model 文件夹内的全部文件。模型入口是以 .model3.json 结尾的文件。

推荐使用官方 Cubism Viewer for Unity。在 Config 页面点击 Load Json，选择模型入口，在 Parameters 页面拖动参数检查转头、眨眼和嘴形。

查看器下载：https://www.live2d.com/zh-CHS/cubism/download/viewer-unity/
操作说明：https://docs.live2d.com/zh-CHS/cubism-editor-manual/using-cubism-viewer-unity/

.cmo3 属于 Cubism Editor 的编辑工程，是否可打开取决于目标 Editor 版本。素材预览展示的是拆层合成效果，最终动作效果请以查看器为准。


**当前效果和使用建议**

已有样例完成图片到 Live2D 文件包的全流程，且收到实际 Forge 运行、完整导出和 Cubism Viewer 试用成功的反馈。自动生成结果适合作为继续精修的初版，效果随输入图片变化。

最好使用正面的半身像，并在查看器中检查眨眼、嘴形、头部角度、发丝遮挡和动作。单张图片没有提供的闭眼、口腔和转头细节，自动补全仍可能不准确。目前推荐得到输出结果后自行修改自动生成的 PSD，再重新导出。

在透明或均匀背景下，图层检查会参考原图过滤多余部件，并尝试恢复遗漏的可见嘴线。复杂背景会跳过部分过滤；恢复闭嘴线条不会自动生成隐藏的口腔。

此前一个半身样例有 23 个可绘制对象和 20 个参数，通过官方 Cubism Core 的一致性检查、初始化和 41 个参数姿态的有限坐标检查。reForge 实际集成、其他设备以及 CMO3 在官方 Editor 中编辑，仍需更多验证。

转换环境存在官方 Cubism Core 和 Node.js 时，报告会记录运行库检查结果。缺少它们时，该项会标为未验证。


**升级和常见安装问题**

升级时，用新版压缩包覆盖 extensions 中的同名插件文件夹，再重启 WebUI。保留原有环境和模型缓存。若使用原项目作为转换环境目录，也需要同步更新该目录中的引擎脚本。

旧版出现“无法定位 PSD2Live Windows x64 便携版”时，先更新插件再检测环境。当前安装器兼容 windows-x64、windows-x86_64 和 windows-amd64 命名，优先下载普通便携版。


**第三方项目与许可**

See-through：https://github.com/shitagaki-lab/see-through
负责单图拆层和遮挡补全，源码按 Apache-2.0 使用。模型权重按各模型仓库中的 LICENSE 和 NOTICE 使用。

PSD2Live：https://github.com/tsunehimatoi/psd2live
负责自动绑模与 Cubism 导出。主程序 GPL-3.0，部分组件 MIT。使用或再分发便携版时请查看上游许可，安装器保留许可文件。

Live2D 官方 Cubism Core：https://cubism.live2d.com/sdk-web/cubismcore/live2dcubismcore.min.js
按官方许可使用，插件压缩包不包含该运行库。

本项目负责环境检查、安装、流程编排、结果检查和界面。第三方源码与工具下载到 third_party。


**开发与打包**

插件仓库中的 engine 包含转换引擎源码，首次准备时会建立独立环境并下载第三方工具。此前已完成 Gradio 4.40.0 / 3.41.2 接口测试。
