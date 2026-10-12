# SD WebUI Live2D · 图片转模型

![Version](https://img.shields.io/badge/Version-v0.1.0__Beta-6366f1?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Windows_x64-0078d4?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.12-3776ab?style=flat-square&logo=python&logoColor=white)
![GPU](https://img.shields.io/badge/NVIDIA-8_GB_verified-76b900?style=flat-square&logo=nvidia&logoColor=white)
![Quantization](https://img.shields.io/badge/Quantization-NF4-a855f7?style=flat-square)

**在 Forge / reForge 中，将一张角色图片转换为可在 Live2D 查看器中使用的初版模型。**

从文生图或图生图发送图片，也可以直接上传；插件会自动拆层、补全遮挡、估计深度、初步绑模，并导出模型与素材文件包。

[下载插件](https://github.com/JSRainEVO/sd-webui-live2d/releases) · [安装](#install) · [模型与环境](#setup) · [耗时](#timing) · [查看模型](#viewer) · [反馈问题](https://github.com/JSRainEVO/sd-webui-live2d/issues)

> **推荐图片：正面的二次元半身像。** 五官清晰、背景简单、头发遮挡较少时，更容易得到可用的初版。当前建议得到输出结果后，自行修改自动生成的 PSD，再重新导出。

<br>

## 一眼了解

| 项目 | 当前情况 |
| :--- | :--- |
| 当前版本 | **v0.1.0_Beta**，首次公开试用版 |
| 使用入口 | Forge / reForge 的 **Live2D** 页面；已有实际 Forge 运行反馈 |
| 转换环境 | Windows x64 · 支持 CUDA 的 NVIDIA 显卡 |
| 显存验证 | **8GB 设备已验证**；8GB 以下尚未验证 |
| 推荐起步尺寸 | **512 × 512**，宽、高可分别设置 |
| 模型下载量 | **约 5.69 GB**，首次转换自动下载，后续复用 |
| 首次准备时间 | **约 30～60 分钟估算**，按稳定 5～10 MB/s 下载速度计算 |
| 模型缓存齐全后 | 512 × 512 全流程 **约 3～5 分钟估算**；AI 阶段实测约 159 秒 |
| 输出结果 | Live2D 运行时模型、贴图、动作、物理文件、CMO3、PSD 和检查报告 |

经验证本项目在最低8gb显存的设备上可用。以上基于现有 8GB 设备的运行结果；较大尺寸会增加显存占用和处理时间。

<br>

## 功能概览

| 功能 | 使用体验 |
| :--- | :--- |
| **出图后直接转换** | 文生图 / 图生图点选单张图片，发送到 Live2D；支持上传 PNG、JPG、WebP |
| **自动拆层与初步绑模** | 拆分角色部件、补全遮挡、估计深度，并导出模型文件包 |
| **独立宽高控制** | 拖动滑条或输入数值；支持按原图填写、交换宽高、保持比例并居中 |
| **首次环境准备** | 检测并补齐独立转换环境，支持提前整库下载模型 |
| **任务控制与结果检查** | 显示运行记录、支持停止任务，输出转换检查报告 |

<br>

<a id="install"></a>

## 安装插件

### 方式一：从仓库地址安装

1. 在 Forge / reForge 中打开 **扩展 → 从网址安装**。
2. 填写下面的地址，点击安装。
3. 重启 WebUI，打开 **Live2D → 环境与运行记录 → 检测并补齐环境**。

```text
https://github.com/JSRainEVO/sd-webui-live2d
```

### 方式二：下载压缩包

1. 从 [Releases](https://github.com/JSRainEVO/sd-webui-live2d/releases) 下载 `sd-webui-live2d.zip`，解压到 WebUI 的 `extensions` 文件夹。
2. 如果使用 **Code → Download ZIP**，将带有 `-main` 后缀的文件夹改名为 `sd-webui-live2d`。
3. 确认下面的目录结构；保留完整 `engine` 文件夹，然后重启 WebUI 并检测环境。

```text
extensions/
└── sd-webui-live2d/
    ├── scripts/
    │   └── live2d_tab.py
    ├── javascript/
    ├── live2d_extension/
    └── engine/
```

**已有转换环境？** 在“转换环境目录”中填写原项目目录，即可复用已有环境和模型缓存。

<br>

<a id="setup"></a>

## 第一次会安装和下载什么

### 环境与转换工具

安装器会在所选转换环境目录内补齐缺少的组件。插件默认目录是 `engine`，独立程序使用项目目录，转换使用独立 Python 环境。

| 类别 | 自动准备的内容 |
| :--- | :--- |
| Python 环境 | Python **3.12**、`.auto_env`，以及需要时安装的 **uv** |
| GPU 计算组件 | PyTorch **2.8.0**、torchvision **0.23.0**、torchaudio **2.8.0**，以及 cu128 软件包提供的 CUDA 运行组件；需下载数 GB 文件 |
| 模型加载与量化 | Diffusers、Transformers、Accelerate、bitsandbytes、huggingface_hub、tokenizers、safetensors 等 |
| 图像与 PSD 处理 | Pillow、NumPy、OpenCV、psd-tools，以及 See-through 依赖清单中的科学计算、界面和辅助组件 |
| 拆层与模型导出 | See-through 源码、PSD2Live **Windows x64 便携版** |

> **运行前请准备好 NVIDIA 显卡驱动和 Git。** 安装器不会自动安装驱动和 Git。“环境已就绪”表示计算环境和工具已准备好，首次转换仍可能下载模型权重。

<details>
<summary>展开：其他环境依赖与独立启动入口</summary>

See-through 的依赖还包括 SciPy、scikit-image、Matplotlib、pandas、scikit-learn、PyQt6 等。具体组件由上游依赖清单决定。

使用独立启动入口 `auto_start.bat` 时，需要系统能够调用 Python 启动器 `py`；该入口属于独立项目，插件仓库中不提供这个启动文件。

</details>

### 两套模型 · 合计约 5.69 GB

| 模型 | 下载量 | 用途 | 完整仓库 |
| :--- | :--- | :--- | :--- |
| **LayerDiff** | **约 3.76 GB** | 拆分角色部件、补全遮挡内容 | [seethroughv0.0.2_layerdiff3d_nf4](https://huggingface.co/24yearsold/seethroughv0.0.2_layerdiff3d_nf4/tree/main) |
| **Marigold** | **约 1.93 GB** | 估计深度、辅助判断部件前后关系 | [seethroughv0.0.1_marigold_nf4](https://huggingface.co/24yearsold/seethroughv0.0.1_marigold_nf4/tree/main) |

**第一次在 WebUI 点击转换后，缺少的模型会自动下载。** 模型以 NF4 量化版本为主，所需编码器、VAE、分词器和配置文件包含在仓库中，需要保留完整目录结构。

5.69 GB 是模型文件下载量和磁盘占用，运行时显存占用另计；环境、安装缓存和生成结果也会占用磁盘。

<details>
<summary>展开：提前下载、缓存位置和代理设置</summary>

- 完成环境安装后，双击转换环境目录中的 `download_models.bat`，即可整库下载两套模型并建立插件识别的目录，无需在网页逐个下载。该工具只下载文件，不加载模型或运行转换。
- 单独使用下载工具时，将 `live2d-model-downloader.zip` 解压到转换环境目录，确认 `download_models.bat`、`download_models.py` 和 `auto_live2d.py` 位于同一层。
- 下载工具使用当前转换环境的 `.model-cache/hub`；两套模型有完整独立缓存时，转换会优先离线加载。
- Forge 继承的缓存设置也可能使文件下载到 Forge 的 `models/diffusers`，实际位置以日志为准。
- 下载中断后，已完成文件会保留，再次运行可继续下载。需要代理时，在提示中填写 HTTP 代理地址；无需代理则直接回车。

</details>

<br>

<a id="timing"></a>

## 需要等多久

### 首次准备：安装环境 → 下载模型 → 转换图片

| 阶段 / 实际下载速度 | 建议预留时间 | 说明 |
| :--- | :--- | :--- |
| 环境安装 | **约 10～30 分钟估算** | 取决于已有组件、软件包下载速度和磁盘速度 |
| 两套模型 · 10 MB/s | **约 10 分钟** | 仅按 5.69 GB 数据量计算 |
| 两套模型 · 5 MB/s | **约 20 分钟** | 仅按数据量计算 |
| 两套模型 · 1 MB/s | **约 95 分钟** | 未计入连接等待和重试 |
| 从零准备 · 稳定 5～10 MB/s | **约 30～60 分钟估算** | 慢速连接可能需要两小时以上 |

上述时间并非每台设备的实测保证；**MB/s 与宽带套餐中的 Mbps 不同**。首次准备的等待时间通常明显长于后续使用。

### 512 × 512 运行时间 · 仅供参考

模型已下载完整后，以 512 × 512 处理正面的二次元半身像：

| 测量范围 | 现有 8GB 显存设备的结果 |
| :--- | :--- |
| AI 阶段：拆层、深度估计、原始 PSD 组装 | **实测约 159 秒（2 分 39 秒）** |
| 全流程：再加图层检查、绑模和导出 | **预计约 3～5 分钟** |

首次模型下载时间另算。全流程时间是估算，图片内容、显卡、磁盘和后台占用都会影响结果。

<details>
<summary>展开：较大尺寸的耗时参考，以及日志不更新的情况</summary>

- 此前 768 处理分辨率的 AI 阶段约 **306 秒**。
- 一次实际 Forge 转换中，1024 处理分辨率仅身体和头部拆层就用了约 **480 秒**。这些测量范围不同，不能直接视为完整流程的对比。
- 日志停在 `Building LayerDiff3D pipeline` 或 `Building Marigold depth pipeline` 时，可能仍在下载或加载模型。下载慢时日志会长时间不更新，请结合下载文件变化判断进展。
- 出现任务失败或错误堆栈时，再按完整错误记录排查。

</details>

<br>

## 使用步骤

1. **选择图片**：在文生图 / 图生图点选单张角色图片，点击“发送到 Live2D”，或上传 PNG、JPG、WebP。批量结果中的拼图需先切换成单张图片。
2. **设置宽高**：拖动“宽度”“高度”滑条或输入数值，也可以按原图填写、交换宽高。首次建议 **512 × 512**。
3. **开始转换**：点击“转换为 Live2D”，等待拆层、深度估计、图层检查和自动绑模。
4. **下载结果**：下载“Live2D 模型文件包”，完整解压并保留 `model` 文件夹内的全部文件。

<details>
<summary>展开：画布尺寸、显存管理和停止任务</summary>

宽、高控制分层 PSD 和素材预览的实际画布尺寸。角色保持原图比例、缩放后居中，画布比例不同时补透明边。尺寸按 **64 像素**对齐，例如 `1000 × 1500` 调整为 `1024 × 1536`。

滑条默认范围为 **64～4096**；“按原图宽高填写”会在需要时扩大上限。内部拆层模型在正方形画布上计算，边长取所选宽、高中较大的值，拆层后统一去掉补边。贴图尺寸由绑模工具打包决定。

转换期间会通过 WebUI 的模型管理接口临时释放显存，并在任务结束后调用恢复接口。“停止”会终止本任务及其转换子进程，中间文件会保留。如果正在等待已有 SD 出图任务，则需要等出图任务释放任务锁后处理停止请求。

任务文件保存在转换环境的 `outputs/webui/时间-任务编号` 中。

</details>

<br>

<a id="viewer"></a>

## 查看生成的模型

推荐使用官方 [Cubism Viewer for Unity](https://www.live2d.com/zh-CHS/cubism/download/viewer-unity/)。

1. 完整解压模型文件包，保留 `model` 文件夹中的全部文件。
2. 在查看器的 **Config → Load Json** 中选择以 `.model3.json` 结尾的模型入口。
3. 在 **Parameters** 页面拖动参数，检查转头、眨眼、嘴形和部件遮挡。

[查看器操作说明](https://docs.live2d.com/zh-CHS/cubism-editor-manual/using-cubism-viewer-unity/)

| 输出内容 | 如何使用 |
| :--- | :--- |
| `.model3.json` 与配套运行时文件、贴图 | 在支持相应模型格式的软件中加载；已收到 Cubism Viewer 试用成功反馈 |
| 动作与物理文件 | 配合运行时模型查看动作效果 |
| `.cmo3` 工程 | 用于 Cubism Editor 编辑，兼容性取决于目标 Editor 版本 |
| 自动分层 PSD | 检查拆层结果，修改素材后重新导出 |
| 生成参数与转换检查报告 | 回看生成设置和检查结果 |

素材预览展示的是拆层合成效果，最终动作效果请以查看器为准。

<br>

## 当前效果与使用建议

已有样例完成图片到 Live2D 文件包的全流程，并收到**实际 Forge 运行、完整导出和 Cubism Viewer 试用成功**的反馈。自动生成结果适合作为继续精修的初版，效果随输入图片变化。

| 输入或输出情况 | 建议 |
| :--- | :--- |
| **正面二次元半身像**，五官清晰、背景简单 | 推荐作为起步输入 |
| 侧脸、复杂姿势、多人图片、复杂背景 | 拆层与补全难度更高，可能需要更多手动修正 |
| 闭眼、口腔、转头等原图未提供的细节 | 自动补全仍可能不准确；在查看器检查，并修改生成的 PSD 后重新导出 |

**目前推荐得到输出结果后自行修改自动生成的 PSD，再重新导出。** 检查眨眼、嘴形、头部角度、发丝遮挡和动作后，再决定如何使用模型。

<details>
<summary>展开：图层检查与已验证范围</summary>

在透明或均匀背景下，图层检查会参考原图过滤多余部件，并尝试恢复遗漏的可见嘴线。复杂背景会跳过部分过滤；恢复闭嘴线条不会自动生成隐藏的口腔。

此前一个半身样例有 **23 个可绘制对象、20 个参数**，通过官方 Cubism Core 的一致性检查、初始化和 **41 个参数姿态**的有限坐标检查。

reForge 实际集成、其他设备以及 CMO3 在官方 Editor 中编辑，仍需更多验证。转换环境存在官方 Cubism Core 和 Node.js 时，报告会记录运行库检查结果；缺少它们时，该项会标为未验证。

</details>

<br>

## 升级与常见问题

<details>
<summary>如何升级并保留环境和模型？</summary>

使用仓库安装时，可以通过 WebUI 扩展页面检查并应用更新，然后重启 WebUI。使用压缩包安装时，用新版覆盖 `extensions` 中的同名插件文件夹并重启。

保留原有环境和模型缓存。若使用原项目作为转换环境目录，也需要同步更新该目录中的引擎脚本。

</details>

<details>
<summary>安装提示“无法定位 PSD2Live Windows x64 便携版”</summary>

先更新插件，再检测环境。当前安装器兼容 `windows-x64`、`windows-x86_64` 和 `windows-amd64` 命名，优先下载普通便携版。

</details>

<br>

## 第三方项目与许可

| 项目 | 用途 | 许可说明 |
| :--- | :--- | :--- |
| [See-through](https://github.com/shitagaki-lab/see-through) | 单图拆层和遮挡补全 | 源码 Apache-2.0；模型权重按各模型仓库的 LICENSE 和 NOTICE 使用 |
| [PSD2Live](https://github.com/tsunehimatoi/psd2live) | 自动绑模与 Cubism 导出 | 主程序 GPL-3.0，部分组件 MIT；安装器保留上游许可文件 |
| [Live2D 官方 Cubism Core](https://cubism.live2d.com/sdk-web/cubismcore/live2dcubismcore.min.js) | 运行库检查 | 按官方许可使用，插件压缩包不包含该运行库 |

本项目负责环境检查、安装、流程编排、结果检查和界面。第三方源码与工具下载到 `third_party`，使用或再分发前请查看对应许可。

<details>
<summary>开发信息</summary>

插件仓库中的 `engine` 包含转换引擎源码，首次准备时会建立独立环境并下载第三方工具。此前已完成 Gradio **4.40.0 / 3.41.2** 接口测试。

</details>
