# person-mask-skeleton-video

把一段普通视频处理成「**人物白色剪影 + 彩色姿态骨架**」的叠加视频，整条流水线可复用：

```
BiRefNet 人物分割 (GPU)  →  全片白色剪影 mask
DWPose   骨架检测 (CPU)  →  全片彩色骨架
像素级叠加 + 按秒分段 + 双背景模式 + 独立音轨
```

最终产出两种版本：
- **black** 纯黑底版：人物白剪影 + 彩色骨架
- **background** 保留原背景版：剪影/骨架覆盖在原视频上

另外默认开启**人物朝向识别**（正对 / 左侧身 / 右侧身 / 背对），骨架上自动叠加朝向箭头和角度标签。

---

## 效果预览

（建议放 GIF 演示）

---

## 为什么需要两个 Python 环境？

| 环境 | 解释器 | 负责 | 必须的依赖 |
|------|--------|------|-----------|
| **GPU 环境** (COMFY_PY) | 带 CUDA 的 torch | BiRefNet 分割（GPU 推理才快） | `torch` + CUDA / `transformers` / `Pillow` |
| **CPU 环境** (BLENDER_PY) | 任意 Python | DWPose 骨架 + 叠加合成 | `onnxruntime` / `opencv-python` / `numpy` |

**这两步可以并行跑，也可以在同一台机器上用两个解释器。** 最省事的组合：
- GPU 环境 = **ComfyUI_windows_portable**（已自带 torch + CUDA，下载即用）
- CPU 环境 = **Blender 内置 Python**（已自带 opencv + numpy，只需 pip install onnxruntime）

如果你只有一套 Python，也可以：
- 把两套依赖装到同一个 conda venv（需要该 Python 同时能跑 torch CUDA 和 onnxruntime）
- 或设 `PMSV_SAME_PY=1` 让两个步骤都用同一个解释器（BiRefNet 会自动 fallback 到 CPU，极慢但应急可用）

---

## 安装步骤

### 1. GPU 环境（BiRefNet 分割）

**方案 A：ComfyUI_windows_portable（Windows 推荐）**
1. 下载 [ComfyUI release](https://github.com/comfyanonymous/ComfyUI/releases) （带 `portable` 字样的 zip）
2. 解压到任意盘（如 `D:\ComfyUI_windows_portable`）
3. 无需额外安装 torch，已自带
4. 安装 transformers：
   ```powershell
   & "D:\ComfyUI_windows_portable\python_embeded\python.exe" -m pip install transformers Pillow huggingface-hub
   ```

**方案 B：conda（Linux/macOS/想要 GPU 但不想装 ComfyUI）**
```bash
conda create -n pmsv-gpu python=3.12
conda activate pmsv-gpu
# CUDA 12.1（最常见 NVIDIA 驱动）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8（旧驱动）
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers Pillow huggingface-hub opencv-python
```

### 2. CPU 环境（DWPose 骨架 + 合成）

**方案 A：Blender（推荐，内置完整 Python）**
1. 下载 [Blender](https://www.blender.org/download/)
2. 安装后，找内置 Python 路径（Windows 例：`C:\Program Files\Blender Foundation\Blender 5.2\python\bin\python.exe`）
3. 安装 onnxruntime：
   ```powershell
   & "C:\Program Files\Blender Foundation\Blender 5.2\python\bin\python.exe" -m pip install onnxruntime
   ```
   （opencv / numpy / pip Blender 内置 Python 已自带）

**方案 B：系统 Python / conda**
```bash
pip install onnxruntime opencv-python numpy
```

### 3. FFmpeg

- **Windows（推荐 WinGet）：**
  ```powershell
  winget install Gyan.FFmpeg.Essentials
  # 关闭终端再开，ffmpeg / ffprobe 会进 PATH
  ```
- **Linux：** `sudo apt install ffmpeg`
- **macOS：** `brew install ffmpeg`

### 4. DWPose ONNX 模型文件

需要两个 `.onnx` 文件：
- `yolox_l.onnx` （检测器）
- `dw-ll_ucoco_384.onnx` （姿态模型）

最省事：**先装好 ComfyUI + ComfyUI ControlNet Aux 节点**（ComfyUI Manager 搜 `comfyui-controlnet-aux`），插件会自动下载这两个文件。然后本项目会自动搜索其位置。

手动下载链接（如不想装 ComfyUI）：
- https://huggingface.co/yzd-v/DWPose/tree/main

### 5. BiRefNet（HuggingFace，自动下载）

首次运行时会自动从 HuggingFace 下载约 200MB 的模型权重到本地缓存。需保证网络通畅（或设置 `HF_ENDPOINT=https://hf-mirror.com` 使用国内镜像）。

---

## 快速开始

```powershell
# 1) 克隆 & 进入项目
git clone https://github.com/<your-user>/person-mask-skeleton-video.git
cd person-mask-skeleton-video

# 2) 确认两个解释器能被找到（config.py 会自动探测常见安装路径）
# 如果探测不到，用环境变量显式指定（前缀 PMSV_）：
$env:PMSV_COMFY_PY  = "D:\ComfyUI_windows_portable\python_embeded\python.exe"
$env:PMSV_BLENDER_PY = "C:\Program Files\Blender Foundation\Blender 5.2\python\bin\python.exe"
# ONNX 模型也可显式指定：
$env:PMSV_DET_PATH  = "D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfyui_controlnet_aux\ckpts\yzd-v\DWPose\yolox_l.onnx"
$env:PMSV_POSE_PATH = "D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\comfyui_controlnet_aux\ckpts\yzd-v\DWPose\dw-ll_ucoco_384.onnx"

# 3) 一键处理（用 CPU 环境跑主控脚本）
& $env:PMSV_BLENDER_PY "src\run.py" --input "C:\your\video.mp4"
```

输出到 `<视频名>_out/`：
```
<视频名>_seg01_black_sound.mp4       # 纯黑底 + 原音轨
<视频名>_seg01_background_sound.mp4  # 保留原背景 + 原音轨
<视频名>_mask.mp4                    # 中间：BiRefNet 剪影
<视频名>_pose.mp4                    # 中间：DWPose 骨架
```

### 全量参数示例

```powershell
# 两种背景都要、按 14 秒分段、开启剪影补全 + 边缘外扩 + 人体轮廓描边
& $env:PMSV_BLENDER_PY "src\run.py" --input SRC.mp4 --bg both --split 14 --fill-mask --dilate 3 --edge

# 复用已有 mask / pose 层（调参数重跑叠加最快）
& $env:PMSV_BLENDER_PY "src\run.py" --input SRC.mp4 --reuse-mask OUT/mask.mp4 --reuse-pose OUT/pose.mp4

# 关闭朝向识别（输出干净骨架，适合喂给视频生成模型二次创作）
& $env:PMSV_BLENDER_PY "src\run.py" --input SRC.mp4 --no-orientation

# 多人物视频：给每个人稳定身份色（可选，默认关闭）
& $env:PMSV_BLENDER_PY "src\run.py" --input SRC.mp4 --numbers
```

---

## 环境变量参考（全部可选）

| 变量 | 说明 | 自动探测 |
|------|------|---------|
| `PMSV_COMFY_PY` | GPU Python (torch+CUDA) | ✅ 搜索 ComfyUI / 系统 PATH |
| `PMSV_BLENDER_PY` | CPU Python (onnxruntime+cv2) | ✅ 搜索 Blender / 系统 PATH |
| `PMSV_FFMPEG` | ffmpeg.exe | ✅ 搜索 PATH / WinGet |
| `PMSV_FFPROBE` | ffprobe.exe | ✅ 默认从 FFMPEG 路径替换 |
| `PMSV_DET_PATH` | DWPose yolox_l.onnx | ✅ 搜索 ComfyUI ckpts |
| `PMSV_POSE_PATH` | DWPose dw-ll_ucoco_384.onnx | ✅ 搜索 ComfyUI ckpts |
| `PMSV_BIRENET_MODEL` | BiRefNet 模型名 | 默认 `ZhengPeng7/BiRefNet-portrait` |
| `PMSV_SPLIT` | 分段秒数（0=不拆） | 默认 14.0 |
| `PMSV_SEG_SIZE` | BiRefNet 输入尺寸 | 默认 512（1024 更精细） |
| `PMSV_QUIET` | 静默 config 打印 | 设 1 关闭启动信息 |

---

## 命令行参数（run.py 全表）

```
--input            源视频路径（必填）
--outdir           输出目录（默认 <源视频同目录>_out）
--bg               背景模式: black / background / both (默认 both)
--split            分段秒数，0=不拆（默认 14，可用 0 关闭）
--seg              BiRefNet 分割尺寸（默认 512；1024 更精细但慢 2-3 倍）
--reuse-mask       复用已有 mask 视频，跳过分割
--reuse-pose       复用已有骨架视频，跳过骨架
--only-overlay     只跑叠加（需同时提供 --reuse-mask + --reuse-pose）
--no-audio         不混入原音轨
--no-orientation   关闭朝向识别预览层（默认开启）
--numbers          多人物稳定身份色（默认关闭）
--fill-mask        对空白剪影帧做时间补全
--dilate           剪影边缘外扩像素（配合 --fill-mask，默认 3）
--edge             给人物边缘画轮廓描边
--edge-color       轮廓描边颜色 RGB（默认浅灰 235,235,235）
```

---

## 性能参考（参考配置：Quadro P2000 5GB GPU / 30fps / 竖屏 1080×1920）

| 步骤 | 速度 | 耗时估算 | 说明 |
|------|------|---------|------|
| BiRefNet (SEG=512) | ~1.25 fps | 9.3s 视频 ≈ 3.7min | GPU 关键；SEG=1024 慢 2-3x 但发丝/手指更准 |
| DWPose (CPU) | ~2.5 fps | 9.3s 视频 ≈ 1.9min | 全片 CPU 可接受；onnxruntime-gpu 会更快 |
| 叠加 + 分段 + 音轨 | 快 | 9.3s 视频 ≈ 30s | 瓶颈不在这步 |

**单帧处理时间 ≈ 0.8s (GPU) + 0.4s (CPU)**

---

## 常见问题排查

**Q: `config.py` 启动就报错"未找到 GPU Python"？**
A: 自动探测只扫了常见路径。确认一下你 GPU Python 的实际位置，然后设：
```powershell
$env:PMSV_COMFY_PY = "D:\实际\路径\python.exe"
```

**Q: BiRefNet 卡在"未检测到 CUDA，速度会非常慢"？**
A: 说明用的是 CPU 解释器跑分割了。确认 `COMFY_PY` 指向的 Python 里 `import torch; torch.cuda.is_available()` 返回 `True`。ComfyUI portable python 自带正确的 torch CUDA。

**Q: DWPose 报找不到 onnxruntime？**
A: 对 Blender 内置 Python 执行一次：
```powershell
& "C:\...python.exe" -m pip install onnxruntime
```
Blender 内置 pip 默认源有时失效，加 `-i https://pypi.tuna.tsinghua.edu.cn/simple` 换国内镜像。

**Q: HuggingFace 下载 BiRefNet 模型慢/失败？**
A: 设镜像：
```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

**Q: BiRefNet 某些帧剪影空白（人物动起来就消失）？**
A: 开 `--fill-mask --dilate 3`。脚本会用前后有效剪影插值填补空白帧并外扩边缘。

**Q: 两个 Python 环境能合并成一个吗？**
A: 技术上可行（装一套 Python 同时带 torch CUDA + onnxruntime），但 onnxruntime 在有 CUDA 的 torch 环境下容易跟 torch 抢 GPU 显存。建议还是分两个解释器。

**Q: 只有一套 Python，不想装 ComfyUI 也不想装 Blender？**
A: 把所有依赖装到同一个 conda venv，然后设 `PMSV_COMFY_PY` 和 `PMSV_BLENDER_PY` 都指向这个 venv 的 python（两个步骤都能跑，BiRefNet 用 GPU，DWPose 用 CPU）。

---

## 目录结构

```
person-mask-skeleton-video/
├── src/
│   ├── config.py           # 通用配置（环境变量 + 自动探测）
│   ├── run.py              # 主控编排：一键处理全流程
│   ├── mask_gpu.py         # 步骤1: BiRefNet 人物分割 (GPU)
│   ├── pose_dwpose.py      # 步骤2: DWPose 骨架检测 (CPU, 可选导出关键点)
│   ├── overlay_seg.py      # 步骤3: 像素叠加 + 分段 + 双背景 + 音轨
│   ├── mask_fill.py        # 可选: 空白剪影帧时间补全
│   └── orientation_overlay.py  # 朝向预览层（已集成进 overlay_seg，保留作独立工具）
├── requirements-gpu.txt    # GPU 环境依赖
├── requirements-cpu.txt    # CPU 环境依赖
├── .gitignore
└── README.md
```

---

## License

MIT
