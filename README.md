# person-mask-skeleton-video

> 一键把普通视频变成「**人物白色剪影 + 彩色姿态骨架**」的叠加视频，支持按秒分段、双背景模式、朝向识别、独立音轨。

## 效果说明

整条流水线自动完成：

```
BiRefNet 人物分割 (GPU)   →  全片白色人物剪影（黑底）
DWPose   骨架检测 (CPU)   →  全片彩色骨架（黑底）
像素级叠加 + 按秒分段 + 双背景模式 + 音轨混入
```

产出两种版本（可全要、也可只选其一）：

| 模式 | 效果 | 适用 |
|------|------|------|
| **black** | 纯黑底 + 白色剪影 + 彩色骨架 | 喂给 AI 视频生成/重绘模型做动作参考 |
| **background** | 在原视频画面上覆盖剪影/骨架 | 人眼检查动作识别精度 |

**默认开启人物朝向识别**：骨架上自动叠加朝向箭头 + 标签（正面 / 左侧身 / 右侧身 / 背面），方便 AI 二次生成时理解人物转身。

---

## 快速开始（Windows）

### 一键运行

```powershell
# 用 CPU 解释器跑主控脚本即可，它会自动调用 GPU 解释器
& "C:\path\to\your\python.exe" src\run.py --input "你的视频.mp4"
```

> `config.py` 会自动探测本机常见安装位置（ComfyUI / Blender / WinGet 的 ffmpeg / controlnet_aux 的 ONNX 模型）。
> 探测不到时会给出明确提示，告诉你装什么、或怎么设环境变量（`PMSV_*` 前缀）。

### 输出示例

假设输入 `下载.mp4`、时长 9 秒（未触发分段）：

```
下载_out/
├── 下载_seg01_black_sound.mp4       # 黑底剪影骨架 + 原音轨
├── 下载_seg01_background_sound.mp4  # 保留背景 + 原音轨
├── 下载_mask.mp4                    # 中间产物：BiRefNet 剪影（可复用）
├── 下载_pose.mp4                    # 中间产物：DWPose 骨架（可复用）
└── 下载_kp.npy                      # 中间产物：关键点数据（供朝向识别）
```

---

## 安装

### 为什么需要两个 Python 环境？

| 环境 | 用途 | 关键依赖 |
|------|------|---------|
| **GPU 环境** `COMFY_PY` | BiRefNet 人物分割 | torch + CUDA + transformers |
| **CPU 环境** `BLENDER_PY` | DWPose 骨架 + 像素合成 | onnxruntime + opencv + numpy |

> 拆开的原因：torch CUDA 和 onnxruntime 在同一环境里容易争抢显存。
> 实在只有一个 Python 也行：把两套依赖都装上，然后把 `PMSV_COMFY_PY` 和 `PMSV_BLENDER_PY` 设成同一个解释器（BiRefNet 仍走 GPU，DWPose 走 CPU）。

### 步骤 1：装 GPU 环境（BiRefNet）

**方案 A：ComfyUI_windows_portable（最省事，Windows 推荐）**

1. 到 https://github.com/comfyanonymous/ComfyUI/releases 下载带 `portable` 的 zip
2. 解压（例：`D:\ComfyUI_windows_portable`）
3. 补装分割模型依赖：
   ```powershell
   & "D:\ComfyUI_windows_portable\python_embeded\python.exe" -m pip install transformers Pillow huggingface-hub
   ```

**方案 B：conda（任意平台）**

```bash
conda create -n pmsv-gpu python=3.12
conda activate pmsv-gpu
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers Pillow huggingface-hub opencv-python
```

### 步骤 2：装 CPU 环境（DWPose + 合成）

**方案 A：Blender 内置 Python（推荐，自带 opencv/numpy）**

1. 到 https://www.blender.org/download/ 下载并安装 Blender
2. 找内置 Python 路径（Windows 例：`C:\Program Files\Blender Foundation\Blender 5.2\python\bin\python.exe`）
3. 补装 onnxruntime：
   ```powershell
   & "C:\Program Files\Blender Foundation\Blender 5.2\python\bin\python.exe" -m pip install onnxruntime
   ```

**方案 B：系统 Python / conda**

```bash
pip install onnxruntime opencv-python numpy
```

### 步骤 3：装 FFmpeg

- Windows：`winget install Gyan.FFmpeg.Essentials`
- Linux：`sudo apt install ffmpeg`
- macOS：`brew install ffmpeg`

### 步骤 4：准备模型文件

| 模型 | 来源 | 大小 |
|------|------|------|
| BiRefNet（分割） | HuggingFace 首次运行时自动下载 `ZhengPeng7/BiRefNet-portrait` | ~200MB |
| DWPose 检测器 `yolox_l.onnx` | ComfyUI ControlNet Aux 插件自动下载，或 https://huggingface.co/yzd-v/DWPose | ~200MB |
| DWPose 姿态 `dw-ll_ucoco_384.onnx` | 同上 | ~350MB |

国内网络下载 HuggingFace 慢时：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

---

## 命令行参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--input` | （必填） | 源视频路径 |
| `--outdir` | `<视频名>_out` | 输出目录 |
| `--bg` | `both` | 背景模式：`black` / `background` / `both` |
| `--split` | 14.0 | 超过此秒数自动分段；`0` = 不拆 |
| `--seg` | 512 | BiRefNet 分割尺寸；`1024` 更精细但慢 2-3 倍 |
| `--reuse-mask` | 无 | 复用已有 mask 视频，跳过分割 |
| `--reuse-pose` | 无 | 复用已有骨架，跳过检测 |
| `--only-overlay` | 关 | 只跑合成（需同时给 `--reuse-mask` 和 `--reuse-pose`） |
| `--no-audio` | 关 | 不混入原音轨 |
| `--no-orientation` | 关 | 关闭朝向识别（输出干净骨架，适合二次生成） |
| `--numbers` | 关 | 多人物稳定身份色（区分多个人物） |
| `--fill-mask` | 关 | 空白剪影帧时间补全（解决动作快时剪影闪烁） |
| `--dilate` | 3 | 剪影边缘外扩像素（配合 `--fill-mask`） |
| `--edge` | 关 | 给人物剪影画轮廓描边 |
| `--edge-color` | `235,235,235` | 描边颜色 RGB |

### 常用命令示例

```powershell
# 标准一键处理
python src\run.py --input SRC.mp4

# 高精度（发丝/手指更准，但慢 2-3 倍）
python src\run.py --input SRC.mp4 --seg 1024

# 动作快导致剪影闪烁 → 时间补全 + 外扩
python src\run.py --input SRC.mp4 --fill-mask --dilate 3

# 喂给 AI 视频生成模型 → 干净骨架（无朝向文字）
python src\run.py --input SRC.mp4 --no-orientation

# 多人视频 → 给每个人分配身份色圆点
python src\run.py --input SRC.mp4 --numbers

# 复用已有 mask/pose 改参数重跑合成（最快）
python src\run.py --input SRC.mp4 --reuse-mask OUT/mask.mp4 --reuse-pose OUT/pose.mp4
```

---

## 环境变量（全部可选）

| 变量 | 说明 |
|------|------|
| `PMSV_COMFY_PY` | GPU Python 路径（torch + CUDA） |
| `PMSV_BLENDER_PY` | CPU Python 路径（onnxruntime + cv2） |
| `PMSV_FFMPEG` / `PMSV_FFPROBE` | ffmpeg / ffprobe 路径 |
| `PMSV_DET_PATH` / `PMSV_POSE_PATH` | DWPose 两个 ONNX 模型路径 |
| `PMSV_BIRENET_MODEL` | BiRefNet HuggingFace 模型名（默认 `ZhengPeng7/BiRefNet-portrait`） |
| `PMSV_SPLIT` | 分段秒数（默认 14） |
| `PMSV_SEG_SIZE` | BiRefNet 输入尺寸（默认 512） |
| `PMSV_HF_OFFLINE` | 设 `1` 强制离线模式（已缓存完模型可加速启动） |
| `PMSV_QUIET` | 设 `1` 关闭 config 启动时的探测信息打印 |

---

## 性能参考（Quadro P2000 5GB GPU / 1080×1920 @ 30fps）

| 步骤 | 速度 | 9.3 秒视频耗时 |
|------|------|---------------|
| BiRefNet（SEG=512） | ~1.25 fps | ≈ 3.7 分钟 |
| DWPose（CPU） | ~2.5 fps | ≈ 1.9 分钟 |
| 叠加 + 分段 + 音轨 | 快 | ≈ 30 秒 |

**显存提示**：5GB GPU 用 `SEG_SIZE=512` 是平衡点；`1024` 可显著提升发丝/手指精度但耗时 2-3 倍。

---

## 常见问题

**Q: 启动时报"未找到 GPU Python"？**
A: 自动探测没扫到你的环境。手动设：
```powershell
$env:PMSV_COMFY_PY = "D:\你的\路径\python.exe"
```

**Q: BiRefNet 报"未检测到 CUDA"？**
A: 说明用了 CPU 解释器跑分割。确认 `PMSV_COMFY_PY` 指向的 Python 里 `torch.cuda.is_available()` 返回 `True`。

**Q: DWPose 报找不到 onnxruntime？**
A: 对 CPU 解释器装一下：
```powershell
& "C:\你的\python.exe" -m pip install onnxruntime
```

**Q: 人物动作快时剪影闪没？**
A: 加 `--fill-mask --dilate 3`，前后有效剪影会插值填补空白帧。

**Q: 想用同一套 Python？**
A: 把 `torch(CUDA) + transformers + onnxruntime + opencv` 全装一个 venv 里，然后把 `PMSV_COMFY_PY` 和 `PMSV_BLENDER_PY` 设成同一个解释器。

---

## 项目结构

```
person-mask-skeleton-video/
├── src/
│   ├── config.py                # 通用配置（环境变量 + 自动探测）
│   ├── run.py                   # 主控：一键处理全流程
│   ├── mask_gpu.py              # 步骤1: BiRefNet 人物分割 (GPU)
│   ├── pose_dwpose.py           # 步骤2: DWPose 骨架检测 (CPU)
│   ├── overlay_seg.py           # 步骤3: 像素叠加 + 分段 + 双背景 + 音轨
│   ├── mask_fill.py             # 可选: 空白剪影帧时间补全
│   └── orientation_overlay.py   # 朝向预览层独立工具
├── requirements-gpu.txt         # GPU 环境依赖
├── requirements-cpu.txt         # CPU 环境依赖
├── .gitignore
└── README.md
```

---

## 许可证

MIT
