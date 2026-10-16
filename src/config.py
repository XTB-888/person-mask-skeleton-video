# -*- coding: utf-8 -*-
"""
person-mask-skeleton-video / 通用配置 config.py
================================================
三层优先级解析所有机器相关路径：
  1) 环境变量（前缀 PMSV_，避免与其他工具冲突）—— 用户显式覆盖
  2) 自动探测（shutil.which + glob 扫描常见安装位置）—— 开箱即用
  3) 清晰的错误提示 —— 探测不到时告诉用户该装什么、怎么设

所有其他脚本 `import config as C` 后直接使用：
  C.COMFY_PY, C.BLENDER_PY, C.FFMPEG, C.DET_PATH 等
"""
import os
import sys
import shutil
import glob
import numpy as np


# ============ 环境变量前缀 ============
# 所有可覆盖的路径都用这个前缀，见下方 _env_or_autodetect()
_ENV_PREFIX = "PMSV_"


# ============ 内部探测工具 ============
def _env(name):
    """读取带前缀的环境变量；未设返回空串。"""
    return os.environ.get(_ENV_PREFIX + name, "").strip()


def _autodetect_python_cuda():
    """寻找带 torch+CUDA 的 Python（ComfyUI portable python / venv / conda / 系统 Python）。"""
    # 1. 优先找 ComfyUI portable（Windows 用户最常见）
    win_paths = [
        # ComfyUI_windows_portable（用户自定义安装盘）
        r"D:\chorme\ComfyUI_windows_portable\python_embeded\python.exe",
        r"D:\ComfyUI_windows_portable\python_embeded\python.exe",
        r"C:\ComfyUI_windows_portable\python_embeded\python.exe",
        r"D:\ComfyUI\python_embeded\python.exe",
    ]
    # glob 搜索：所有盘根目录/Users 下的 ComfyUI_windows_portable
    for pattern in [
        r"D:\*\ComfyUI_windows_portable\python_embeded\python.exe",
        r"D:\ComfyUI*\python_embeded\python.exe",
        r"C:\Users\*\ComfyUI_windows_portable\python_embeded\python.exe",
        r"C:\Users\*\ComfyUI*\python_embeded\python.exe",
    ]:
        found = glob.glob(pattern)
        if found:
            win_paths.extend(found)
    # 2. 再试 PATH 里的 python
    win_paths.append(shutil.which("python") or "")
    win_paths.append(shutil.which("python3") or "")

    for p in win_paths:
        if p and os.path.exists(p):
            return p
    return ""


def _autodetect_python_cpu():
    """寻找带 onnxruntime+cv2 的 Python（Blender 内置 python / 系统 python / conda）。"""
    candidates = [
        # Blender 内置
        r"D:\Blender\5.2\python\bin\python.exe",
        r"C:\Program Files\Blender Foundation\Blender 5.2\python\bin\python.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.4\python\bin\python.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.3\python\bin\python.exe",
        # Blender 在其他盘
        r"D:\Blender Foundation\Blender 5.2\python\bin\python.exe",
        # 系统 PATH
        shutil.which("python") or "",
        shutil.which("python3") or "",
    ]
    # glob 搜索 Blender 所有盘
    for pattern in [
        r"C:\Program Files\Blender Foundation\*\python\bin\python.exe",
        r"D:\Blender*\python\bin\python.exe",
        r"D:\*\Blender*\python\bin\python.exe",
    ]:
        found = glob.glob(pattern)
        if found:
            candidates.extend(found)
    # Linux/macOS 兜底
    candidates.extend(["/usr/bin/python3", "/opt/homebrew/bin/python3"])

    for p in candidates:
        if p and os.path.exists(p):
            return p
    return ""


def _autodetect_ffmpeg():
    """PATH 里找 ffmpeg；找不到试常见 Windows 安装位置。"""
    p = shutil.which("ffmpeg") or ""
    if p:
        return p
    # WinGet 安装位置
    winget_glob = glob.glob(
        r"C:\Users\*\AppData\Local\Microsoft\WinGet\Packages\*\ffmpeg*\bin\ffmpeg.exe"
    )
    if winget_glob:
        return winget_glob[0]
    # 手动解压的常见位置
    for c in [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"D:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]:
        if os.path.exists(c):
            return c
    return ""


def _autodetect_onnx_model(model_name):
    """在常见 ComfyUI / controlnet_aux 安装位置搜索 DWPose ONNX。
    model_name: 'yolox_l.onnx' 或 'dw-ll_ucoco_384.onnx'。"""
    # ComfyUI ckpts 目录的 glob 搜索
    search_patterns = [
        # ComfyUI 标准位置（glob 接受正斜杠，避免 raw-string 末尾反斜杠转义问题）
        "D:/*/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/*/DWPose/" + model_name,
        "C:/*/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/*/DWPose/" + model_name,
        # 直接在 ComfyUI_windows_portable 下
        "D:/*/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/*/DWPose/" + model_name,
        # 任何盘根的 ckpts
        "D:/*/*/ckpts/*/DWPose/" + model_name,
        "C:/*/*/ckpts/*/DWPose/" + model_name,
    ]
    for pat in search_patterns:
        found = glob.glob(pat)
        if found:
            return found[0]
    # 兜底：从 Python site-packages controlnet_aux 找
    try:
        import controlnet_aux
        aux_dir = os.path.dirname(controlnet_aux.__file__)
        candidates = glob.glob(os.path.join(aux_dir, "ckpts", "**", "DWPose", model_name), recursive=True)
        if candidates:
            return candidates[0]
    except Exception:
        pass
    return ""


# ============ 解析器 ============
def _resolve(name, autodetect_fn=None, must_exist=True, desc=""):
    """按 环境变量 → 自动探测 → 缺失报错 的顺序解析一个配置项。"""
    val = _env(name)
    source = "env"
    if not val and autodetect_fn is not None:
        val = autodetect_fn()
        source = "autodetect" if val else "not_found"
    if must_exist and val and not os.path.exists(val):
        print(f"[config] WARNING: {desc or name} 路径不存在: {val} (来源={source})", file=sys.stderr)
    if not val:
        msg = (
            f"\n{'='*60}\n"
            f"person-mask-skeleton-video 未找到: {desc or name}\n"
            f"{'='*60}\n"
        )
        if name == "COMFY_PY":
            msg += (
                "需要带 torch + CUDA 的 Python。推荐安装:\n"
                "  1) ComfyUI_windows_portable（Windows 最省事）:\n"
                "     https://github.com/comfyanonymous/ComfyUI/releases\n"
                "  2) 或 conda create -n comfy python=3.12 && conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia\n"
                "然后设置环境变量:\n"
                "  $env:PMSV_COMFY_PY = 'D:\\path\\to\\python.exe'\n"
            )
        elif name == "BLENDER_PY":
            msg += (
                "推荐安装 Blender（内置 Python 自带 onnxruntime + cv2）:\n"
                "  https://www.blender.org/download/\n"
                "然后设置环境变量:\n"
                "  $env:PMSV_BLENDER_PY = 'C:\\Program Files\\Blender Foundation\\Blender 5.2\\python\\bin\\python.exe'\n"
                "或用系统 Python: pip install opencv-python onnxruntime\n"
            )
        elif name in ("FFMPEG", "FFPROBE"):
            msg += (
                "推荐用 WinGet 安装:\n"
                "  winget install Gyan.FFmpeg.Essentials\n"
                "或手动下载: https://www.gyan.dev/ffmpeg/builds/\n"
                "然后确保 ffmpeg/ffprobe 在 PATH 里，或:\n"
                "  $env:PMSV_FFMPEG = 'C:\\path\\to\\ffmpeg.exe'\n"
                "  $env:PMSV_FFPROBE = 'C:\\path\\to\\ffprobe.exe'\n"
            )
        elif name == "DET_PATH":
            msg += (
                "DWPose ONNX 模型（yolox_l.onnx + dw-ll_ucoco_384.onnx）\n"
                "推荐通过 ComfyUI ControlNet Aux 下载:\n"
                "  https://github.com/Fannovel16/comfyui_controlnet_aux\n"
                "把两个 .onnx 放到同目录后设置:\n"
                "  $env:PMSV_DET_PATH = 'C:\\path\\to\\yolox_l.onnx'\n"
                "  $env:PMSV_POSE_PATH = 'C:\\path\\to\\dw-ll_ucoco_384.onnx'\n"
            )
        elif name == "POSE_PATH":
            msg += "（见 DET_PATH 的提示）\n"
        msg += f"{'='*60}\n"
        if must_exist:
            raise SystemExit(msg)
    return val


# ============ 对外暴露的配置变量 ============

# -- Python 解释器（两个，必须分开，因为 CUDA 版 torch 和 CPU 版 onnxruntime 难共存）
COMFY_PY   = _resolve("COMFY_PY",   _autodetect_python_cuda,   desc="GPU Python (torch + CUDA)")
BLENDER_PY = _resolve("BLENDER_PY", _autodetect_python_cpu,    desc="CPU Python (onnxruntime + cv2)")

# -- FFmpeg / FFprobe
FFMPEG  = _resolve("FFMPEG",  _autodetect_ffmpeg,  desc="ffmpeg.exe")
def _ffprobe_from_ffmpeg():
    """从 ffmpeg.exe 路径推断 ffprobe.exe（Windows 大小写不敏感）。"""
    try:
        import re
        return re.sub(r"(?i)ffmpeg(\.exe)?$", r"ffprobe\1", FFMPEG)
    except Exception:
        return os.path.join(os.path.dirname(FFMPEG), "ffprobe.exe")

FFPROBE = _resolve("FFPROBE", _ffprobe_from_ffmpeg, desc="ffprobe.exe")

# -- DWPose ONNX 模型
DET_PATH  = _resolve("DET_PATH",  lambda: _autodetect_onnx_model("yolox_l.onnx"),        desc="DWPose 检测 ONNX (yolox_l.onnx)")
POSE_PATH = _resolve("POSE_PATH", lambda: _autodetect_onnx_model("dw-ll_ucoco_384.onnx"), desc="DWPose 姿态 ONNX (dw-ll_ucoco_384.onnx)")

# -- BiRefNet（HuggingFace 模型标识符，无需本地路径）
BIRENET_MODEL = os.environ.get(_ENV_PREFIX + "BIRENET_MODEL", "ZhengPeng7/BiRefNet-portrait")

# ============ 处理参数（全部可用环境变量覆盖，也可命令行覆盖）============
def _env_float(name, default):
    v = _env(name)
    return float(v) if v else default

def _env_int(name, default):
    v = _env(name)
    return int(v) if v else default

SPLIT          = _env_float("SPLIT", 14.0)       # 超过此秒数自动分段（0=不拆）
SEG_SIZE       = _env_int("SEG_SIZE", 512)        # BiRefNet 分割尺寸 (512快 / 1024精细)
MAX_FRAME_WIDTH = _env_int("MAX_FRAME_WIDTH", 1080)
OUTPUT_FPS     = _env_float("OUTPUT_FPS", 30.0)
BG_MODES       = os.environ.get(_ENV_PREFIX + "BG_MODES", "black,background").split(",")

# 多人物编号（默认关闭 = 干净版，适合二次生成）
NUMBERS = os.environ.get(_ENV_PREFIX + "NUMBERS", "0") in ("1", "true", "True", "yes")

# 朝向/转身识别（默认开启，叠加进成品骨架）
ORIENTATION = os.environ.get(_ENV_PREFIX + "ORIENTATION", "1") in ("1", "true", "True", "yes")

# ============ 模型归一化（BiRefNet 预处理）============
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============ 启动时打印解析结果 ============
def _print_resolved():
    print("[config] 已解析:", flush=True)
    print(f"  COMFY_PY   = {COMFY_PY}")
    print(f"  BLENDER_PY = {BLENDER_PY}")
    print(f"  FFMPEG     = {FFMPEG}")
    print(f"  FFPROBE    = {FFPROBE}")
    print(f"  DET_PATH   = {DET_PATH}")
    print(f"  POSE_PATH  = {POSE_PATH}")
    print(f"  BIRENET_MODEL = {BIRENET_MODEL}")
    print(f"  SPLIT={SPLIT}  SEG_SIZE={SEG_SIZE}  BG_MODES={BG_MODES}", flush=True)


# 直接 import 就打印一次（不影响静默 import，因为后续脚本可能不需要此输出）
if os.environ.get(_ENV_PREFIX + "QUIET", "0") not in ("1", "true", "True"):
    # 仅在主脚本入口打印，避免每个子脚本都重复
    if sys.argv and sys.argv[0].endswith("run.py"):
        _print_resolved()
