# -*- coding: utf-8 -*-
"""
person-mask-skeleton-video / step1: 人物分割 mask（BiRefNet, GPU）
=================================================================
输入一段视频，输出白色人物剪影 + 黑底 mask 视频（人物=白,背景=黑）。
必须用 config.COMFY_PY 环境运行（torch + CUDA），否则 GPU 起不来会很慢。

用法（由主控 run.py 调用或单独执行）:
  COMFY_PY mask_gpu.py --input IN.mp4 --output OUT.mp4
可选: --seg 512 | --maxw 1080 | --frames 0(0=全部)
"""
import os, sys, time, argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# BiRefNet 模型首次运行需从 HuggingFace 下载（约 200MB）。
# 若已离线缓存完成，可显式设 PMSV_HF_OFFLINE=1 加速启动。
# 国内网络可额外设 $env:HF_ENDPOINT='https://hf-mirror.com' 使用镜像。
if os.environ.get("PMSV_HF_OFFLINE", "0") in ("1", "true", "True"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
import cv2, numpy as np, torch
from PIL import Image
from transformers import AutoModelForImageSegmentation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


def preprocess(frame_bgr, seg_size):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((seg_size, seg_size), Image.BILINEAR)
    img = np.array(pil, dtype=np.float32) / 255.0
    img = (img - C.IMAGENET_MEAN) / C.IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))[None, ...].astype(np.float32)
    return torch.from_numpy(img), (h, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seg", type=int, default=C.SEG_SIZE)
    ap.add_argument("--maxw", type=int, default=C.MAX_FRAME_WIDTH)
    ap.add_argument("--frames", type=int, default=0, help="0=全部")
    ap.add_argument("--gpu-only", action="store_true")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("警告: 未检测到 CUDA，速度会非常慢")

    model = AutoModelForImageSegmentation.from_pretrained(C.BIRENET_MODEL, trust_remote_code=True)
    model.to(device).eval()
    dt = next(model.parameters()).dtype

    cap = cv2.VideoCapture(a.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or C.OUTPUT_FPS
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if a.maxw > 0 and ow > a.maxw:
        pw = a.maxw; ph = int(oh * pw / ow)
    else:
        pw, ph = ow, oh
    max_f = total if a.frames <= 0 else min(a.frames, total)
    print(f"输入 {ow}x{oh} @{fps:.1f}fps {total}帧 | 处理 {pw}x{ph} seg={a.seg} 输出{max_f}帧")

    out = cv2.VideoWriter(a.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (ow, oh), True)
    import subprocess  # noqa
    t0 = time.time(); n_missing = 0
    for idx in range(max_f):
        ok, frame = cap.read()
        if not ok: break
        fs = frame if pw == ow else cv2.resize(frame, (pw, ph))
        t, _ = preprocess(fs, a.seg)
        with torch.no_grad():
            pred = model(t.to(device).to(dt))[-1].sigmoid()
        m = cv2.resize(pred.squeeze().cpu().float().numpy(), (ow, oh))
        out.write(cv2.cvtColor((m * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        if (idx + 1) % 10 == 0 or idx + 1 == max_f:
            el = time.time() - t0
            print(f" 帧 {idx+1}/{max_f} | { (idx+1)/el:.2f} fps | ETA {(max_f-idx-1)/max((idx+1)/el,0.01):.0f}s", flush=True)
    cap.release(); out.release()
    print(f"mask 完成: {max_f}帧, {time.time()-t0:.1f}s -> {a.output}")


if __name__ == "__main__":
    main()
