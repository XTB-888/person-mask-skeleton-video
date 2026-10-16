# -*- coding: utf-8 -*-
"""
person-mask-skeleton-video / 主控编排脚本 run.py
=================================================================
一键处理流程：
  1. 探测源视频时长，判断是否需要分段（> SPLIT 秒才拆，余段保留）
  2. BiRefNet 人物分割（GPU, COMFY_PY 环境）-> 全片白色剪影 mask
  3. DWPose 骨架检测（CPU, BLENDER_PY 环境）-> 全片彩色骨架
  4. 像素级叠加 + 按秒分段 + 双背景模式（black/background）+ 独立音轨

用法:
  BLENDER_PY run.py --input 视频路径 [--outdir 输出目录] [--bg both] [--split 14]
  BLENDER_PY run.py --input IN --reuse-mask MASK.mp4 --reuse-pose POSE.mp4   # 复用已有层
  BLENDER_PY run.py --input IN --outdir OUT --only-overlay --mask M --pose P  # 仅合成

说明：
  - 全片只跑一次模型（单帧率高），分段在最终合成阶段完成，避免逐段重跑 GPU/CPU。
  - mask 用 COMFY_PY(torch+CUDA)，pose 与合成用 BLENDER_PY(onnxruntime+cv2)。
  - 两个 python 都可直接运行本脚本，脚本内部会按阶段选择正确解释器。
"""
import os
import sys
import subprocess
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

# 本脚本统一用 BLENDER_PY(cv2+onnxruntime) 跑控制逻辑；模型阶段用对应解释器
HERE = os.path.dirname(os.path.abspath(__file__))


def run_with(interp, script, args, label):
    cmd = [interp, os.path.join(HERE, script)] + [str(a) for a in args]
    print(f"\n=== [{label}] {' '.join(cmd)} ===", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"[{label}] 失败: {' '.join(cmd)}")
    return r


def probe_duration(video):
    """返回 (秒数, 宽, 高, 帧率)。复用 ffprobe。"""
    r = subprocess.run([C.FFPROBE, "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,avg_frame_rate",
                        "-show_entries", "format=duration",
                        "-of", "csv=p=0", video], capture_output=True, text=True)
    lines = [l for l in r.stdout.strip().split("\n") if l]
    # 输出形如: "1920,1080,30000/1001,17.8"  (宽,高,帧率,时长)
    w, h = int(lines[0].split(",")[0]), int(lines[0].split(",")[1])
    fr = lines[0].split(",")[2]
    dur = float(lines[-1])
    if "/" in fr:
        num, den = fr.split("/")
        fps = float(num) / float(den) if float(den) else 30.0
    else:
        fps = float(fr) or 30.0
    return dur, w, h, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="源视频路径")
    ap.add_argument("--outdir", default="", help="输出目录（默认与源视频同目录 <name>_out）")
    ap.add_argument("--bg", default="both",
                    choices=["black", "background", "both"],
                    help="背景模式: black纯黑 / background保留原背景 / both都要(默认)")
    ap.add_argument("--split", type=float, default=C.SPLIT, help="分段秒数(0=不拆)")
    ap.add_argument("--seg", type=int, default=C.SEG_SIZE, help="BiRefNet 分割尺寸")
    ap.add_argument("--reuse-mask", default="", help="复用已有 mask 视频，跳过分割")
    ap.add_argument("--reuse-pose", default="", help="复用已有骨架视频，跳过骨架")
    ap.add_argument("--only-overlay", action="store_true",
                    help="只执行合成阶段（需 --mask --pose）")
    ap.add_argument("--no-audio", action="store_true", help="不混音轨")
    ap.add_argument("--numbers", action="store_true",
                    help="为多个人物绘制稳定编号（可选，默认关闭=干净版，建议二次生成用无编号版）")
    ap.add_argument("--numbers-color", default="",
                    help="编号颜色 RGB，如 200,200,200（默认低调灰，不干扰二次生成）")
    ap.add_argument("--no-orientation", action="store_true",
                     help="关闭人物朝向/转身识别预览层（默认开启，依 config.ORIENTATION）")
    ap.add_argument("--edge", action="store_true",
                    help="在人物剪影边缘画轮廓描边，突出动作/姿态边界线（可选，默认关闭）")
    ap.add_argument("--edge-color", default="",
                    help="轮廓描边颜色 RGB，如 235,235,235（可选，默认浅灰）")
    ap.add_argument("--fill-mask", action="store_true",
                    help="对人物分割剪影做时间补全：BiRefNet 在动作快时偶发分割失败导致剪影空白帧，"
                         "用前后有效剪影插值承接补上（可选，默认关闭）")
    ap.add_argument("--dilate", type=int, default=3,
                    help="剪影边缘外扩像素数（配合 --fill-mask 使用，默认3，设0关闭）。"
                         "把人物剪影向外扩并闭运算填洞，使其真正包住人物边缘，"
                         "避免 background 版露出人物原貌")
    args = ap.parse_args()
    do_orientation = C.ORIENTATION and not args.no_orientation

    if not os.path.exists(args.input):
        raise SystemExit(f"输入视频不存在: {args.input}")

    dur, w, h, fps = probe_duration(args.input)
    outdir = args.outdir or os.path.splitext(args.input)[0] + "_out"
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]
    print(f"源视频: {w}x{h} @{fps:.2f}fps, {dur:.1f}s")

    # 决定是否拆分
    seg = args.split if args.split > 0 and dur > args.split + 0.5 else 0.0
    split_arg = str(seg) if seg > 0 else "0"
    print(f"分段策略: {'不拆' if seg == 0 else f'按 {seg}s 拆分'}")

    mask = args.reuse_mask
    pose = args.reuse_pose

    # ---------------- 1. mask（BiRefNet, GPU） ----------------
    if args.only_overlay:
        if not (args.reuse_mask and args.reuse_pose):
            raise SystemExit("--only-overlay 需要同时提供 --reuse-mask 与 --reuse-pose")
    elif not mask:
        mask = os.path.join(outdir, f"{base}_mask.mp4")
        if os.path.exists(mask):
            print(f"检测到已有 mask，跳过分割: {mask}")
        else:
            run_with(C.COMFY_PY, "mask_gpu.py",
                     ["--input", args.input, "--output", mask,
                      "--seg", args.seg], "BiRefNet 人物分割")
    else:
        print(f"复用 mask: {mask}")

    # ---------------- 1.5 mask 时间补全（可选 --fill-mask） ----------------
    # BiRefNet 在动作快/剧烈运动时会偶发分割失败，导致某些帧剪影空白（人物明明在画面）。
    # 开启后用前后最近的有效剪影线性插值承接，消除"动作稍快剪影就消失"的观感。
    # 补全后的 mask 以 _mask_fill.mp4 覆盖 mask 变量，后续 overlay 会用补全版。
    if args.fill_mask:
        fill_path = os.path.join(outdir, f"{base}_mask_fill.mp4")
        print(f"开启 mask 时间补全 -> {fill_path}")
        run_with(C.BLENDER_PY, "mask_fill.py",
                 ["--input", mask, "--output", fill_path,
                  "--dilate", args.dilate],
                 "mask 时间补全(消除空白剪影帧+边缘外扩)")
        mask = fill_path   # 后续步骤使用补全后的 mask

    # ---------------- 2. pose（DWPose, CPU） ----------------
    # 朝向识别启用时，导出骨架阶段顺带导出每帧关键点(kp.npy)；否则普通骨架。
    # 骨架视频本身不含朝向标注，朝向是独立预览层（第4步），二者互不影响。
    kp_path = os.path.join(outdir, f"{base}_kp.npy")
    if do_orientation:
        if args.only_overlay:
            raise SystemExit("朝向识别不可与 --only-overlay 同用：需要重跑骨架并导出关键点")
        # pose 需要关键点：有 kp 则复用；无则重跑骨架并导出
        if not os.path.exists(kp_path):
            pose = os.path.join(outdir, f"{base}_pose.mp4")
            print("朝向识别需要关键点数据，将重跑骨架并导出（若已有骨架会覆盖）")
            run_with(C.BLENDER_PY, "pose_dwpose.py",
                     ["--input", args.input, "--output", pose,
                      "--export-kp", kp_path] +
                     (["--numbers"] if args.numbers else []) +
                     (["--numbers-color", args.numbers_color] if args.numbers_color else []),
                     "DWPose 骨架(导出关键点)")
        else:
            print(f"复用已有朝向数据: {kp_path}")
            if not pose or not os.path.exists(pose):
                pose = os.path.join(outdir, f"{base}_pose.mp4")
    elif args.only_overlay:
        pass
    elif not pose:
        pose = os.path.join(outdir, f"{base}_pose.mp4")
        if os.path.exists(pose):
            print(f"检测到已有骨架，跳过检测: {pose}")
        else:
            run_with(C.BLENDER_PY, "pose_dwpose.py",
                     ["--input", args.input, "--output", pose] +
                     (["--numbers"] if args.numbers else []) +
                     (["--numbers-color", args.numbers_color] if args.numbers_color else []),
                     "DWPose 骨架")
    else:
        print(f"复用 pose: {pose}")

    # ---------------- 3. overlay + 分段 + 音轨（朝向判断直接叠加到骨架部分） ----------------
    bg_map = {"black": "black", "background": "background", "both": "black,background"}
    ov_args = ["--input", args.input, "--mask", mask, "--pose", pose,
               "--outdir", outdir, "--bg", bg_map[args.bg],
               "--split", f"{split_arg}", "--prefix", base]
    # 朝向判断默认开启：把关键点传给合成层，直接在成品骨架上方画朝向箭头/标签
    if do_orientation and os.path.exists(kp_path):
        ov_args += ["--kp", kp_path]
    # 人体边缘描绘（可选）：给人物动作画边界线
    if args.edge:
        ov_args.append("--edge")
        if args.edge_color:
            ov_args += ["--edge-color", args.edge_color]
    if args.no_audio:
        ov_args.append("--no-audio")
    run_with(C.BLENDER_PY, "overlay_seg.py", ov_args, "像素叠加+分段+音轨")

    print(f"\n全部完成! 输出目录: {outdir}")


if __name__ == "__main__":
    main()
