# -*- coding: utf-8 -*-
"""
person-mask-skeleton-video / step3: 像素级叠加 + 按秒分段 + 双背景模式 + 独立音轨
=================================================================
输入：
  --input  源视频（带音轨的原始视频）
  --mask   BiRefNet 人物分割层（白色剪影+黑底，同帧数）
  --pose   DWPose 骨架层（彩色骨架+黑底，同帧数）
输出（每段 x 每种背景模式，均为独立 mp4；_sound 结尾为已混入对应时间窗音轨版本）：
  black      纯黑底版（人物白剪影 + 彩色骨架）
  background 保留原背景版（人物剪影/骨架覆盖于原视频上）

用法（通常由 run.py 调用）:
  BLENDER_PY overlay_seg.py --input SRC.mp4 --mask MASK.mp4 --pose POSE.mp4 \
      --outdir OUT --bg black,background --split 14 --no-audio
"""
import cv2
import numpy as np
import os
import sys
import time
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


TAG_TEXT = {'front': '正面', 'side_R': '右侧身', 'side_L': '左侧身',
            'back': '背对', 'unknown': '?'}


def _draw_orientation(canvas, orient_list, H_frame=540):
    """在骨架顶部叠加朝向箭头+角度+标签。orient_list: [(ang, tag, (x1,y1,x2,y2)), ...]
    位置用人物检测框顶部；若框异常则退回用骨架外接框（由调用方保证提供框）。
    返回直接在 canvas 上绘制（不复制）。"""
    h, w = canvas.shape[:2]
    scale = max(0.7, min(1.4, h / 540.0))
    if not orient_list:
        return
    for (A, T, bb) in orient_list:
        if bb is None:
            continue
        x1, y1, x2, y2 = bb
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        cx = int(0.5 * (x1 + x2))
        y1i = int(y1)
        # 编号画在人物框顶部上方；朝向标签错开到框内侧下方，避免遮挡编号
        top = max(6, y1i + (20 if y1i > 60 else 44))
        # 朝向箭头（水平：sgn 表示面朝左/右，长度随转身程度）
        sgn = 1 if A > 30 else (-1 if A < -30 else 0)
        arrow_len = int(30 * scale) + int(abs(A) / 180.0 * 40 * scale)
        ax2 = cx + sgn * arrow_len
        color = (255, 210, 0) if sgn < 0 else ((40, 120, 255) if sgn > 0 else (0, 255, 255))
        if sgn != 0:
            cv2.arrowedLine(canvas, (cx, top), (ax2, top), color, max(2, int(2 * scale)), tipLength=0.4)
        # 朝向只区分正面/背面：back -> 背面, 其余（front/side_*）-> 正面
        txt = "背面" if T == "back" else "正面"
        fs = 0.8 * scale
        th = max(1, int(round(1.6 * scale)))
        (tw, thh), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        bx1, by1 = cx - tw // 2 - 4, top - 12 - thh
        bx2, by2 = bx1 + tw + 8, top + 2
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (30, 30, 30), -1)
        cv2.putText(canvas, txt, (bx1 + 4, top - 8), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th)


def _draw_body_edge(canvas, frame_m, color=(235, 235, 235), thickness=2):
    """从人物分割剪影提取轮廓并描边，突出人物动作/姿态的边界线。
    - 用 mask 二值化的外轮廓（RETR_EXTERNAL + CHAIN_APPROX_SIMPLE）。
    - 画在骨架/剪影之上，让肢体与躯干的边缘更清晰可辨，便于观察动作精确度。
    - frame_m: 同帧的 BiRefNet 人物分割帧（白色剪影 + 黑底）。
    - canvas: 已叠加骨架的成品画面，直接在原图上绘制。
    """
    if frame_m is None or frame_m.size == 0:
        return canvas
    gray = cv2.cvtColor(frame_m, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(canvas, contours, -1, tuple(int(c) for c in color), thickness)
    return canvas


# 统一输出编码目标：H.264 + AAC（兼容手机/微信/网页/相册）
# 本机 OpenCV 写不了 H.264(缺 OpenH264)，故 vcodec 兼容回退，最终在混音时用 ffmpeg 统一转码。
VCODEC_FROM = 'mp4v'   # OpenCV 写无声分段时用的编码（mp4v/mpeg4）
VCODEC_TO   = 'libx264'  # 最终成品视频编码
VCRF        = '21'        # H.264 质量（越小越清晰，18-23 为视觉无损区间）
AUDIO_CODEC = 'aac'       # 最终成品音频编码
MOVFLAGS    = '+faststart'  # moov 前置，便于流播/即时预览


def _pick_vcodec(ffmpeg):
    """在 ffmpeg 可用时统一用 libx264；不可用（精简版等）则回退 copy。"""
    try:
        r = subprocess.run([ffmpeg, '-hide_banner', '-encoders'],
                           capture_output=True)
        if b'libx264' in r.stdout:
            return VCODEC_TO
    except Exception:
        pass
    return 'copy'


def add_audio(src, seg_path, out_path, start_s, dur_s, ffmpeg):
    """从源视频截取 [start_s, start_s+dur_s] 的音频，混入无声分段视频并统一转码为 H.264+AAC。

    参数顺序要点：
      -ss / -t 放在第一个 -i（源）之前 → 只作用于源输入（精确 seek）
      seg_path 作为第二个输入，-map 1:v 取它的画面, -map 0:a 取源音频
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        return out_path  # 已存在则跳过
    v_c = _pick_vcodec(ffmpeg)
    args = [
        ffmpeg, '-y',
        '-ss', f'{start_s:.3f}', '-t', f'{dur_s:.3f}', '-i', src,
        '-i', seg_path,
        '-map', '1:v:0', '-map', '0:a:0',
        '-c:v', v_c if v_c != 'copy' else VCODEC_FROM,
        '-c:a', AUDIO_CODEC,
        '-shortest',
    ]
    if v_c != 'copy':
        args += ['-preset', 'medium', '-crf', VCRF, '-pix_fmt', 'yuv420p']
    args += ['-movflags', MOVFLAGS, out_path]
    r = subprocess.run(args, capture_output=True)
    if r.returncode != 0:
        # 编码失败时回退为仅复制（保留音频），保证不丢成品
        print('  H.264 转码失败，回退复制视频流:', r.stderr.decode(errors='ignore')[-300:])
        args = [
            ffmpeg, '-y',
            '-ss', f'{start_s:.3f}', '-t', f'{dur_s:.3f}', '-i', src,
            '-i', seg_path,
            '-map', '1:v:0', '-map', '0:a:0',
            '-c:v', 'copy', '-c:a', AUDIO_CODEC,
            '-movflags', MOVFLAGS, '-shortest', out_path,
        ]
        r = subprocess.run(args, capture_output=True)
        if r.returncode != 0:
            print('  音频合成失败:', r.stderr.decode(errors='ignore')[-500:])
            return None
    os.remove(seg_path)  # 成功后删除无声版，避免重复
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='源视频（带音轨）')
    ap.add_argument('--mask', required=True, help='人物分割 mask 视频')
    ap.add_argument('--pose', required=True, help='DWPose 骨架视频')
    ap.add_argument('--outdir', required=True, help='输出目录')
    ap.add_argument('--bg', default=','.join(C.BG_MODES),
                    help='背景模式，逗号分隔，可选 black,background；默认两者都要')
    ap.add_argument('--split', type=float, default=C.SPLIT, help='每段秒数')
    ap.add_argument('--prefix', default='seg', help='输出文件名前缀')
    ap.add_argument('--no-audio', action='store_true', help='不混音轨')
    ap.add_argument('--kp', default='', help='可选：pose_dwpose --export-kp 导出的 .npy，把朝向判断叠加到成品骨架部分')
    ap.add_argument('--edge', action='store_true',
                    help='可选：在人物剪影边缘画轮廓描边，突出人物动作/姿态边界线（默认关闭=不描边）')
    ap.add_argument('--edge-color', default='235,235,235',
                    help='轮廓描边颜色 RGB，如 235,235,235（默认浅灰，不干扰骨架颜色）')
    a = ap.parse_args()

    raw = [m.strip().lower() for m in a.bg.split(',') if m.strip()]
    if 'both' in raw:
        raw = C.BG_MODES  # both 表示两种背景都要
    modes = [m for m in raw if m in C.BG_MODES]
    assert modes, f'背景模式无效: {a.bg}，请选 black 和/或 background 或 both'

    os.makedirs(a.outdir, exist_ok=True)

    try:
        EDGE_COLOR = tuple(int(v) for v in a.edge_color.split(','))
    except Exception:
        EDGE_COLOR = (235, 235, 235)

    # ---------------- 朝向数据（可选）：读取 kp + 时间平滑 ----------------
    frame_orient = {}   # 帧号 -> [(ang, tag, (x1,y1,x2,y2)), ...] 每人一个；未检测到人则不在此帧
    if a.kp and os.path.exists(a.kp):
        kd = np.load(a.kp, allow_pickle=True).item()
        fidx = list(kd['frame_idx'])
        angles = list(kd['angle'])
        tags = list(kd['tag'])
        boxes = list(kd['boxes'])
        # 逐人索引做时间平滑（unwrapped 循环中值），消除转身瞬间抖动
        n_inst = 0
        for al in angles:
            if len(al) > n_inst:
                n_inst = len(al)
        if n_inst > 0:
            for i in range(n_inst):
                seq, ids = [], []
                for fi, (al, tl) in enumerate(zip(angles, tags)):
                    if i < len(al):
                        seq.append(float(al[i]))
                        ids.append(fi)
                if len(seq) < 4:
                    continue
                unwrapped = np.unwrap(np.deg2rad(np.asarray(seq))) * 180.0 / np.pi
                for k in range(len(ids)):
                    lo, hi = max(0, k - 3), min(len(ids), k + 4)
                    sm = (np.median(unwrapped[lo:hi]) + 180.0) % 360.0 - 180.0
                    A = float(sm)
                    T = 'front' if abs(A) < 40 else ('back' if abs(A) >= 100 else ('side_R' if A > 0 else 'side_L'))
                    bx = boxes[ids[k]]
                    bb = tuple(float(v) for v in bx[i]) if i < len(bx) else (0, 0, 0, 0)
                    lst = frame_orient.setdefault(fidx[ids[k]], [])
                    while len(lst) <= i:
                        lst.append(None)
                    lst[i] = (A, T, bb)

    cap_s = cv2.VideoCapture(a.input)
    cap_m = cv2.VideoCapture(a.mask)
    cap_p = cv2.VideoCapture(a.pose)

    total = int(cap_m.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise SystemExit(f'无法读取输入(mask={a.mask} 帧数={total})，请确认 mask/pose/源视频文件存在且可解码')
    fps = cap_m.get(cv2.CAP_PROP_FPS) or C.OUTPUT_FPS
    W = int(cap_m.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap_m.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # split<=0 表示不拆, 整段作为一个分片; 否则每 split 秒一段
    if a.split <= 0:
        fpi = total
    else:
        fpi = max(1, int(round(a.split * fps)))
    n_seg = max(1, int(np.ceil(total / fpi)))
    print(f'共 {total} 帧, 每段 {fpi} 帧, 拆成 {n_seg} 段, {W}x{H} @{fps:.1f}fps, 模式={modes}')

    fourcc = cv2.VideoWriter_fourcc(*VCODEC_FROM)
    writers = {}  # (mode, seg_index) -> VideoWriter
    for mode in modes:
        for i in range(n_seg):
            name = f"{a.prefix}_seg{i+1:02d}_{mode}.mp4"
            writers[(mode, i)] = cv2.VideoWriter(os.path.join(a.outdir, name), fourcc, fps, (W, H))

    t0 = time.time()
    for idx in range(total):
        ok_s, frame_src = cap_s.read()
        ok_m, frame_m = cap_m.read()
        ok_p, frame_p = cap_p.read()
        if not (ok_s and ok_m and ok_p):
            break
        seg = min(idx // fpi, n_seg - 1)

        # 像素级叠加: 骨架像素直接写到 mask 上（人物区域白剪影，骨架彩色覆盖）
        overlay = frame_m.copy()
        pose_mask = (frame_p.sum(axis=2) > 10)
        overlay[pose_mask] = frame_p[pose_mask]

        # 人体边缘描绘（可选 --edge 时启用）：给人物动作画边界线，突出姿态轮廓
        if a.edge:
            _draw_body_edge(overlay, frame_m, color=EDGE_COLOR)

        # 朝向判断叠加到骨架部分（可选 --kp 时启用）
        orient_list = frame_orient.get(idx) if a.kp else None
        if orient_list:
            _draw_orientation(overlay, orient_list, H_frame=H)

        for mode in modes:
            if mode == 'black':
                writers[(mode, seg)].write(overlay)
            elif mode == 'background':
                combine = frame_src.copy()
                person_zone = (frame_m.sum(axis=2) > 10) | pose_mask
                combine[person_zone] = overlay[person_zone]
                if a.edge:
                    _draw_body_edge(combine, frame_m, color=EDGE_COLOR)
                if orient_list:
                    _draw_orientation(combine, orient_list, H_frame=H)
                writers[(mode, seg)].write(combine)
            else:
                raise SystemExit(f'未知背景模式: {mode}')

        if idx % 50 == 0 or idx == total - 1:
            print(f'  帧 {idx+1}/{total} | seg {seg+1}/{n_seg} | {time.time()-t0:.1f}s', flush=True)

    for w in writers.values():
        w.release()
    cap_s.release(); cap_m.release(); cap_p.release()
    print(f'视频分段合成完成 -> {a.outdir}')

    # ---------------- 音频 ----------------
    if a.no_audio:
        # 不混音轨时，无声分段即成品：统一转码为 H.264+AAC 编码的 mp4
        if _pick_vcodec(C.FFMPEG) != 'copy':
            for mode in modes:
                for i in range(n_seg):
                    seg_path = os.path.join(a.outdir, f"{a.prefix}_seg{i+1:02d}_{mode}.mp4")
                    tmp_path = seg_path + '.h264.mp4'
                    args = [C.FFMPEG, '-y', '-i', seg_path,
                            '-c:v', VCODEC_TO, '-preset', 'medium', '-crf', VCRF,
                            '-pix_fmt', 'yuv420p', '-an',
                            '-movflags', MOVFLAGS, tmp_path]
                    r = subprocess.run(args, capture_output=True)
                    if r.returncode == 0 and os.path.exists(tmp_path):
                        os.replace(tmp_path, seg_path)  # 原位替换成 H.264 版
                    else:
                        print('  无音轨转码失败:', r.stderr.decode(errors='ignore')[-300:])
            print('无音轨模式: 成片已统一转为 H.264 编码')
        else:
            print('无音轨模式: ffmpeg 无 H.264 编码器，保留 OpenCV mp4v 编码')
        print('全部完成')
        return
    print('混入对应时间窗音轨 ...')
    for mode in modes:
        for i in range(n_seg):
            seg_path = os.path.join(a.outdir, f"{a.prefix}_seg{i+1:02d}_{mode}.mp4")
            out_path = os.path.join(a.outdir, f"{a.prefix}_seg{i+1:02d}_{mode}_sound.mp4")
            # 起始时间按帧对齐: 段 i 从 i*fpi/fps 秒开始, 时长 fpi/fps 秒
            start_s = i * fpi / fps
            dur_s = fpi / fps
            add_audio(a.input, seg_path, out_path, start_s, dur_s, C.FFMPEG)
    print('全部完成')


if __name__ == '__main__':
    main()
