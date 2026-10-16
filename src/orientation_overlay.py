# -*- coding: utf-8 -*-
"""
person-mask-skeleton-video / 朝向预览层（独立，不改任何原层）
=================================================================
读取 pose_dwpose.py 导出的每帧检测框+关键点+朝向角数据 (.npy) 与源视频，
重绘一个「带转身/朝向标注」的预览视频：
  - 左右半身骨架分色（左=青, 右=橙），转身时左右关系一目了然
  - 每人头顶画「朝向箭头 + 朝向角 + 朝向标签」(正对/右侧身/左侧身/背对)
  - 可叠加 BiRefNet 剪影（--mask）或不叠（默认纯黑底）

输出：<前缀>_orientation.mp4（与源同帧率）。原 pose/叠加/background/black 各层保持不变。

用法（通常由 run.py 调用）:
  BLENDER_PY orientation_overlay.py --input SRC.mp4 --kp KP.npy \
      --outdir OUT --prefix NAME [--mask MASK.mp4] [--bg black|background]
"""
import cv2
import numpy as np
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


# ---- 左右分色骨架（与 pose_dwpose.draw_wholebody 对应，但左右用不同颜色）----
# openpose 索引: 0鼻 1颈 2右肩 3右肘 4右手 5左肩 6左肘 7左手 8右髋 9右膝 10右踝
#               11左髋 12左膝 13左踝 14右眼 15左眼 16右耳 17左耳
# 左半身: 1-5-6-7, 1-11-12-13, 13-15, 15-13 等（用青色）
# 右半身: 1-2-3-4, 1-8-9-10, 14 等（用橙色）
# 躯干主线: 左右肩->髋 用中性色
LINK_LEFT = [(1, 5), (5, 6), (6, 7), (1, 11), (11, 12), (12, 13), (14, 15), (15, 17), (13, 17)]
LINK_RIGHT = [(1, 2), (2, 3), (3, 4), (1, 8), (8, 9), (9, 10), (14, 16), (16, 17)]
COLOR_LEFT = (255, 200, 0)     # BGR 青
COLOR_RIGHT = (0, 120, 255)    # BGR 橙
COLOR_TORSO = (200, 200, 200)  # 中性灰（肩髋）

def _pt(kp, i, conf=0.3):
    if kp[i][2] < conf:
        return None
    return int(round(kp[i][0])), int(round(kp[i][1]))

def draw_orientation_skel(canvas, kps):
    """kps: (133,3) 单个实例。左右分色画 2D 骨架。"""
    def seg(k, a, b):
        pa, pb = _pt(k, a), _pt(k, b)
        if pa and pb:
            cv2.line(canvas, pa, pb, COLOR_TORSO, 2)
    def seg_l(k, a, b):
        pa, pb = _pt(k, a), _pt(k, b)
        if pa and pb:
            cv2.line(canvas, pa, pb, COLOR_LEFT, 2)
    def seg_r(k, a, b):
        pa, pb = _pt(k, a), _pt(k, b)
        if pa and pb:
            cv2.line(canvas, pa, pb, COLOR_RIGHT, 2)
    # 躯干
    for a, b in [(1, 2), (1, 5), (2, 8), (5, 11), (8, 11)]:
        seg(kps, a, b)
    # 左右臂/腿
    for a, b in LINK_LEFT:
        seg_l(kps, a, b)
    for a, b in LINK_RIGHT:
        seg_r(kps, a, b)
    # 头
    nose = _pt(kps, 0)
    if nose:
        cv2.circle(canvas, nose, 4, (0, 255, 255), -1)


TAG_TEXT = {
    'front': '正面',
    'side_R': '右侧身',
    'side_L': '左侧身',
    'back': '背对',
    'unknown': '?',
}

def draw_orientation_tag(canvas, boxes, angles, tags, ids=None, H_frame=540):
    """在每个人物框顶部画朝向箭头 + 角度 + 标签。"""
    h, w = canvas.shape[:2]
    scale = max(0.7, min(1.4, h / 540.0))
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        if i >= len(angles):
            break
        ang = float(angles[i])
        tag = tags[i] if i < len(tags) else 'unknown'
        cx = int(0.5 * (x1 + x2))
        top = max(6, int(y1) - 6)
        # 朝向箭头（以角度为箭头方向，0=朝画面右，180=朝左）
        rad = np.deg2rad(ang)
        ax = cx + int(38 * scale * np.cos(rad))
        ay = top - int(38 * scale * np.sin(rad)) * 0
        # 画水平箭头更直观：用角度符号表示面朝左/右，长度表示转身程度
        sgn = 0
        if ang > 30:
            sgn = 1
        elif ang < -30:
            sgn = -1
        arrow_len = int(30 * scale) + int(abs(ang) / 180.0 * 40 * scale)
        ax2 = cx + sgn * arrow_len
        color = COLOR_LEFT if sgn < 0 else (COLOR_RIGHT if sgn > 0 else (0, 255, 255))
        if sgn != 0:
            cv2.arrowedLine(canvas, (cx, top), (ax2, top), color, max(2, int(2 * scale)),
                            tipLength=0.4)
        # 文字: 角度 + 标签
        txt = f"{int(round(ang))}° {TAG_TEXT.get(tag, tag)}"
        fs = 0.8 * scale
        th = max(1, int(round(1.6 * scale)))
        (tw, thh), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        bx1, by1 = cx - tw // 2 - 4, top - 12 - thh
        bx2, by2 = bx1 + tw + 8, top + 2
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (30, 30, 30), -1)
        cv2.putText(canvas, txt, (bx1 + 4, top - 8), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255), th)


def _smooth_angle_seq(seq, win=7):
    """对一串朝向角做循环角度的时间中值滤波，消除转身瞬间抖动。
    seq: 一维 array(单位度，范围约定 ~[-180,180])。返回同形状的平滑序列。
    """
    seq = np.asarray(seq, dtype=float)
    n = len(seq)
    if n < 2:
        return seq
    # 角度反卷：允许跨 ±180 的连续转身（如 175 -> -165 视为同一方向继续）
    unwrapped = np.unwrap(np.deg2rad(seq)) * 180.0 / np.pi
    out = np.zeros(n)
    half = win // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.median(unwrapped[lo:hi])
    # 卷回 [-180,180]
    out = (out + 180.0) % 360.0 - 180.0
    return out


def _tag_from_angle(ang):
    """由平滑角度重推朝向标签（对应 estimate_orientation 的语义）。"""
    a = abs(ang)
    if a < 40.0:
        return 'front'
    if a < 100.0:
        return 'side_R' if ang > 0 else 'side_L'
    return 'back'


def _apply_time_smooth(all_angle, all_tag):
    """对每个人物实例（按帧索引位置对齐）逐条做时间平滑，返回新角度/标签。
    实例跨帧对齐：这里假设每帧实例顺序相对稳定（多为单人/独立人框），
    直接按实例索引位置逐列滤波；若某实例在某帧不存在则跳过该帧。
    """
    n_inst_max = max((len(a) for a in all_angle), default=0)
    new_angle = [list(a) for a in all_angle]
    new_tag = [list(t) for t in all_tag]
    if n_inst_max == 0:
        return new_angle, new_tag
    for i in range(n_inst_max):
        # 收集该实例索引在所有帧的角度
        seq, idxs = [], []
        for fi, (a, t) in enumerate(zip(all_angle, all_tag)):
            if i < len(a):
                seq.append(a[i])
                idxs.append(fi)
        if len(seq) < 4:
            continue
        smooth = _smooth_angle_seq(seq, win=7)
        for k, fi in enumerate(idxs):
            new_angle[fi][i] = float(smooth[k])
            new_tag[fi][i] = _tag_from_angle(smooth[k])
    return new_angle, new_tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='源视频')
    ap.add_argument('--kp', required=True, help='pose_dwpose --export-kp 导出的 .npy')
    ap.add_argument('--outdir', required=True, help='输出目录')
    ap.add_argument('--prefix', default='seg', help='输出文件名前缀')
    ap.add_argument('--mask', default='', help='可选：BiRefNet 剪影层，叠加白剪影')
    ap.add_argument('--bg', default='black', choices=['black', 'background'],
                    help='black=纯黑底 / background=保留原背景')
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    data = np.load(a.kp, allow_pickle=True).item()
    frame_idx = list(data['frame_idx'])
    all_boxes = list(data['boxes'])
    all_kp = list(data['kp'])
    all_angle = list(data['angle'])
    all_tag = list(data['tag'])
    # 时间平滑，消除转身瞬间标签/角度的抖动
    all_angle, all_tag = _apply_time_smooth(all_angle, all_tag)
    frame2data = {int(f): i for i, f in enumerate(frame_idx)}

    cap = cv2.VideoCapture(a.input)
    caps_m = cv2.VideoCapture(a.mask) if a.mask else None
    fps = cap.get(cv2.CAP_PROP_FPS) or C.OUTPUT_FPS
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.join(a.outdir, f"{a.prefix}_orientation_{a.bg}.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    print(f'生成朝向预览 {W}x{H} @{fps:.1f}fps {total}帧 -> {out_path}')

    t0 = time.time()
    idx = 0
    while True:
        ok_s, frame = cap.read()
        if not ok_s or idx >= total:
            break
        if caps_m:
            ok_m, frame_m = caps_m.read()
        m = frame2data.get(idx)
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        if m is not None:
            boxes = all_boxes[m]
            kps = all_kp[m]
            angles = all_angle[m]
            tags = all_tag[m]
            if a.bg == 'background':
                canvas = frame.copy()
            for i in range(len(boxes)):
                if i < len(kps) and len(kps[i]) >= 18:
                    draw_orientation_skel(canvas, kps[i])
            draw_orientation_tag(canvas, np.array(boxes), angles, tags, H_frame=H)
        else:
            if a.bg == 'background':
                canvas = frame.copy()
        writer.write(canvas)
        idx += 1
        if idx % 50 == 0 or idx == total:
            print(f'  frame {idx}/{total} | {time.time()-t0:.1f}s', flush=True)

    cap.release()
    if caps_m:
        caps_m.release()
    writer.release()
    print('朝向预览完成 ->', out_path)


if __name__ == '__main__':
    main()
