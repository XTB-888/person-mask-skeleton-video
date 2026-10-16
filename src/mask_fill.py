# -*- coding: utf-8 -*-
"""mask 时间补全：对人物分割剪影中的"空白帧"做前后承接填充。
背景：BiRefNet 在人物快速动作/剧烈运动时偶尔分割失败，导致某些帧 mask 空白
（人物明明在画面但剪影为空）。本脚本把这类空白帧用前后最近的有效剪影
线性插值承接补上，消除"动作稍快剪影就消失"的观感。

用法:
  BLENDER_PY mask_fill.py --input MASK.mp4 --output MASK_fill.mp4 [--thr 0.001]

--thr 人物占比阈值：某帧白像素占比低于 thx 记为空白帧（默认 0.001）。
"""
import os, cv2, numpy as np, argparse


def analyze_blank(mask_video, thr):
    """读一遍 mask：返回 (帧数, fps, W, H, is_blank list)。"""
    cap = cv2.VideoCapture(mask_video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    blank = []
    for i in range(total):
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        r = int((g > 10).sum()) / float(g.size)
        blank.append(r < thr)
    cap.release()
    return total, fps, W, H, blank


def dilate_frame(f, kernel_px):
    """把剪影边缘外扩 kernel_px 像素，并用闭运算填充内部细小空洞。
    返回处理后的帧（BGR，背景保持黑）。kernel_px<=0 时原样返回。
    """
    if kernel_px <= 0:
        return f
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    bw = (cv2.threshold(g, 12, 255, cv2.THRESH_BINARY)[1]).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px*2+1, kernel_px*2+1))
    # 闭运算填内部空隙，再膨胀向外扩边缘
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
    bw = cv2.dilate(bw, k)
    out = np.zeros_like(f)
    out[bw > 0] = 255   # 被剪影覆盖区域置白，其余保持黑
    return out


def find_segments(blank):
    """把空白帧聚类成连续段 -> [(start, end), ...]"""
    segs = []
    i = 0
    n = len(blank)
    while i < n:
        if blank[i]:
            j = i
            while j < n and blank[j]:
                j += 1
            segs.append((i, j - 1))
            i = j
        else:
            i += 1
    return segs


def nearest_valid(blank, idx, direction):
    """在 blank 中朝 direction(-1=前向, +1=后向) 找最近的有效帧号；找不到返回 None。"""
    n = len(blank)
    k = idx + direction
    while 0 <= k < n:
        if not blank[k]:
            return k
        k += direction
    return None


def read_single_frame(mask_video, idx):
    cap = cv2.VideoCapture(mask_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    cap.release()
    return f if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--thr', type=float, default=0.001, help='人物占比阈值，低于此记为空白帧')
    ap.add_argument('--dilate', type=int, default=3,
                    help='剪影边缘外扩像素数（把人物剪影向外扩，使其真正包住人物边缘，'
                         '避免 background 版露出人物原貌。默认3，设0关闭）。'
                         '同时用闭运算填充剪影内部的细小空洞。')
    a = ap.parse_args()

    total, fps, W, H, blank = analyze_blank(a.input, a.thr)
    n_blank = sum(blank)
    segs = find_segments(blank)
    print(f'总帧 {total}, 空白帧 {n_blank} ({100.0*n_blank/total:.1f}%), 空白段 {len(segs)}')

    # 预计算每个空白段的承接帧（前向/后向最近有效帧号）
    plan = {}
    for (s, e) in segs:
        prev_v = nearest_valid(blank, s, -1)
        next_v = nearest_valid(blank, e, +1)
        plan[(s, e)] = (prev_v, next_v)

    # 顺序写：有效帧原样，空白帧前后承接插值
    os.makedirs(os.path.dirname(a.output) or '.', exist_ok=True)
    out = cv2.VideoWriter(a.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H), True)
    cap = cv2.VideoCapture(a.input)
    cur_seg_idx = 0
    last_valid_frame = None   # 前向最近有效帧内容（顺序读时维护）
    segs_sorted = sorted(plan.keys())
    for idx in range(total):
        ok, f = cap.read()
        if not ok:
            break
        if blank[idx]:
            # 从 segs_sorted 找包含 idx 的段
            seg = None
            for (s, e) in segs_sorted:
                if s <= idx <= e:
                    seg = (s, e)
                    break
            prev_v, next_v = plan[seg]
            # 承接内容
            if prev_v is not None and next_v is not None:
                pf = last_valid_frame  # 前向有效帧
                nf = read_single_frame(a.input, next_v)  # 后向有效帧（本段反向读一次）
                seg_len = seg[1] - seg[0] + 1
                t = (idx - seg[0] + 1) / float(seg_len + 1)  # 0..1
                pf = pf.astype(np.float32)
                nf = nf.astype(np.float32)
                fill = (pf * (1 - t) + nf * t).astype(np.uint8)
            elif prev_v is not None:
                fill = last_valid_frame
            elif next_v is not None:
                fill = read_single_frame(a.input, next_v)
            else:
                fill = f  # 无任何有效承接（整片都空白），保底原样
            out.write(dilate_frame(fill, a.dilate))
        else:
            out.write(dilate_frame(f, a.dilate))
            last_valid_frame = f  # 更新前向最近有效帧
        if (idx + 1) % 25 == 0 or idx + 1 == total:
            print(f' 帧 {idx+1}/{total}', flush=True)
    cap.release()
    out.release()
    print(f'mask 补全完成 -> {a.output}')


if __name__ == '__main__':
    main()
