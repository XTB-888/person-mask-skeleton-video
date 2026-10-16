# -*- coding: utf-8 -*-
"""基于 ComfyUI comfyui_controlnet_aux 官方 DWPose ONNX 流程的姿态骨架提取。

关键区别（相对旧版）:
- 使用官方 RTMPose SimCC 解码 (get_simcc_maximum) 而非 soft-argmax
- 使用官方 get_warp_matrix/top_down_affine 仿射预处理，输入 (288,384)
- 使用 yolox 官方 demo_postprocess + multiclass_nms
- 按官方 wholebody.py 做 MMPose->OpenPose 关键点重排
- 绘制: 官方 draw_bodypose / face / hand
"""
import os
import sys
import argparse
import time
import numpy as np
import cv2
import onnxruntime as ort
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

# ---------------- 配置（默认，命令行 --input/--output 会覆盖） ----------------
DET_PATH = C.DET_PATH
POSE_PATH = C.POSE_PATH
SRC_VIDEO = ""   # 由 main 用 --input 填充
OUT_VIDEO = ""   # 由 main 用 --output 填充
DEBUG_FRAME = ""

DET_INPUT = (640, 640)      # (h, w)
POSE_INPUT = (288, 384)     # (w, h) RTMPose DWPose 384 输入是 288x384

MAX_FRAMES = 0              # 0 = 全部
STEP = 1
SHOW = False                 # 是否做可视化（骨架+人物混合）
SEG_ALPHA = 0.0
DRAW_FACE = True
DRAW_HAND = True
CONF_THRE = 0.3

# ---------------- 检测 (cv_ox_det) ----------------
def nms(boxes, scores, nms_thr):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]
    return keep

def multiclass_nms(boxes, scores, nms_thr, score_thr):
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        keep = nms(valid_boxes, valid_scores, nms_thr)
        if len(keep) > 0:
            cls_inds = np.ones((len(keep), 1)) * cls_ind
            dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
            final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)

def demo_postprocess(outputs, img_size, p6=False):
    grids = []
    expanded_strides = []
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]
    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]
    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))
    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    return outputs

def det_preprocess(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)),
                             interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def inference_detector(session, oriImg, detect_classes=(0,), dtype=np.float32):
    img, ratio = det_preprocess(oriImg, DET_INPUT)
    input = img[None, :, :, :].astype(dtype)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input})
    predictions = demo_postprocess(output[0], DET_INPUT)[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is None:
        return None
    final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
    isscore = final_scores > 0.3
    iscat = np.isin(final_cls_inds, detect_classes)
    isbbox = [i and j for (i, j) in zip(isscore, iscat)]
    final_boxes = final_boxes[isbbox]
    return final_boxes

# ---------------- 姿态 (cv_ox_pose) ----------------
def bbox_xyxy2cs(bbox, padding=1.25):
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]
    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding
    if dim == 1:
        center = center[0]
        scale = scale[0]
    return center, scale

def _fix_aspect_ratio(bbox_scale, aspect_ratio):
    w, h = np.hsplit(bbox_scale, [1])
    bbox_scale = np.where(w > h * aspect_ratio,
                          np.hstack([w, w / aspect_ratio]),
                          np.hstack([h * aspect_ratio, h]))
    return bbox_scale

def _rotate_point(pt, angle_rad):
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt

def _get_3rd_point(a, b):
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c

def get_warp_matrix(center, scale, rot, output_size, shift=(0., 0.), inv=False):
    shift = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0., src_w * -0.5]), rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])
    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return warp_mat

def top_down_affine(input_size, bbox_scale, bbox_center, img):
    w, h = input_size
    warp_size = (int(w), int(h))
    bbox_scale = _fix_aspect_ratio(bbox_scale, aspect_ratio=w / h)
    center = bbox_center
    scale = bbox_scale
    rot = 0
    warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))
    img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
    return img, bbox_scale

def pose_preprocess(img, out_bbox, input_size=POSE_INPUT):
    img_shape = img.shape[:2]
    out_img, out_center, out_scale = [], [], []
    if len(out_bbox) == 0:
        out_bbox = [[0, 0, img_shape[1], img_shape[0]]]
    for i in range(len(out_bbox)):
        x0, y0, x1, y1 = out_bbox[i][0], out_bbox[i][1], out_bbox[i][2], out_bbox[i][3]
        bbox = np.array([x0, y0, x1, y1])
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)
        resized_img, scale = top_down_affine(input_size, scale, center, img)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        resized_img = (resized_img - mean) / std
        out_img.append(resized_img)
        out_center.append(center)
        out_scale.append(scale)
    return out_img, out_center, out_scale

def get_simcc_maximum(simcc_x, simcc_y):
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)
    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)
    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.] = -1
    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)
    return locs, vals

def decode(simcc_x, simcc_y, simcc_split_ratio=2.0):
    keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints /= simcc_split_ratio
    return keypoints, scores

def pose_postprocess(outputs, model_input_size, center, scale, simcc_split_ratio=2.0):
    all_key, all_score = [], []
    for i in range(len(outputs)):
        simcc_x, simcc_y = outputs[i]
        keypoints, scores = decode(simcc_x, simcc_y, simcc_split_ratio)
        keypoints = keypoints / model_input_size * scale[i] + center[i] - scale[i] / 2
        all_key.append(keypoints[0])
        all_score.append(scores[0])
    return np.array(all_key), np.array(all_score)

def inference_pose(session, out_bbox, oriImg, model_input_size=POSE_INPUT, dtype=np.float32):
    resized_img, center, scale = pose_preprocess(oriImg, out_bbox, model_input_size)
    input = np.stack(resized_img, axis=0).transpose(0, 3, 1, 2).astype(dtype)
    input_name = session.get_inputs()[0].name
    all_outputs = session.run(None, {input_name: input})
    outputs = []
    # 官方对 batch 逐条拆分: all_outputs[i][batch_idx:batch_idx+1]
    for batch_idx in range(len(all_outputs[0])):
        outputs.append([all_outputs[i][batch_idx:batch_idx + 1, ...] for i in range(len(all_outputs))])
    keypoints, scores = pose_postprocess(outputs, model_input_size, center, scale)
    return keypoints, scores

# ---------------- 绘制 (util.py 官方) ----------------
def draw_bodypose(canvas, keypoints, color=None, xinsr_stick_scaling=False):
    # keypoints: list[ (x_px, y_px) or None ]
    # color: 若给定，整具骨架统一用该身份色（用于人物身份配色，无数字）；否则用默认肢彩虹色。
    H, W, _ = canvas.shape
    stickwidth = 4
    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5],
        [6, 7], [7, 8], [2, 9], [9, 10],
        [10, 11], [2, 12], [12, 13], [13, 14],
        [2, 1], [1, 15], [15, 17], [1, 16],
        [16, 18],
    ]
    base_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    kpts = []
    for kp in keypoints[:18]:
        if kp is None:
            kpts.append(None)
        else:
            x, y = kp
            kpts.append((x, y))
    for (k1_index, k2_index), base in zip(limbSeq, base_colors):
        k1 = kpts[k1_index - 1]
        k2 = kpts[k2_index - 1]
        if k1 is None or k2 is None:
            continue
        X = np.array([k1[1], k2[1]])
        Y = np.array([k1[0], k2[0]])
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        col = color if color is not None else base
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in col])
    for kp, base in zip(kpts, base_colors):
        if kp is None:
            continue
        x, y = kp
        col = color if color is not None else base
        cv2.circle(canvas, (int(x), int(y)), 4, col, thickness=-1)
    return canvas

def draw_facepose(canvas, keypoints):
    if not keypoints:
        return canvas
    H, W, _ = canvas.shape
    for kp in keypoints:
        if kp is None:
            continue
        x, y = kp
        x = int(x)
        y = int(y)
        if x > 1 and y > 1:
            cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas

def _hsv_to_rgb(h, s, v):
    """HSV(h:0-360 deg, s,v:0-1) -> (r,g,b) 0-255 int tuple, 替代 matplotlib.colors.hsv_to_rgb."""
    c = v * s
    x = c * (1 - abs((h / 60.0) % 2 - 1))
    m = v - c
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))

def draw_handpose(canvas, keypoints):
    if not keypoints:
        return canvas
    H, W, _ = canvas.shape
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    import math
    for ie, (e1, e2) in enumerate(edges):
        k1 = keypoints[e1]
        k2 = keypoints[e2]
        if k1 is None or k2 is None:
            continue
        x1, y1 = k1
        x2, y2 = k2
        if x1 > 1 and y1 > 1 and x2 > 1 and y2 > 1:
            # 替代 matplotlib.colors.hsv_to_rgb: 手部每段骨架用不同色相
            h = (ie / float(len(edges))) * 360.0
            c_rgb = _hsv_to_rgb(h, 1.0, 1.0)
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), c_rgb, thickness=2)
    for kp in keypoints:
        if kp is None:
            continue
        x, y = kp
        if x > 1 and y > 1:
            cv2.circle(canvas, (int(x), int(y)), 4, (0, 0, 255), thickness=-1)
    return canvas

import math

def draw_wholebody(canvas, keypoints_info, color=None):
    """keypoints_info: (N, K, 3) 已经过 MMPose->OpenPose 重排; 坐标是像素坐标.
    兼容传入 (K,3) 单实例。color: 若给定，每具骨架统一用该身份色（人物按色区分，无数字）。"""
    keypoints_info = np.asarray(keypoints_info)
    if keypoints_info.ndim == 2:
        keypoints_info = keypoints_info[None]     # (K,3) -> (1,K,3)
    for instance in keypoints_info:
        body_kps = []
        for i in range(18):
            x, y, s = instance[i]
            body_kps.append((x, y) if s >= CONF_THRE else None)
        draw_bodypose(canvas, body_kps, color=color)

        if DRAW_FACE:
            face_kps = []
            for i in range(24, 92):
                x, y, s = instance[i]
                face_kps.append((x, y) if s >= CONF_THRE else None)
            # openpose face 70 点, DWPose 只有 68, 补眼睛两点用于后续
            draw_facepose(canvas, face_kps)

        if DRAW_HAND:
            left_hand = []
            for i in range(92, 113):
                x, y, s = instance[i]
                left_hand.append((x, y) if s >= CONF_THRE else None)
            right_hand = []
            for i in range(113, 134):
                x, y, s = instance[i]
                right_hand.append((x, y) if s >= CONF_THRE else None)
            draw_handpose(canvas, left_hand)
            draw_handpose(canvas, right_hand)
    return canvas

# ---------------- 多人物编号（可选） ----------------
# DWPose 本身无跨帧跟踪。为让编号在时间轴上保持稳定，这里做轻量级
# 帧间匹配 + 多帧停留窗口 + ID 回收复用：
#   1) 画编号前先把检测框按空间位置排序（左上角 x，次要 y），保证画面上编号顺序稳定；
#   2) 双向最优匹配：先由每个旧 ID 竞争最近的当前框，再保证一个框只被一个旧 ID 占用，
#      避免 NMS 输出顺序抖动导致的 ID 互换；
#   3) 多帧停留窗口：人物短暂漏检/消失后保留 ID 若干帧，恢复后用同一 ID；
#   4) 长时间不可见的人物 ID 放回回收池，分配给新出现人物，编号不会无限膨胀。
POS_TRACK_MODE = "center"      # 中心点匹配
MAX_TRACK_RADIUS = 0.30        # 匹配半径阈值（相对画面高度），超出视为新人物
TRACK_MAX_AGE   = 30           # 停留窗口帧数（约 1s@30fps），超过则释放 ID 回收（放大以吸收短暂漏检）
# 活跃编号池上限：画面同时最多约 6 人，留缓冲。超过此数时，最久未匹配的旧轨迹
# 直接淘汰并把 ID 放回回收池，防止重叠时旧 ID 不死、新 ID 无限递增（编号膨胀
# 正是颜色大跳变的根因——从 2 跳到 80 远比 2↔3 换色刺眼）。
MAX_ACTIVE_IDS = 8
# 人物重叠时纯几何分不清谁是谁，用检测框内的肤色/衣着颜色直方图辅助仲裁。
# 经真实视频实测：alpha=0（纯几何）28 次强重叠互换 -> alpha=0.6 降为 15 次。ema 平滑反而更差（人物运动中颜色本会变）。
APPEARANCE_ALPHA = 0.6          # 外观辅助权重，叠加在几何距离上作内部择优
APPEARANCE_BINS = 24            # HSV 色相直方图分桶


def _box_center(box):
    return 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])


def extract_appearance_hist(frame, box):
    """从检测框主体中部提取归一化 HSV 色相直方图，作为人物外观特征。
    避开边/头部/背景，只取框内中段且饱和度/亮度达标的像素。无可提取像素时返回 None。
    """
    if frame is None:
        return None
    Hf, Wf, _ = frame.shape
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(x1, Wf - 1)); x2 = max(x1, min(x2, Wf - 1))
    y1 = max(0, min(y1, Hf - 1)); y2 = max(y1, min(y2, Hf - 1))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    cy0, cy1 = y1 + int((y2 - y1) * 0.25), y1 + int((y2 - y1) * 0.75)
    cx0, cx1 = x1 + int((x2 - x1) * 0.2), x1 + int((x2 - x1) * 0.8)
    roi = frame[cy0:cy1, cx0:cx1]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].ravel(); s = hsv[:, :, 1].ravel(); v = hsv[:, :, 2].ravel()
    mask = (s > 40) & (v > 40)
    hsel = h[mask]
    if hsel.size < 20:
        return None
    hist, _ = np.histogram(hsel, bins=APPEARANCE_BINS, range=(0, 180))
    hist = hist.astype(float)
    total = hist.sum()
    if total <= 0:
        return None
    return hist / total


def appearance_dist(h1, h2):
    """Bhattacharyya 距离 [0,~1]。直方图缺失时给中性 0.5。"""
    if h1 is None or h2 is None:
        return 0.5
    eps = 1e-6
    return 1.0 - float(np.sum(np.sqrt(h1 * h2 + eps)))


def _sort_boxes(boxes):
    """按空间位置（左上角 x，次要 y）稳定排序，返回 (排序后的 boxes, 原索引)。"""
    if boxes is None or len(boxes) == 0:
        return boxes, np.array([], int)
    boxes = np.asarray(boxes, float)
    order = np.lexsort((boxes[:, 1], boxes[:, 0]))   # 先按 x，再按 y
    return boxes[order], order


def assign_ids(boxes, prev_centers, H, frame=None, radius=MAX_TRACK_RADIUS):
    """把当前帧 boxes (N,4) 映射到稳定的编号列表 (长度 N)。

    prev_centers 元素: [cx, cy, id, age, hist]（age=连续未匹配帧数, hist=外观直方图或 None）。
    双向最优匹配 + 停留窗口 + ID 回收；帧内多人重叠时用外观颜色直方图辅助仲裁，
    从而比纯几何更稳定地保持同一人物身份。
    """
    boxes = np.asarray(boxes, float) if len(boxes) else np.zeros((0, 4), float)
    N = len(boxes)
    ids = np.zeros(N, dtype=np.int32)
    if N == 0:
        for t in prev_centers:
            t[3] += 1     # 无人时也让停留窗口推进，超龄回收
        return ids

    rad = radius * H
    centers = np.array([_box_center(b) for b in boxes])   # (N,2)
    # 当前帧每个框的外观
    cur_hists = [extract_appearance_hist(frame, b) for b in boxes] if frame is not None else [None]*N

    # ---- 1. 旧 ID -> 新框 的候选距离矩阵（几何欧氏距离，作候选门槛）----
    M = len(prev_centers)                       # 旧轨迹数
    dist = np.zeros((M, N), dtype=np.float64)   # 纯几何距离（判定是否同一人候选）
    cost = np.zeros((M, N), dtype=np.float64)   # 几何 + 外观加权（重叠时择优）
    for i, t in enumerate(prev_centers):
        d = np.hypot(centers[:, 0] - t[0], centers[:, 1] - t[1])
        dist[i] = d
        for j in range(N):
            dj = float(d[j])
            if dj <= rad:
                cost[i, j] = dj + APPEARANCE_ALPHA * float(appearance_dist(t[4], cur_hists[j])) * H
            else:
                cost[i, j] = np.inf

    # ---- 2. 双向最优匹配（贪心+独占），保证一个框只被一个旧 ID 占用 ----
    used_box = np.zeros(N, dtype=bool)
    used_trk = np.zeros(M, dtype=bool)
    assign = np.full(N, -1, dtype=int)   # box -> track 索引
    # 旧 ID 依次选自己候选内 外观+几何 综合最小的框
    order_trk = np.argsort(np.min(dist, axis=1))   # 几何最近者优先，减少误抢
    for i in order_trk:
        crow = cost[i].copy()
        crow[used_box] = np.inf
        j = int(np.argmin(crow))
        if dist[i, j] <= rad:     # 候选门槛仍按几何距离，外观只作重叠时择优
            assign[j] = i
            used_box[j] = True
            used_trk[i] = True

    # ---- 3. 分配编号 + 更新轨迹（位置 + 外观）----
    # 目标：活跃编号总数不超过 MAX_ACTIVE_IDS。编号只在编号池内轮转、不无限膨胀，
    # 从而避免重叠时\"从 2 跳到 80\"的颜色大跳变。
    #
    # 未匹配旧轨迹 = 两条处置：
    #   - 作为\"借号池\"：无匹配新框优先借其最久未匹配的编号（该轨迹随之淘汰）；
    #   - 作为\"短暂漏检待恢复\"：编号没被借走时保留进下一帧（在活跃池上限内）。
    # 固定编号池：编号只在 1..MAX_ACTIVE_IDS 内轮转（画面最多约 6 人，池设 8 留缓冲），
    # 编号绝不无限膨胀，从根上避免重叠时"从 2 跳到 80"的颜色大跳变。
    # 未匹配旧轨迹 = 待恢复候选：占住自身编号防新框抢走；受池上限约束，超出淘汰（编号自动回池）。
    pending = []                       # 未匹配旧轨迹（age 已推进，未超龄）-> 待恢复
    for i in range(M):
        if not used_trk[i]:
            prev_centers[i][3] += 1
            if prev_centers[i][3] <= TRACK_MAX_AGE:
                pending.append(prev_centers[i])

    # 当前占用编号集合 = 匹配到的旧编号 + 待恢复轨迹编号（占住防抢）
    active = set(int(prev_centers[i][2]) for i in range(M) if used_trk[i])
    for t in pending:
        active.add(int(t[2]))

    new_centers = []
    for j in range(N):
        i = assign[j]
        if i >= 0:
            pid = int(prev_centers[i][2])
        else:
            pid = _recycle_id(active)   # 从固定池取"未占用"最小号，绝不在池外涨号
        new_centers.append([centers[j, 0], centers[j, 1], pid, 0, cur_hists[j]])
        active.add(pid)
        ids[j] = pid

    # 保留待恢复轨迹（编号空闲则保留进下一帧）；受池上限约束，超出的最久待恢复淘汰（编号回池）
    pending.sort(key=lambda t: t[3], reverse=True)
    for t in pending:
        if len(new_centers) >= MAX_ACTIVE_IDS:
            break                       # 池满，剩余待恢复轨迹淘汰（编号自动回收）
        new_centers.append(t)
    prev_centers[:] = new_centers
    return ids


# 固定编号池：编号只在 1..MAX_ACTIVE_IDS 内轮转，绝不无限膨胀
_free_ids = []          # 池内可复用编号（小号优先）
_next_id = [1]          # 仅用于池内递增到 MAX_ACTIVE_IDS，上限封顶


def _recycle_id(active=()):
    """从固定编号池(1..MAX_ACTIVE_IDS)取一个\"当前未占用\"的最小编号。

    active: 当前帧已占用的编号集合。优先复用池内小号；若池内全部被占用则
    (人数>MAX时)在池内轮转取最小编号。编号永不超过 MAX_ACTIVE_IDS，
    从根上防止\"重叠时旧编号不死、新编号无限递增\"的颜色大跳变。
    """
    taken = set(int(p) for p in active)
    for cand in range(1, MAX_ACTIVE_IDS + 1):
        if cand not in taken and cand not in _free_ids:
            return cand
    # 池内小号均被占用：先复用 _free_ids 里最小的
    if _free_ids:
        _free_ids.sort()
        return _free_ids.pop(0)
    # 极端情形：1..MAX 全占用且无空闲 -> 在池内轮转（取最大可用，尽量不抢当前活跃最小号）
    for cand in range(MAX_ACTIVE_IDS, 0, -1):
        if cand not in taken:
            return cand
    return 1   # 兜底：真超员时轮转最小号（罕见）


def _put_back_id(pid):
    """把超龄人物的 ID 放回回收池（避免重复放入）。"""
    if pid not in _free_ids and pid < _next_id[0]:
        _free_ids.append(int(pid))


GOLDEN_ANGLE = 137.50776   # 色相黄金角，保证相邻 id 色差最大，人物身份色区分度高


def id_color(pid):
    """按稳定 id 分配一条专属身份色（基于色相黄金角），同一人物跨帧同色。"""
    pid = int(pid)
    h = (pid * GOLDEN_ANGLE) % 360.0
    return _hsv_to_rgb(h, 0.85, 0.98)


def draw_id_badges(canvas, boxes, colors, sort=True):
    """在每个人物身体中央绘制纯色身份徽章（圆形，无数字），跨帧颜色稳定。"""
    H, W, _ = canvas.shape
    r = max(10, int(round(0.022 * W)))           # 随画布宽缩放
    if sort and boxes is not None and len(boxes) > 1:
        boxes, order = _sort_boxes(boxes)
        colors = [colors[int(i)] for i in order]
    for (x1, y1, x2, y2), col in zip(boxes, colors):
        cx = int(0.5 * (x1 + x2))
        cy = int(0.5 * (y1 + y2))      # 检测框垂直中心 = 人物身体中部
        cv2.circle(canvas, (cx, cy), r, col, -1)
        cv2.circle(canvas, (cx, cy), r, (10, 10, 10), 2)   # 深色细描边，黑底上更清晰
    return canvas


def draw_numbers(canvas, boxes, ids, color=(200, 200, 200), sort=True):
    """在每个人物身体中央绘制编号标签（圆形底 + 白字，跨帧稳定）。
    sort=True 时先按空间位置排序再绘制，保证画面上的编号顺序稳定。
    """
    H, W, _ = canvas.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.7, round(0.8 * H / 540.0, 2))   # 随高度缩放
    thick = max(1, int(round(1.8 * H / 540.0)))
    if sort and boxes is not None and len(boxes) > 1:
        boxes, order = _sort_boxes(boxes)
        ids = np.asarray(ids, dtype=np.int32)[order]
    for (x1, y1, x2, y2), pid in zip(boxes, ids):
        cx = int(0.5 * (x1 + x2))
        cy = int(0.5 * (y1 + y2))      # 检测框垂直中心 = 人物身体中部
        text = str(int(pid))
        (tw, th), _ = cv2.getTextSize(text, font, fs, thick)
        r = max(10, int((tw * 0.7) // 2 + 4))   # 圆形半径，容纳数字
        cv2.circle(canvas, (cx, cy), r, color, -1)
        cv2.putText(canvas, text, (cx - tw // 2, cy + th // 2), font, fs, (20, 20, 20), thick)
    return canvas

# ---------------- 朝向/转身估计（可选，独立预览层用） ----------------
# 基于 2D 关键点推断人物朝向角（面向+转动方向），仅作可视化提示。
# openpose 关键点索引（重排后）：0鼻子 1颈 2右肩 3右肘 4右手 5左肩 6左肘 7左手
#                              8右髋 9右膝 10右踝 11左髋 12左膝 13左踝 14右眼 15左眼 16右耳 17左耳
# 朝向规则：
#   用左右肩/左右髋的 x 投影差 + 可见眼睛/耳朵数量 + 鼻子相对肩中心位置，
#   综合判断人物是正对/侧身/背对以及偏向角度（0=正对镜头, ±90=侧身, ±160~180=背对）。

def _kp(x, y, s, idx):
    """返回某关键点坐标(像素)及是否可靠(置信度够)。"""
    ok = bool(s[idx] >= CONF_THRE)
    return (float(x[idx]), float(y[idx])) if ok else None

def estimate_orientation(kp_info, img_w):
    """kp_info: (N,K,3) 单个人物关键点。返回 (angle_deg, tag, side)。
    angle 含义: 0=正对镜头; +90=人物朝画面右侧(右脸/右背因转身); -90=朝左侧; ~180=背对。
    tag: 'front'正对 / 'side_R'右侧身 / 'side_L'左侧身 / 'back'背对 / 'unknown'。
    side: 转身侧, -1左 / 1右 / 0不明确。

    关键判据：
      - 前后朝向由「左右肩/髋在图像中的左右顺序」决定（最可靠）：
          正对镜头 -> 本人的左肩(Lsh)出现在图像右侧, 左髋(Lhip)也在右侧
          背对镜头 -> 本人的右肩(Rsh)跑到图像右侧, 右髋(Rhip)也在右侧
      - 转身程度由「鼻子相对身体中心的偏移量(nose_off)」决定。
    """
    kps = kp_info
    nose   = _kp(kps[:,0], kps[:,1], kps[:,2], 0)
    neck   = _kp(kps[:,0], kps[:,1], kps[:,2], 1)
    rsh    = _kp(kps[:,0], kps[:,1], kps[:,2], 2)
    lsh    = _kp(kps[:,0], kps[:,1], kps[:,2], 5)
    rhip   = _kp(kps[:,0], kps[:,1], kps[:,2], 8)
    lhip   = _kp(kps[:,0], kps[:,1], kps[:,2], 11)
    reye   = _kp(kps[:,0], kps[:,1], kps[:,2], 14)
    leye   = _kp(kps[:,0], kps[:,1], kps[:,2], 15)
    rear   = _kp(kps[:,0], kps[:,1], kps[:,2], 16)
    lear   = _kp(kps[:,0], kps[:,1], kps[:,2], 17)

    # ---- 1. 肩/髋顺序：正对(+1) vs 背对(-1) ----
    # 肩顺序优先，肩不可靠时用髋顺序兜底。
    shoulder_w = 0.0
    s_ord = None          # +1 左肩在图像右(=正对), -1 右肩在图像右(=背对)
    if lsh and rsh:
        shoulder_w = abs(lsh[0] - rsh[0])
        s_ord = 1 if lsh[0] > rsh[0] else -1
    hip_w = 0.0
    h_ord = None
    if lhip and rhip:
        hip_w = abs(lhip[0] - rhip[0])
        h_ord = 1 if lhip[0] > rhip[0] else -1
    # 肩判定可信时用它；肩宽过小(近乎侧影)则参考髋。
    if s_ord is not None and shoulder_w >= 6:
        order = s_ord
        width = shoulder_w
    elif h_ord is not None:
        order = h_ord
        width = hip_w
    else:
        order, width = None, 0.0

    # 身体中心（肩优先，肩缺失用髋）
    if lsh and rsh:
        cx = 0.5 * (lsh[0] + rsh[0])
    elif lhip and rhip:
        cx = 0.5 * (lhip[0] + rhip[0])
    else:
        cx = None

    # 鼻子相对身体中心的归一化偏移（转身程度/偏头方向）
    nose_off = None
    if nose is not None and cx is not None and width > 1:
        nose_off = (nose[0] - cx) / width

    eyes = sum(1 for e in (reye, leye) if e is not None)
    ears = sum(1 for e in (rear, lear) if e is not None)

    if order is None or width < 1:
        return 0.0, 'unknown', 0

    front = order > 0   # 左肩/左髋在图像右侧 -> 面向镜头

    if front:
        # ===== 面向镜头半侧 =====
        if nose is not None and nose_off is not None:
            s = 1 if nose_off > 0 else (-1 if nose_off < 0 else 0)
            a = abs(nose_off)
            if a >= 0.55:
                ang = s * 85.0
                tag = f'side_{"R" if s > 0 else "L"}'
            elif a >= 0.28:
                ang = s * 55.0
                tag = f'side_{"R" if s > 0 else "L"}'
            else:
                ang = s * 15.0
                tag = 'front'
            return ang, tag, s
        else:
            # 面向镜头但脸被遮挡（低头/手臂挡脸）=> 仍算正面
            return 0.0, 'front', 0
    else:
        # ===== 背对镜头半侧 =====
        # 核心：右肩/右髋已跑到图像右侧 => 背对。
        # 若仍检测到脸(半侧/未完全背对)，用鼻子偏移判断正往哪一侧转。
        if nose is not None and nose_off is not None:
            s = 1 if nose_off > 0 else (-1 if nose_off < 0 else 0)
            a = abs(nose_off)
            if a >= 0.4:
                ang = 165.0 * s
            else:
                ang = 175.0 if s == 0 else 175.0 * s
            tag = 'back'
            return ang, tag, s
        else:
            return 175.0, 'back', 0


# ---------------- 主流程 ----------------
def main():
    global SRC_VIDEO, OUT_VIDEO, MAX_FRAMES
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--frames', type=int, default=0, help='0=全部')
    ap.add_argument('--numbers', action='store_true',
                    help='为多个人物绘制稳定编号（可选，默认关闭=干净版）')
    ap.add_argument('--numbers-color', default='200,200,200',
                    help='编号颜色 RGB，如 200,200,200（默认低调灰，不干扰二次生成）')
    ap.add_argument('--export-kp', default='',
                    help='可选：把每帧检测框+关键点+朝向角导出为 .npy（供朝向预览层重绘用）')
    a = ap.parse_args()
    SRC_VIDEO = a.input
    OUT_VIDEO = a.output
    MAX_FRAMES = a.frames
    NUMBERS = a.numbers
    try:
        NC = tuple(int(v) for v in a.numbers_color.split(','))
    except Exception:
        NC = (200, 200, 200)
    if NUMBERS:
        print(f'编号显示已开启 | 颜色 RGB={NC}')

    det_session = ort.InferenceSession(DET_PATH, providers=['CPUExecutionProvider'])
    pose_session = ort.InferenceSession(POSE_PATH, providers=['CPUExecutionProvider'])
    # 确认姿态模型实际输入尺寸
    inp = pose_session.get_inputs()[0]
    print('Pose input name:', inp.name, 'shape:', inp.shape)
    cap = cv2.VideoCapture(SRC_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Source: {SRC_VIDEO} | {W}x{H} | {fps}fps | {total} frames')
    os.makedirs(os.path.dirname(OUT_VIDEO) or '.', exist_ok=True)
    writer = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    n_frames = total if MAX_FRAMES <= 0 else min(MAX_FRAMES, total)
    t0 = time.time()
    prev_centers = []   # 编号跟踪：[(cx, cy, id, age, hist), ...] 外观辅助
    num_frames = 0
    kp_export = [] if a.export_kp else None   # [(frame_idx, boxes, kp_info, angles, tags)]
    for idx in range(0, n_frames, STEP):
        ok = cap.grab()
        if not ok:
            break
        # grab 后直接读取当前帧
        ok, frame = cap.retrieve()
        if not ok:
            break
        # DWPose 模型期望 BGR; ComfyUI 输入 RGB, 直接转成 BGR 给模型
        img = frame.copy()  # frame 是 BGR
        # 检测
        boxes = inference_detector(det_session, img, detect_classes=(0,))
        if boxes is None or len(boxes) == 0:
            canvas = np.zeros((H, W, 3), dtype=np.uint8)
            writer.write(canvas)
            if kp_export is not None:
                kp_export.append((idx, np.zeros((0, 4), float), np.zeros((0, 133, 3), float),
                                  np.zeros((0,), float), []))
        else:
            keypoints, scores = inference_pose(pose_session, boxes, img, POSE_INPUT)
            # 合并 (K,2)+(K,) -> (N, K, 3)
            keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
            # neck 补点 + MMPose -> OpenPose 重排 (官方 wholebody.py)
            neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
            neck[:, 2:4] = np.logical_and(keypoints_info[:, 5, 2:4] > 0.3,
                                          keypoints_info[:, 6, 2:4] > 0.3).astype(int)
            new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
            mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
            openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
            new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
            keypoints_info = new_keypoints_info

            canvas = np.zeros((H, W, 3), dtype=np.uint8)
            if NUMBERS:
                # 身份色模式：按稳定 id 为每个人物分配专属色，整具骨架同色 + 中心纯色徽章（无数字）
                ids = assign_ids(boxes, prev_centers, H, frame=frame)
                colors = [id_color(int(p)) for p in ids]
                for inst, col in zip(keypoints_info, colors):
                    draw_wholebody(canvas, inst, color=col)
                draw_id_badges(canvas, boxes, colors)
            else:
                # 默认干净版：官方默认肢彩虹色（不按人物分配身份）
                draw_wholebody(canvas, keypoints_info)
            if kp_export is not None:
                angles = np.zeros((len(keypoints_info),), float)
                tags = []
                for pi, inst in enumerate(keypoints_info):
                    ang, tag, _ = estimate_orientation(inst, W)
                    angles[pi] = ang
                    tags.append(tag)
                kp_export.append((idx, np.array(boxes, float), keypoints_info.copy(),
                                  angles, tags))
            if SHOW and SEG_ALPHA > 0:
                canvas = cv2.addWeighted(frame, SEG_ALPHA, canvas, 1 - SEG_ALPHA, 0)
            writer.write(canvas)
            if NUMBERS:
                num_frames += 1

        if idx % 25 == 0 or idx == n_frames - 1:
            el = time.time() - t0
            print(f'frame {idx+1}/{n_frames} | {el:.1f}s elapsed')

    cap.release()
    writer.release()
    if kp_export is not None:
        os.makedirs(os.path.dirname(a.export_kp) or '.', exist_ok=True)
        np.save(a.export_kp, dict(
            frame_idx=[e[0] for e in kp_export],
            boxes=[e[1] for e in kp_export],
            kp=[e[2] for e in kp_export],
            angle=[e[3] for e in kp_export],
            tag=[e[4] for e in kp_export],
        ))
        print('Keypoints exported ->', a.export_kp)
    print('DONE ->', OUT_VIDEO)

if __name__ == '__main__':
    main()
