import copy
import numpy as np


def calc_iou(a, b):
    # Decode information
    a_left, a_top, a_w, a_h, _ = a
    b_left, b_top, b_w, b_h, _ = b

    # Calculate iou
    left, top = max(a_left, b_left), max(a_top, b_top)
    right, bot = min(a_left + a_w, b_left + b_w), min(a_top + a_h, b_top + b_h)
    inter = max(0, right - left) * max(0, bot - top)

    # Form
    iou = np.zeros((3,))
    iou[0] = inter / (a_w * a_h + b_w * b_h - inter)
    iou[1] = inter / (a_w * a_h)
    iou[2] = inter / (b_w * b_h)

    return iou


def check_iou(a_boxes, b_boxes):
    # Check iou between boxes 'a' and boxes 'b'
    iou = np.zeros((len(a_boxes), len(b_boxes), 3))
    for idx, a in enumerate(a_boxes):
        for jdx, b in enumerate(b_boxes):
            iou[idx, jdx, :] = calc_iou(a, b)
    return iou


# Filter over detected boxes
def merge_boxes(boxes):
    while True:
        # Check connectivity of boxes
        iou = check_iou(boxes, boxes)
        con = 0.5 <= iou[:, :, 0]
        con += 0.66 <= iou[:, :, 1]
        con += 0.66 <= iou[:, :, 2]
        con = 1 <= con

        # End condition
        if np.sum(con) == len(boxes):
            break

        # Pick idx to delete
        del_idx_can = np.where(np.sum(con, axis=0) >= 2)[0]
        scores = [boxes[d_idx_c][-1] for d_idx_c in del_idx_can]
        del_idx = del_idx_can[scores.index(min(scores))]
        boxes.pop(del_idx)

    return boxes
