import cv2
import copy
import config
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import utils.img_trans as img_trans


def calc_iou(a, b):
    # Decode information
    a_left, a_top, a_w, a_h = a[4:8]
    b_left, b_top, b_w, b_h = b[4:8]

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


# Construct distance matrix between 'a' and 'b'
def construct_dist_mat(tracks, boxes, iou):
    # Generate empty array, Generate connectivity matrix,
    dist = np.ones((len(tracks), len(boxes))) * 1000
    con = config.iou_thr <= iou[:, :, 0]

    # Measure distance when 'a' and 'b' are able to connect
    for idx, track in enumerate(tracks):
        for jdx, box in enumerate(boxes):
            if con[idx, jdx]:
                dist[idx, jdx] = np.sqrt(np.sum((track.track[-1][15] - box[15])**2))

    return dist


def get_last_valid_info(track, thr):
    last_info = None
    for jdx in range(-1, -len(track)-1, -1):
        if thr <= track[jdx][14]:
            last_info = copy.deepcopy(track[jdx])
            break
    return last_info


def check_fin(track):
    # Decode
    img_w, img_h = track.track[-1][8], track.track[-1][9]

    # Calculate next position of 4 corners and center point
    next_left, next_top, next_right, next_bot, next_box_size = track.get_future(3)

    # Check next position and box size
    if next_left < 0 or next_right < 0 or img_w <= next_left or img_w <= next_right \
            or next_top <= 0 or next_bot <= 0 or img_h < next_top or img_h < next_bot:
        return True

    # Calculate resolution pattern
    if next_box_size < 660:
        return True

    return False


def read_patch(box):
    # Decode
    scene, cam, f_num, _, left, top, w, h, img_w, img_h = box[0:10]
    left, top, w, h = round(left), round(top), round(w), round(h)

    # Read Frame, Get patch
    frame_path = config.data_path + '/%s/frame/%s_f%04d.jpg' % (cam, cam, f_num)
    frame = cv2.imread(frame_path)[:, :, [2, 1, 0]]
    patch = frame[max(top, 0):min(top + h, img_h), max(left, 0):min(left + w, img_w), :]
    patch, _, _, _ = img_trans.letterbox(patch)
    patch = transforms.ToTensor()(patch).unsqueeze(0).cuda()

    return patch


def compare_feat(track, new_box, model):
    # Get last info
    last_box = get_last_valid_info(track.track, 0.2)

    # Read patch, Extract feature
    last_feat = last_box[15]
    new_patch = read_patch(new_box)
    _, new_feat, _ = model(new_patch)

    # Measure distance
    dist = np.sqrt(np.sum((last_feat - F.normalize(new_feat, dim=1).squeeze().cpu().numpy()) ** 2))
    return 0.1 if dist <= config.dist_thr else 0


