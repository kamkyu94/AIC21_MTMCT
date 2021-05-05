import cv2
import torch
import copy
import config
import numpy as np
import scipy.optimize as opt
import torch.nn.functional as F
from torchvision import transforms
from utils import img_trans, tracker


# For post1.py
def get_last_valid_info(track, thr):
    last_info = None
    for jdx in range(-1, -len(track)-1, -1):
        if thr <= track[jdx][14]:
            last_info = copy.deepcopy(track[jdx])
            break
    return last_info


def trim(track):
    last_box = get_last_valid_info(track, 0.2)
    track = [box for box in track if box[2] <= last_box[2]]
    return track


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


def get_min_dist(track1, track2):

    # Find best object scores
    track1_obj_scores = [box[14] for box in track1]
    track1_best_obj_scores = sorted(track1_obj_scores)[-5:]
    track2_obj_scores = [box[14] for box in track2]
    track2_best_obj_scores = sorted(track2_obj_scores)[-5:]

    dist = []
    for track1_best_obj_score in track1_best_obj_scores:
        # Get box and feature
        feat1 = track1[track1_obj_scores.index(track1_best_obj_score)][15]
        for track2_best_obj_score in track2_best_obj_scores:
            # Get box and feature, Measure distance
            feat2 = track2[track2_best_obj_scores.index(track2_best_obj_score)][15]
            dist.append(np.sqrt(np.sum((feat1 - feat2) ** 2)))

    return np.min(dist)


def construct_dist_mat(tracks1, tracks2):
    # Generate empty array, Generate connectivity matrix,
    dist = np.ones((len(tracks1), len(tracks2))) * 1000
    last_boxes = copy.deepcopy([track[-1] for track in tracks1])
    start_boxes = copy.deepcopy([track[0] for track in tracks2])

    iou = check_iou(last_boxes, start_boxes)
    con = 0.5 <= iou[:, :, 0]
    con += 0.66 <= iou[:, :, 1]
    con += 0.66 <= iou[:, :, 2]
    con = 1 <= con

    # Measure distance when 'a' and 'b' are able to connect
    for idx, track1 in enumerate(tracks1):
        for jdx, track2 in enumerate(tracks2):
            if idx < jdx:
                if 0 < track2[0][2] - track1[-1][2] <= 100 and con[idx, jdx]:
                    dist[idx, jdx] = get_min_dist(track1, track2)

    # Post process dist mat
    for idx in range(dist.shape[0]):
        for jdx in range(dist.shape[1]):
            dist[idx, jdx] = dist[idx, jdx] if dist[idx, jdx] <= 0.837 else 1000

    return dist


def fill(track1, track2):
    track1 = copy.deepcopy(track1)
    track2 = copy.deepcopy(track2)
    for i in range(track2[0][2] - track1[-1][2] - 1):
        last_box = copy.deepcopy(track1[-1])
        last_box[2] += 1
        last_box[14] = 0.15
        track1.append(last_box)

    return track1 + track2


def rematch_static(tracks):
    tracks_run = copy.deepcopy([track for track in tracks if not check_static(track)])
    tracks_static = copy.deepcopy([track for track in tracks if check_static(track)])

    while True:
        # Construct distance matrix
        dist = construct_dist_mat(tracks_static, tracks_static)

        # Break condition
        if np.sum(dist <= 0.837) == 0:
            break

        # Hungarian
        row_ind, col_ind = opt.linear_sum_assignment(dist)
        row_ind, col_ind = list(row_ind), list(col_ind)

        # Match
        for idx in range(len(row_ind)):
            if dist[row_ind[idx], col_ind[idx]] <= 0.837:
                tracks_static[col_ind[idx]] = fill(tracks_static[row_ind[idx]], tracks_static[col_ind[idx]])
                tracks_static[row_ind[idx]] = 0

        tracks_static = [track for track in tracks_static if track != 0]

    return tracks_run + tracks_static


def rematch_run(tracks):
    tracks_static = copy.deepcopy([track for track in tracks if check_static(track)])
    tracks_run = copy.deepcopy([track for track in tracks if not check_static(track)])

    while True:
        # Construct distance matrix
        dist = construct_dist_mat(tracks_run, tracks_run)

        # Break condition
        if np.sum(dist <= 0.837) == 0:
            break

        # Hungarian
        row_ind, col_ind = opt.linear_sum_assignment(dist)
        row_ind, col_ind = list(row_ind), list(col_ind)

        # Match
        for idx in range(len(row_ind)):
            if dist[row_ind[idx], col_ind[idx]] <= 0.837:
                tracks_run[col_ind[idx]] = tracks_run[row_ind[idx]] + tracks_run[col_ind[idx]]
                tracks_run[row_ind[idx]] = 0

        tracks_run = [track for track in tracks_run if track != 0]

    return tracks_static + tracks_run


def rematch_static_run(tracks):
    tracks_static = copy.deepcopy([track for track in tracks if check_static(track)])
    tracks_run = copy.deepcopy([track for track in tracks if not check_static(track)])

    while True:
        # Construct distance matrix,
        dist = construct_dist_mat(tracks_static, tracks_run)

        # Break condition
        if np.sum(dist <= 0.837) == 0:
            break

        # Hungarian
        row_ind, col_ind = opt.linear_sum_assignment(dist)
        row_ind, col_ind = list(row_ind), list(col_ind)

        # Match
        for idx in range(len(row_ind)):
            if dist[row_ind[idx], col_ind[idx]] <= 0.837:
                tracks_run[col_ind[idx]] = tracks_static[row_ind[idx]] + tracks_run[col_ind[idx]]
                tracks_static[row_ind[idx]] = 0

        tracks_static = [track for track in tracks_static if track != 0]

    return tracks_run + tracks_static


def rematch_run_static(tracks):
    tracks_run = copy.deepcopy([track for track in tracks if not check_static(track)])
    tracks_static = copy.deepcopy([track for track in tracks if check_static(track)])

    while True:
        # Construct distance matrix,
        dist = construct_dist_mat(tracks_run, tracks_static)

        # Break condition
        if np.sum(dist <= 0.837) == 0:
            break

        # Hungarian
        row_ind, col_ind = opt.linear_sum_assignment(dist)
        row_ind, col_ind = list(row_ind), list(col_ind)

        # Match
        for idx in range(len(row_ind)):
            if dist[row_ind[idx], col_ind[idx]] <= 0.837:
                tracks_static[col_ind[idx]] = tracks_run[row_ind[idx]] + tracks_static[col_ind[idx]]
                tracks_run[row_ind[idx]] = 0

        tracks_run = [track for track in tracks_run if track != 0]

    return tracks_static + tracks_run


def check_short(track):
    # Check length
    last_box = get_last_valid_info(track, 0.1)
    if last_box[2] - track[0][2] < config.min_len:
        return True
    else:
        return False


def check_static(track):
    iou = calc_iou(track[0], get_last_valid_info(track, 0.1))
    if 0.5 <= iou[0] or 0.66 <= iou[1] or 0.66 <= iou[2]:
        return True
    else:
        return False


# For post2.py
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


def compare_feat(track, box, model):
    # Get last info
    last_box = get_last_valid_info(track.track, 0.2)

    # Read patch, Extract feature
    last_feat = last_box[15]
    patch = read_patch(box)
    _, feat, _ = model(patch)

    # Measure distance
    dist = np.sqrt(np.sum((last_feat - F.normalize(feat, dim=1).squeeze().cpu().numpy()) ** 2))
    return 0.1 if dist <= config.dist_thr else 0


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


def lengthen(track, model):
    # Forward
    t_f = tracker.Track(track[0])
    for i in range(1, len(track)):
        t_f.update(track[i])

    # Backward
    t_b = tracker.Track(track[-1])
    for i in range(len(track)-2, -1, -1):
        t_b.update(track[i])

    # Forward
    while True:
        # Check finish
        if check_fin(t_f):
            break

        # Get next box, check
        next_box = t_f.get_next_box()
        next_box[2] += 1
        if 2001 < next_box[2]:
            break

        # Compare feature
        with torch.no_grad():
            next_box[14] = compare_feat(t_f, next_box, model)
            if next_box[14] == 0.1:
                t_f.update(next_box)
                track.append(next_box)
            else:
                break

    # Backward
    while True:
        # Check finish
        if check_fin(t_b):
            break

        # Get next box, check
        next_box = t_b.get_next_box()
        next_box[2] -= 1
        if next_box[2] < 1:
            break

        # Compare feature
        with torch.no_grad():
            next_box[14] = compare_feat(t_b, next_box, model)
            if next_box[14] == 0.1:
                t_b.update(next_box)
                track.insert(0, next_box)
            else:
                break

    return track


# For post3.py
def get_min_f_num_diff(a_track, b_track, con):
    # Calculate speed
    a_track = copy.deepcopy([box for box in a_track if type(box) is not str])
    b_track = copy.deepcopy([box for box in b_track if type(box) is not str])

    if con == 1:
        a_lat, a_lon = a_track[-1][12], a_track[-1][13]
        b_lat, b_lon = b_track[0][12], b_track[0][13]
        dist = np.sin(np.deg2rad(b_lat)) * np.sin(np.deg2rad(a_lat)) \
               + np.cos(np.deg2rad(b_lat)) * np.cos(np.deg2rad(a_lat)) * np.cos(np.deg2rad(b_lon - a_lon))
    else:
        b_lat, b_lon = b_track[-1][12], b_track[-1][13]
        a_lat, a_lon = a_track[0][12], a_track[0][13]
        dist = np.sin(np.deg2rad(a_lat)) * np.sin(np.deg2rad(b_lat)) \
               + np.cos(np.deg2rad(a_lat)) * np.cos(np.deg2rad(b_lat)) * np.cos(np.deg2rad(a_lon - b_lon))

    # Final calculation
    dist = np.rad2deg(np.arccos(dist)) * 60 * 1.1515 * 1.609344
    min_f_num_diff = dist / (100 / 36000)

    return min_f_num_diff
