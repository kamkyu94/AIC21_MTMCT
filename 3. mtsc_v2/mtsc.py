import os
import torch
import config
import pickle
import torch.nn as nn
from utils import utils
from utils.tracker import *
import scipy.optimize as opt
import nets.estimator as estimator

# GPU setting
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Set paths
det_path = '../outputs/1. det/mask_rcnn_0.2/'
mtsc_path = '../outputs/2. mtsc/mask_rcnn_0.2/'

# Read result
with open(det_path + 'det_small_feat_res8.pickle', 'rb') as f:
    det_feat = pickle.load(f)

# Detection results
# dict[scene][cam][f_num] = [[scene, cam, f_num, idx, left, top, w, h, img_w, img_h, center x, center y, gps_x, gps_y,
#                             objectiveness score, curr feat, smooth feat], ...]

# Define and load model
model = estimator.Estimator()
model = nn.DataParallel(model).cuda()
model.load_state_dict(torch.load(config.weight_path))
model.eval()


def match(tracks, boxes, dist, dist_thr):
    # Hungarian algorithm with left pairs
    row_ind, col_ind = opt.linear_sum_assignment(dist)
    row_ind, col_ind = list(row_ind), list(col_ind)

    # Check distance between connections, Merge boxes 'b' to tracks 'a'
    con_row_ind, con_col_ind = [], []
    for idx in range(len(row_ind)):
        if dist[row_ind[idx], col_ind[idx]] <= dist_thr:
            # Merge track
            tracks[row_ind[idx]].update(copy.deepcopy(boxes[col_ind[idx]]))

            # Get Connected index
            con_row_ind.append(row_ind[idx])
            con_col_ind.append(col_ind[idx])

    # Occluded tracks
    left_row_ind = [x for x in range(len(tracks)) if x not in con_row_ind]
    con_tracks = copy.deepcopy([tracks[i] for i in con_row_ind])
    left_tracks = copy.deepcopy([tracks[i] for i in left_row_ind])

    # Left boxes
    left_col_ind = [x for x in range(len(boxes)) if x not in con_col_ind]
    left_boxes = copy.deepcopy([boxes[i] for i in left_col_ind])

    return con_tracks, left_tracks, left_boxes


def match_on_tracks_boxes(on_tracks, boxes):
    # Get distance matrix, Match
    iou = utils.check_iou([on_track.get_next_box() for on_track in on_tracks], boxes)
    dist = utils.construct_dist_mat(on_tracks, boxes, iou)
    con_tracks, left_tracks, left_boxes = match(on_tracks, boxes, dist, config.dist_thr)

    # Get iou matrix, Re-match
    iou = utils.check_iou([left_track.get_next_box() for left_track in left_tracks], left_boxes)[:, :, 0]
    con_tracks_, left_tracks_, left_boxes_ = match(left_tracks, left_boxes, 1 - iou, 0.5)

    return con_tracks + con_tracks_, left_tracks_, left_boxes_


def match_occ_tracks_left_boxes(occ_tracks, boxes, on_tracks):
    # Get distance matrix, Match
    iou = utils.check_iou([occ_track.get_next_box() for occ_track in occ_tracks], boxes)
    dist = utils.construct_dist_mat(occ_tracks, boxes, iou)
    con_tracks, left_tracks, left_boxes = match(occ_tracks, boxes, dist, config.dist_thr)

    # # Get iou matrix, Re-match
    # iou = utils.check_iou([left_track.get_next_box() for left_track in left_tracks], left_boxes)[:, :, 0]
    # con_tracks_, left_tracks_, lecheft_boxes_ = match(left_tracks, left_boxes, 1 - iou, 0.5)

    # New starting boxes
    on_tracks = on_tracks + con_tracks # + con_tracks_
    for left_box in left_boxes:
        on_tracks.append(Track(copy.deepcopy(left_box)))

    return left_tracks, on_tracks


# Check trajectory is ended or not
def get_fin_tracks(occ_tracks, f_num):
    # Finish
    fin_idx = []
    for idx, track in enumerate(occ_tracks):
        # If no re-detection is too long
        if config.max_search < (f_num - utils.get_last_valid_info(track.track, 0.2)[2]):
            fin_idx.append(idx)
            continue

        # Check whether this trajectory is finished or not
        if utils.check_fin(track):
            fin_idx.append(idx)
            continue

    # Divide run, stop, finish tracks
    fin_tracks = copy.deepcopy([occ_tracks[i] for i in fin_idx])
    not_fin_idx = [x for x in range(len(occ_tracks)) if x not in fin_idx]
    occ_tracks = copy.deepcopy([occ_tracks[i] for i in not_fin_idx])

    return fin_tracks, occ_tracks


# Update occluded trajectories
def update(occ_tracks):
    for idx, track in enumerate(occ_tracks):
        # Calculate next box
        next_box = track.get_next_box()

        # Compare feature
        with torch.no_grad():
            next_box[14] = utils.compare_feat(track, next_box, model)

        # Add
        occ_tracks[idx].update(next_box)

    return occ_tracks


# Multi-object tracking in single camera
def mtsc(scene, cam):
    # Get frame numbers
    f_nums = list(det_feat[scene][cam].keys())

    # Start condition (Get bounding boxes of frame 0)
    on_tracks, occ_tracks, occ_tracks_, left_boxes, result = [], [], [], [], []

    # Tracking
    for f_num in f_nums:
        # Get bounding boxes of frame T
        boxes = copy.deepcopy(det_feat[scene][cam][f_num])

        # Logging
        print(f_num, 'Track On: %d, Occ: %d, Det: %d' % (len(on_tracks), len(occ_tracks), len(boxes)))

        if (len(on_tracks) + len(occ_tracks)) == 0 and len(boxes) > 0:
            on_tracks = [Track(box) for box in copy.deepcopy(det_feat[scene][cam][f_num])]
            print(f_num, 'Track newly started On: %d' % len(on_tracks))
            continue

        # Match online tracks and detected boxes
        if len(on_tracks) > 0 and len(boxes):
            on_tracks, occ_tracks_, left_boxes = match_on_tracks_boxes(on_tracks, boxes)
            print(f_num, 'Track On: %d, New Occ: %d, Left: %d' % (len(on_tracks), len(occ_tracks_), len(left_boxes)))

            # Delete too short
            occ_tracks_ = [track for track in occ_tracks_ if 3 < len(track.track)]

        # If there is no occluded tracks
        if len(occ_tracks) == 0 and len(left_boxes) > 0:
            for left_box in left_boxes:
                on_tracks.append(Track(copy.deepcopy(left_box)))
            print(f_num, 'New tracks added')

        # Match occluded tracks and left boxes
        elif len(occ_tracks) > 0 and len(left_boxes) > 0:
            on_nums = len(on_tracks)
            occ_tracks, on_tracks = match_occ_tracks_left_boxes(occ_tracks, left_boxes, on_tracks)
            print(f_num, 'Re-connect %d' % (len(on_tracks) - on_nums))

            # Delete too short
            occ_tracks = [track for track in occ_tracks if 3 < len(track.track)]

        # Merge occluded tracks
        occ_tracks += copy.deepcopy(occ_tracks_)

        # Check finished tracks
        if len(occ_tracks) > 0:
            # Exclude and newly start tracks
            fin_tracks, occ_tracks = get_fin_tracks(occ_tracks, f_num)
            result += copy.deepcopy(fin_tracks)
            print(f_num, 'Track Fin: %d' % len(fin_tracks))

        # Update occluded tracks
        if len(occ_tracks) > 0:
            # Classify excluded tracks [delete, occlude, finish]
            occ_tracks = update(occ_tracks)
            print(f_num, 'Updated Occ: %d' % len(occ_tracks))

        # Logging
        print(f_num, 'Track On: %d, Occ: %d, Result: %d\n' % (len(on_tracks), len(occ_tracks), len(result)))

    # Merge
    result += on_tracks
    result += occ_tracks

    # Post process
    result = [r.track for r in result]
    print('Final Track Result: %d' % len(result))

    return result


def map_new_id(result):
    # Reorder in frame number order
    result = sorted(result, key=lambda track: track[0][2])

    # Change idx to object id
    for idx in range(len(result)):
        for jdx in range(len(result[idx])):
            result[idx][jdx][3] = idx

    return result


if __name__ == "__main__":
    # Get mstc results for each camera
    mtsc_results = {}
    for scene in det_feat.keys():
        mtsc_results[scene] = {}
        for cam in det_feat[scene].keys():
            # Tracking
            print('%s_%s starts' % (scene, cam))
            mtsc_results[scene][cam] = map_new_id(mtsc(scene, cam))
            print('')

    # Save result
    with open(mtsc_path + 'det_small_mtsc_v2_res8_fut3_del3.pickle', 'wb') as handle:
        pickle.dump(mtsc_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
