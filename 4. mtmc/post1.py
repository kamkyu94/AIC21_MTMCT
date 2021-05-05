import copy
import pickle
from utils import utils

# Read mtsc results
data_path = '../../dataset/AIC21_Track3/test/'
result_path = '../outputs/2. mtsc/mask_rcnn_0.2/det_small_mtsc_v2_res8_fut3_del3'
with open(result_path + '.pickle', 'rb') as f:
    result = pickle.load(f)

# MTSC results
# dict[scene][cam] = [trajectory, ...]
# trajectory = [[scene, cam, f_num, obj_id, left, top, w, h, img_w, img_h, center x, center y, gps_x, gps_y,
#                objectiveness score, curr feat, smooth feat], ...]

# Post processing 1
track_cnt, bbox_cnt = 0, {}
track_post_cnt, bbox_post_cnt = 0, {}
for scene in result.keys():
    for cam in result[scene].keys():
        # Count
        track_cnt += len(result[scene][cam])
        bbox_cnt[cam] = 0
        for track in result[scene][cam]:
            bbox_cnt[cam] += len(track)

        # Trim, Delete with length, Re-match, Delete static
        result[scene][cam] = [utils.trim(track) for track in result[scene][cam]]
        result[scene][cam] = [track for track in result[scene][cam] if not utils.check_short(track)]
        result[scene][cam] = utils.rematch_static(result[scene][cam])
        result[scene][cam] = utils.rematch_static_run(result[scene][cam])
        result[scene][cam] = utils.rematch_run_static(result[scene][cam])
        result[scene][cam] = utils.rematch_run(result[scene][cam])
        result[scene][cam] = [track for track in result[scene][cam] if not utils.check_static(track)]

        # Count after post processing
        track_post_cnt += len(result[scene][cam])
        bbox_post_cnt[cam] = 0
        for track in result[scene][cam]:
            bbox_post_cnt[cam] += len(track)

# Map new id
result_new_id = copy.deepcopy(result)
for scene in result.keys():
    for cam in result[scene].keys():
        for t_idx in range(len(result[scene][cam])):
            for b_idx in range(len(result[scene][cam][t_idx])):
                result_new_id[scene][cam][t_idx][b_idx][3] = t_idx

# Save final trajectory features and its information
with open(result_path + '_v2_post1.pickle', 'wb') as handle:
    pickle.dump(result_new_id, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Logging
print('\nPost1')
print('Trajectories %d -> %d' % (track_cnt, track_post_cnt))
print('Bbox: ', bbox_cnt, ', Sum: ', sum(list(bbox_cnt.values())))
print('Bbox: ', bbox_post_cnt, ', Sum: ', sum(list(bbox_post_cnt.values())))
