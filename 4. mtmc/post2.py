import torch
import config
import pickle
import torch.nn as nn
from utils import utils
import nets.estimator as estimator

# Read mtsc results
data_path = '../../dataset/AIC21_Track3/test/'
result_path = '../outputs/2. mtsc/mask_rcnn_0.2/det_small_mtsc_v2_res8_fut3_del3_v2_post1'
with open(result_path + '.pickle', 'rb') as f:
    result = pickle.load(f)

# MTSC results
# dict[scene][cam] = [trajectory, ...]
# trajectory = [[scene, cam, f_num, obj_id, left, top, w, h, img_w, img_h, center x, center y, gps_x, gps_y,
#                objectiveness score, curr feat, smooth feat], ...]

# Define and load model
model = estimator.Estimator()
model = nn.DataParallel(model).cuda()
model.load_state_dict(torch.load(config.weight_path))
model.eval()

# Post processing 2 - Lengthening
bbox_len_cnt = {}
for scene in result.keys():
    for cam in result[scene].keys():
        bbox_len_cnt[cam] = 0
        for t_idx, track in enumerate(result[scene][cam]):
            result[scene][cam][t_idx] = utils.lengthen(result[scene][cam][t_idx], model)
            bbox_len_cnt[cam] += len(result[scene][cam][t_idx])

# Save final trajectory features and its information
with open(result_path + '2.pickle', 'wb') as handle:
    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Logging
print('\nLengthen')
print('Bbox: ', bbox_len_cnt, ', Sum: ', sum(list(bbox_len_cnt.values())))
