import os
import cv2
import copy
import pickle
import numpy as np
import img2gps as img2gps
import pre_process as pre_process

# Set
basic_path = '../../dataset/AIC21_Track3/test/S06/'

# Visualize and save
det_info = {'S06': {}}
cams = os.listdir(basic_path)
for cam in cams:
    # Newly add
    if cam not in det_info.keys():
        det_info['S06'][cam] = {}

    # Open detection file
    det_file_path = basic_path + '/%s/det/det_mask_rcnn.txt' % cam
    det_file = np.loadtxt(det_file_path, dtype=np.float32, delimiter=',').reshape(-1, 10)

    # Read RoI
    roi = cv2.imread(basic_path + cam + '/roi.jpg').astype('float32')[:, :, 0] / 255
    img_h, img_w = roi.shape[0], roi.shape[1]

    # Filter with objectiveness score and size
    det_file = det_file[det_file[:, 6] >= 0.2, :]
    det_file = det_file[det_file[:, 4] * det_file[:, 5] >= 660, :]

    # For each frame
    for f_num in range(1, int(np.max(det_file[:, 0] + 1))):
        # Newly add
        if f_num not in det_info['S06'][cam].keys():
            det_info['S06'][cam][f_num] = []

        # Select detection results
        det_results = copy.deepcopy(det_file[det_file[:, 0] == f_num, :])

        # Pre-process with IoU
        det_results = det_results[:, 2:7]
        det_results = [det_result for det_result in det_results]
        det_results = pre_process.merge_boxes(det_results)

        # Save each bbox
        for det_result in det_results:
            # Read result
            left, top, w, h, score = det_result

            # Calculate gps
            gps = img2gps.to_gps_coord('test', 'S06', cam, [left, top, w, h])

            # Calculate info
            x, y = left + w * 0.5, top + h * 0.8

            # Save info
            save_info = ['S06', cam, f_num, 0, left, top, w, h, img_w, img_h, x, y, gps[0], gps[1], score]
            det_info['S06'][cam][f_num].append(save_info)

    # print current status
    print('S06_%s Finished' % cam)


# Save save_info
with open('../outputs/1. det/mask_rcnn_0.2/det_small.pickle', 'wb') as f:
    pickle.dump(det_info, f, pickle.HIGHEST_PROTOCOL)
