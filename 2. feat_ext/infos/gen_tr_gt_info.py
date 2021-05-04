import os
import cv2
import pickle
import utils.img2gps as img2gps

# Data Path
data_path = '../../../dataset/AIC21_Track3/train/'

# # Create patches directory
# save_path = data_path + 'patch_gt/'
# if not os.path.exists(save_path):
#     os.mkdir(save_path)

# Read detection result files and save its information with filtering
info_obj_id = {}
scenes = os.listdir(data_path)
for scene in scenes:
    # # Skip
    # if scene == 'patch_gt':
    #     continue

    cams = os.listdir(data_path + scene)
    for cam in cams:
        # Set path
        cam_path = data_path + scene + '/%s/' % cam

        # Read RoI
        roi = cv2.imread(cam_path + 'roi.jpg').astype('float32')[:, :, 0] / 255
        img_h, img_w = roi.shape[0], roi.shape[1]

        # Read gt result file
        gt = open(cam_path + 'gt/gt.txt', 'r').readlines()
        for line in gt:
            line = line.split(',')
            f_num, obj_id = int(line[0]), int(line[1])
            left, top, w, h = round(float(line[2])), round(float(line[3])), round(float(line[4])), round(float(line[5]))

            # Reorder object id
            obj_id = (obj_id - 1) if obj_id < 96 else (obj_id - 146)

            # # Read frame image
            # img_path = cam_path + 'frame/%s_f%04d.jpg' % (cam, f_num)
            # frame_img = cv2.imread(img_path)
            #
            # # Save bbox patch
            # bbox = frame_img[top:top+h, left:left+w, :]
            # cv2.imwrite(save_path + '%s_%s_f%04d_%d.jpg' % (scene, cam, f_num, obj_id), bbox)

            # Calculate gps
            gps = img2gps.to_gps_coord('train', scene, cam, [left, top, w, h])

            # Calculate info
            x, y = left + w * 0.5, top + h * 0.8

            # Save info
            save_info = [scene, cam, f_num, obj_id, left, top, w, h, img_w, img_h, x, y, gps[0], gps[1], 1]

            # Save gt result
            if obj_id not in info_obj_id.keys():
                info_obj_id[obj_id] = []
            info_obj_id[obj_id].append(save_info)

        # print current status
        print('%s_%s finished' % (scene, cam))

# Save save_info
with open('../../2. feat_ext/outputs/tr_gt_info.pickle', 'wb') as f:
    pickle.dump(info_obj_id, f, pickle.HIGHEST_PROTOCOL)
