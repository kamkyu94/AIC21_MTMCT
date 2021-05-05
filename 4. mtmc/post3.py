import cv2
import pickle
import numpy as np

zone_dict = {'c041': ['empty.png', 'empty.png', 'from_c042.png', 'to_c042.png'],
             'c042': ['from_c041.png', 'to_c041.png', 'from_c043.png', 'to_c043.png'],
             'c043': ['from_c042.png', 'to_c042.png', 'from_c044.png', 'to_c044.png'],
             'c044': ['from_c043.png', 'to_c043.png', 'from_c045.png', 'to_c045.png'],
             'c045': ['from_c044.png', 'to_c044.png', 'from_c046.png', 'to_c046.png'],
             'c046': ['from_c045.png', 'to_c045.png', 'empty.png', 'empty.png']}

# Read mtsc results
data_path = '../../dataset/AIC21_Track3/test/'
result_path = '../outputs/2. mtsc/mask_rcnn_0.2/det_small_mtsc_v2_res8_fut3_del3_v2_post12'
with open(result_path + '.pickle', 'rb') as f:
    result = pickle.load(f)

# Filtering 2
track_filt = {}
track_filt_cnt, bbox_filt_cnt = 0, {}
for scene in result.keys():
    track_filt[scene] = {}
    for cam in result[scene].keys():
        # Read zones
        zone_names = zone_dict[cam]
        zones = [cv2.imread(data_path + '%s/%s/zone/%s' % (scene, cam, z))[:, :, 0] for z in zone_names]
        zones = [(z > 0).astype('int32') for z in zones]
        bbox_filt_cnt[cam] = 0

        # For each trajectories
        track_filt[scene][cam] = []
        for track in result[scene][cam]:
            # Get info
            scene, cam, obj_id = track[0][0], track[0][1], track[0][3]

            # Get info
            first_x, first_y, last_x, last_y = track[0][10], track[0][11], track[-1][10], track[-1][11]
            first_x, first_y = round(np.ceil(first_x)), round(np.ceil(first_y))
            last_x, last_y = round(np.floor(last_x)), round(np.floor(last_y))

            # Process start
            if zones[0][first_y, first_x] == 1:
                track.insert(0, 'from_previous_cam')
            elif zones[2][first_y, first_x] == 1:
                track.insert(0, 'from_next_cam')

            # Process last
            if zones[1][last_y, last_x] == 1:
                track.append('to_previous_cam')
            elif zones[3][last_y, last_x] == 1:
                track.append('to_next_cam')

            # Filter out
            if type(track[0]) is not str and type(track[-1]) is not str:
                continue

            # Add filtered information to the new dictionary
            track_filt[scene][cam].append(track)
            track_filt_cnt += 1

            # Add Count
            bbox_filt_cnt[cam] += len([box for box in track if type(box) is not str])

        # Logging
        print('%s_%s finished' % (scene, cam))

# Save final trajectory features and its information
with open(result_path + '3.pickle', 'wb') as handle:
    pickle.dump(track_filt, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Logging
print('trajectories %d' % track_filt_cnt)
print('Bbox: ', bbox_filt_cnt, ', Sum: ', sum(list(bbox_filt_cnt.values())))
