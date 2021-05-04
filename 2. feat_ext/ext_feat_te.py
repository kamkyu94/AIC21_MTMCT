import cv2
import sys
import copy
import torch
import config
import pickle
import torch.nn as nn
import torch.nn.functional as F
import nets.estimator as estimator
from torchvision import transforms
import utils.img_trans as img_trans

# GPU setting
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Paths
data_path = '../../dataset/AIC21_Track3/test/'
check_path = config.save_path + 'resnext_17.t7'

# Define model, Restore trained weight
model = estimator.Estimator()
model = nn.DataParallel(model).cuda()
model.load_state_dict(torch.load(check_path))


def extract_feature(pickle_path, save_name):
    # Read scene and camera list
    with open(pickle_path, 'rb') as f:
        te_box = pickle.load(f)

    print('Start feature extraction')
    with torch.no_grad():
        # Set model as evaluation
        model.eval()

        # Start
        cnt = 0
        te_feat = {}
        for scene in te_box.keys():
            te_feat[scene] = {}
            for cam in te_box[scene].keys():
                te_feat[scene][cam] = {}
                for f_num in te_box[scene][cam].keys():
                    te_feat[scene][cam][f_num] = []
                    for b_idx, box in enumerate(te_box[scene][cam][f_num]):
                        # Decode
                        scene, cam, f_num, _, left, top, w, h, img_w, img_h = box[:10]
                        left, top, w, h = round(left), round(top), round(w), round(h)

                        # Skip box outside of image
                        if left < img_w and top < img_h and 0 < left + w and 0 < top + h:
                            # Calculate valid patch coordinate
                            new_top, new_bot = max(top, 0), min(top + h, img_h)
                            new_left, new_right = max(left, 0), min(left + w, img_w)
                            new_w, new_h = (new_right - new_left), (new_bot - new_top)

                            # Skip with box size
                            if 660 <= new_w * new_h and 0.5 <= (new_w * new_h) / (w * h):
                                # Get patch
                                frame_path = data_path + '/%s/%s/frame/%s_f%04d.jpg' % (scene, cam, cam, f_num)
                                frame = cv2.imread(frame_path)[:, :, [2, 1, 0]]
                                patch = frame[new_top:new_bot, new_left:new_right, :].copy()
                                patch, _, _, _ = img_trans.letterbox(patch)
                                patch = transforms.ToTensor()(patch).unsqueeze(0).cuda()

                                # Inference
                                _, feat, _ = model(patch)
                                feat = F.normalize(feat, dim=1)
                                feat = feat.squeeze().cpu().numpy()

                                # Store extracted feature
                                te_feat[scene][cam][f_num].append(copy.deepcopy(te_box[scene][cam][f_num][b_idx][:15]))
                                te_feat[scene][cam][f_num][-1].append(feat)

                                # Logging
                                cnt += 1
                                if cnt % 1000 == 0:
                                    print('Feature extraction %d finished' % cnt)
                                    sys.stdout.flush()

    # Save final trajectory features and its information
    with open(save_name, 'wb') as handle:
        pickle.dump(te_feat, handle, protocol=pickle.HIGHEST_PROTOCOL)


extract_feature('../outputs/1. det/mask_rcnn_0.2/det_small.pickle',
                '../outputs/1. det/mask_rcnn_0.2/det_small_feat_res8.pickle')
