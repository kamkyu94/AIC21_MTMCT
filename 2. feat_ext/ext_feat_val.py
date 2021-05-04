import cv2
import sys
import torch
import config
import pickle
import torch.nn as nn
from utils import img_trans
import torch.nn.functional as F
import nets.estimator as estimator
from torchvision import transforms

# GPU setting
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Paths
data_path = '../../dataset/AIC21_Track3/validation/'
check_path = config.save_path + 'resnext_20.t7'

# Define model, Restore trained weight
model = estimator.Estimator()
model = nn.DataParallel(model).cuda()
model.load_state_dict(torch.load(check_path))


def extract_feature(pickle_path, save_name):
    # Read scene and camera list
    with open(pickle_path, 'rb') as f:
        val_box = pickle.load(f)

    print('Start feature extraction')
    with torch.no_grad():
        # Set model as evaluation
        model.eval()

        # Start
        cnt, val_feat = 0, {}
        for obj_id in val_box.keys():
            val_feat[obj_id] = []
            for b_idx, box in enumerate(val_box[obj_id]):
                # Decode
                scene, cam, f_num, obj_id, left, top, w, h, img_w, img_h = box[:10]

                # Get patch
                frame_path = data_path + '/%s/%s/frame/%s_f%04d.jpg' % (scene, cam, cam, f_num)
                frame = cv2.imread(frame_path)[:, :, [2, 1, 0]]
                patch = frame[max(top, 0):min(top+h, img_h), max(left, 0):min(left+w, img_w), :].copy()
                patch, _, _, _ = img_trans.letterbox(patch)
                patch = transforms.ToTensor()(patch).unsqueeze(0).cuda()

                # Inference
                _, feat, _ = model(patch)
                feat = F.normalize(feat, dim=1)
                feat = feat.squeeze().cpu().numpy()

                # Store extracted feature
                val_feat[obj_id].append([scene, cam, f_num, obj_id, feat])
                cnt += 1

                # Logging
                if cnt % 1000 == 0:
                    print('Feature extraction %d / 185427 finished' % cnt)
                    sys.stdout.flush()

    # Save final trajectory features and its information
    with open(config.save_path + save_name, 'wb') as handle:
        pickle.dump(val_feat, handle, protocol=pickle.HIGHEST_PROTOCOL)


extract_feature('infos/val_gt_info.pickle', 'val.pickle')
extract_feature('infos/val_gt_info_random_box_size.pickle', 'val_rand.pickle')
