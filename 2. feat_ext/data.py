import cv2
import copy
import pickle
import config
import random
from utils import img_trans
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class TrainDataManager(Dataset):
    def __init__(self):
        super(TrainDataManager, self).__init__()
        # Read info file
        with open(config.tr_info_obj_id_path, 'rb') as f:
            self.info_obj_id = pickle.load(f)

        # Set Paths
        self.image_dir = config.tr_image_dir

        # Define transform
        self.jitter = transforms.ColorJitter(0, 0, 0, 0)
        self.flip = transforms.RandomHorizontalFlip()
        self.to_tensor = transforms.ToTensor()

        # Initially generate list of usable object ID
        # We will sample the object only when ID is included in this list
        # Element will be discarded when the number of image path of the corresponding object ID is less than k_num
        self.possible_obj_id = None

        # Generate data queue and pk batch queue
        self.new_info_obj_id = None
        self.pk_batch_queue = None

    def gen_new_info_obj_id(self):
        new_info_obj_id = copy.deepcopy(self.info_obj_id)
        for obj_id in new_info_obj_id.keys():
            random.shuffle(new_info_obj_id[obj_id])

        return new_info_obj_id

    def gen_pk_batch_queue(self):
        pk_batch_queue = []
        while len(self.possible_obj_id) >= config.p_num:
            # Sample p queue
            p_queue = random.sample(self.possible_obj_id, config.p_num)

            # Construct pk batch queue
            for obj_id in p_queue:
                for _ in range(config.k_num):
                    pk_batch_queue.append(self.new_info_obj_id[obj_id].pop())

                if len(self.new_info_obj_id[obj_id]) < config.k_num:
                    self.possible_obj_id.remove(obj_id)

        return pk_batch_queue

    def read_image(self, info):
        # Decode info
        scene, cam, f_num, obj_id = info[:4]

        # Decode info, Randomly sample size and location of bounding box
        left, top, w, h, img_w, img_h = info[4:10]
        r_w, r_h = random.uniform(0.8, 1.6), random.uniform(0.8, 1.2)
        r_left, r_top = random.uniform(0, 1), random.uniform(0, 1)
        w_, h_ = round(w * r_w), round(h * r_h)
        left_, top_ = round(left - (w_ - w) * r_left), round(top - (h_ - h) * r_top)

        # Read Frame, Get patch
        frame_path = self.image_dir + '/%s/%s/frame/%s_f%04d.jpg' % (scene, cam, cam, f_num)
        frame = cv2.imread(frame_path)[:, :, [2, 1, 0]]
        patch = frame[max(top_, 0):min(top_ + h_, img_h), max(left_, 0):min(left_ + w_, img_w), :].copy()

        # Augmentation
        patch = np.asarray(self.flip(self.jitter(Image.fromarray(patch))))
        patch, _, _, _ = img_trans.letterbox(patch)
        patch = img_trans.random_affine(patch)
        patch = self.to_tensor(patch)

        return patch

    def __len__(self):
        return len(self.pk_batch_queue)

    def __getitem__(self, idx):
        # Get data
        info = self.pk_batch_queue[idx]

        # Read image and form info vector
        patch = self.read_image(info)

        # Create data dictionary
        data = {'patch': patch, 'obj_id': info[3]}

        return data
