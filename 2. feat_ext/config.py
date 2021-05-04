# Data path
tr_image_dir = '../../dataset/AIC21_Track3/train/'
tr_info_obj_id_path = 'infos/tr_gt_info.pickle'

# Save path
save_path = './outputs/resnext_8/'
log_path = save_path + 'log.txt'

# Configurations
seed = 10000
num_epoch = 20
num_warm_up = 5
milestones = [10, 15]

# Training
k_num = 4
p_num = 18
base_lr = 0.01
num_ide_class = 184
batch_size = k_num * p_num

# Patch size
img_h = 320
img_w = 320
