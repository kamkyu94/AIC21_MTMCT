import os
import sys
import torch
import random
import numpy as np
import torch.nn as nn
import config as config
import nets.estimator as estimator
from data import TrainDataManager

# GPU setting
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

# Set Random seed
random.seed(config.seed)
np.random.seed(config.seed)

# Logging
if not os.path.exists(config.save_path):
    os.mkdir(config.save_path)
log = open(config.log_path, 'w')

# DataManager and DataLoader for train and validation
tr_dm = TrainDataManager()
tr_dl = torch.utils.data.DataLoader(tr_dm, batch_size=config.batch_size, shuffle=False, num_workers=2, drop_last=True)

# Define model
model = estimator.Estimator()
model = nn.DataParallel(model).cuda()

# Optimizer, Criterion
opt = torch.optim.SGD(params=model.parameters(), lr=config.base_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
ce = nn.CrossEntropyLoss()
triplet = nn.TripletMarginLoss()

# Learning rate scheduler - cosine annealing
warm_up_sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch: epoch / config.num_warm_up)
step_sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=config.milestones, gamma=0.1)

# Generate masks which is needed to calculate triplet loss
pos_mask = np.zeros((config.batch_size, config.batch_size))
neg_mask = np.zeros((config.batch_size, config.batch_size))
for j in range(config.batch_size):
    for k in range(j + 1, config.batch_size):
        if k < (j // config.k_num + 1) * config.k_num:
            pos_mask[[j, k], [k, j]] = 1
        else:
            neg_mask[[j, k], [k, j]] = 1


def train_one_epoch():
    # Ready
    tr_dm.possible_obj_id = list(range(config.num_ide_class))
    tr_dm.new_info_obj_id = tr_dm.gen_new_info_obj_id()
    tr_dm.pk_batch_queue = tr_dm.gen_pk_batch_queue()

    # Training
    model.train()
    tr_loss, tr_ide_acc, num_iter = 0, 0, 0
    print('Start Training')
    for idx, data in enumerate(tr_dl):
        # Set zero gradient
        sys.stdout.flush()
        opt.zero_grad()

        # Get train batch
        patch, obj_id = data['patch'].cuda(), data['obj_id'].cuda()

        # Forward pass
        feat_tri, _, ide = model(patch)

        # Calculate pairwise distance matrix
        feat = feat_tri.squeeze().detach().cpu().numpy()
        square = np.sum(feat ** 2, 1, keepdims=True)
        dist_mat = np.repeat(square, config.batch_size, 1) + np.transpose(np.repeat(square, config.batch_size, 1))
        dist_mat -= 2 * np.matmul(feat, np.transpose(feat))

        # For positive sampling and negative sampling (Batch Hard)
        dist_mat_pos = np.abs(dist_mat * pos_mask)
        dist_mat_neg = np.abs(dist_mat * neg_mask + (1 - neg_mask) * np.max(dist_mat) * 10)
        pos_idx, neg_idx = np.argmax(dist_mat_pos, 1), np.argmin(dist_mat_neg, 1)

        # Triplet loss
        triplet_loss = triplet(feat_tri, feat_tri[pos_idx, :],  feat_tri[neg_idx, :])

        # Loss, Back-propagation
        loss = ce(ide, obj_id) + triplet_loss
        loss.backward()
        opt.step()

        # Loss, Accuracy
        tr_loss += loss.item()
        pred = ide.argmax(dim=1, keepdim=True)
        tr_ide_acc += pred.eq(obj_id.view_as(pred)).sum().item()
        num_iter += 1

    # Loss, Accuracy
    tr_loss = tr_loss / num_iter
    tr_ide_acc = tr_ide_acc / (num_iter * config.batch_size)

    return tr_loss, tr_ide_acc


def train():
    # Warm-up
    for epoch in range(config.num_warm_up):
        warm_up_sch.step()
        warm_up_loss, warm_up_ide_acc = train_one_epoch()

        # Logging
        print('Warm Up Epoch %d, Loss: %f, IDE Acc: %f' % (epoch + 1, warm_up_loss, warm_up_ide_acc))
        log.write('Warm Up Epoch %d, Loss: %f, IDE Acc: %f\n' % (epoch + 1, warm_up_loss, warm_up_ide_acc))
        log.flush()

    # Train
    for epoch in range(config.num_epoch):
        # Train
        tr_loss, tr_ide_acc = train_one_epoch()
        step_sch.step()

        # Logging
        print('Epoch %d, Train Loss: %f, IDE Acc: %f' % (epoch + 1, tr_loss, tr_ide_acc))
        log.write('Epoch %d, Train Loss: %f, IDE Acc: %f\n' % (epoch + 1, tr_loss, tr_ide_acc))
        log.flush()

        # Save outputs
        torch.save(model.state_dict(), config.save_path + 'resnext_' + str(int(epoch + 1)) + '.t7')

    # Close
    log.close()


if __name__ == "__main__":
    train()
