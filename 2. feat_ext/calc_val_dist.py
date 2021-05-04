import pickle
import config
import random
import numpy as np

random.seed(config.seed)
np.random.seed(config.seed)

# Read extracted features
result_path = './outputs/resnext_8/'
with open(result_path + 'val.pickle', 'rb') as f:
    val_feat = pickle.load(f)

# val_feat
# dict[gt_id] = [[scene, cam, f_num, gt_id, feature], [scene, cam, f_num, gt id, feature], ...]


# Sample queue
def sample_a_b(queue):
    # Split queue and get information about first bbox 'a'
    a_info, aff, cam_rel = queue
    a_cam, a_f_num, a_gt_id = a_info[1], a_info[2], a_info[3]

    # Different id, different camera
    if aff == 0 and cam_rel == 0:
        # Select info
        b_gt_id_candidates = list(val_feat.keys())
        b_gt_id_candidates.remove(a_gt_id)
        b_gt_id = random.choice(b_gt_id_candidates)
        b_info = random.choice(val_feat[b_gt_id])

        while b_info[1] == a_cam:
            b_gt_id_candidates = list(val_feat.keys())
            b_gt_id_candidates.remove(a_gt_id)
            b_gt_id = random.choice(b_gt_id_candidates)
            b_info = random.choice(val_feat[b_gt_id])

    # Different id, same camera
    elif aff == 0 and cam_rel == 1:
        infos = []
        for gt_id in val_feat.keys():
            for info in val_feat[gt_id]:
                if info[2] != a_gt_id and info[1] == a_cam and a_f_num <= info[2]:
                    infos.append(info)

        # Select info
        if len(infos) == 0:
            _, b_info, _, _ = sample_a_b([a_info, 1, 1])
        else:
            b_info = random.choice(infos)

    # Same id, different camera
    elif aff == 1 and cam_rel == 0:
        infos = []
        for info in val_feat[a_gt_id]:
            if info[1] != a_cam:
                infos.append(info)

        # Select info
        if len(infos) == 0:
            _, b_info, _, _ = sample_a_b([a_info, 0, 0])
        else:
            b_info = random.choice(infos)

    # Same id, same camera
    else:
        infos = []
        for info in val_feat[a_gt_id]:
            if info[1] == a_cam and 0 < (info[2] - a_f_num) <= 10:
                infos.append(info)

        # Select info
        if len(infos) == 0:
            b_info = a_info
        else:
            b_info = random.choice(infos)

    return a_info, b_info, aff, cam_rel


def sample_queue():
    # Generate queue
    queues = []
    for gt_id in val_feat.keys():
        for info in val_feat[gt_id]:
            queues += [[info, 0, 0], [info, 1, 0], [info, 0, 1], [info, 1, 1]]

    # Shuffle and sample
    queues_sampled = []
    [np.random.shuffle(queues) for _ in range(5)]
    for idx, q in enumerate(queues[:100000]):
        queues_sampled.append(sample_a_b(q))
        print(idx + 1, ' / 100000') if (idx + 1) % 500 == 0 else None

    return queues_sampled


def measure_dist(a_feat, b_feat, metric):
    dist = 10000
    if metric == 'euclidean':
        dist = np.sqrt(np.sum(((a_feat - b_feat) ** 2)))
    elif metric == 'cosine':
        dist = np.maximum(0.0, 1 - np.sum(a_feat * b_feat))

    return dist


def measure_accuracy(thr_step=0.001, thr_max=5):
    # # Sampled queue
    # queues = sample_queue()
    #
    # # Save sampled queue
    # with open(result_path + 'queues.pickle', 'wb') as handle:
    #     pickle.dump(queues, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Read sampled queue
    with open(result_path + 'queues.pickle', 'rb') as file:
        queues = pickle.load(file)

    # Generate dists
    dists = {0: {0: [], 1: []}, 1: {0: [], 1: []}}
    for q in queues:
        a_info, b_info, aff, cam_rel = q
        a_feat, b_feat = a_info[-1], b_info[-1]
        dists[aff][cam_rel].append(measure_dist(a_feat, b_feat, 'euclidean'))

    # # Get length
    # same_cam_num = len(dists[0][1]) + len(dists[1][1])
    # different_cam_num = len(dists[0][0]) + len(dists[1][0])
    # total_num = len(dists[0][0]) + len(dists[0][1]) + len(dists[1][0]) + len(dists[1][1])

    # # Measure accuracy
    # thr, mtsc_acc, mtmc_acc, total_acc = 0, [], [], []
    # while thr <= thr_max:
    #     # Increase threshold
    #     thr += thr_step
    #
    #     # Threshold
    #     result00 = np.sum(thr < np.array(dists[0][0]))
    #     result01 = np.sum(thr < np.array(dists[0][1]))
    #     result10 = np.sum(np.array(dists[1][0]) <= thr)
    #     result11 = np.sum(np.array(dists[1][1]) <= thr)
    #
    #     # Measure accuracy
    #     mtsc_acc.append((result01 + result11) / same_cam_num)
    #     mtmc_acc.append((result00 + result10) / different_cam_num)
    #     total_acc.append((result00 + result01 + result10 + result11) / total_num)

    # # Find threshold
    # mtsc_max_acc = np.max(mtsc_acc)
    # mtsc_thr = (mtsc_acc.index(mtsc_max_acc) + 1) * thr_step
    # mtmc_max_acc = np.max(mtmc_acc)
    # mtmc_thr = (mtmc_acc.index(mtmc_max_acc) + 1) * thr_step
    # total_max_acc = np.max(total_acc)
    # total_thr = (total_acc.index(total_max_acc) + 1) * thr_step
    #
    # # Logging
    # print('MTSC Acc :%f, MTSC threshold :%f' % (mtsc_max_acc, mtsc_thr))
    # print('MTMC Acc :%f, MTMC threshold :%f' % (mtmc_max_acc, mtmc_thr))
    # print('Total Acc :%f, Total threshold :%f\n' % (total_max_acc, total_thr))

    print('Same id, Same cam:', np.sum(np.array(dists[1][1]) <= 0.5) / len(dists[1][1]))
    print('Diff id, Same cam:', np.sum(0.5 < np.array(dists[0][1])) / len(dists[0][1]))

    # print('Same id, Diff cam:', np.sum(np.array(dists[1][0]) <= mtmc_thr) / len(dists[1][0]))
    # print('Diff id, Diff cam:', np.sum(mtmc_thr < np.array(dists[0][0])) / len(dists[0][0]))


measure_accuracy()
