import copy
import pickle
import numpy as np
from utils import utils
import scipy.optimize as opt

# Read mtsc results
result_path = '../outputs/2. mtsc/mask_rcnn_0.2/det_small_mtsc_v2_res8_fut3_del3_post123'
save_path = '../outputs/3. mtmc/mask_rcnn_0.2/det_small_mtsc_v2_res8_fut3_del3_post123_final'
# result_path = '../outputs/2. mtsc/fairmot_affine_hsv_0.3/det_mtsc_v2_res8_fut3_del3_post123'
# save_path = '../outputs/3. mtmc/fairmot_affine_hsv_0.3/det_mtsc_v2_res8_fut3_del3_post123'
with open(result_path + '.pickle', 'rb') as f:
    mtsc_results = pickle.load(f)


# Measure distance between trajectory
def measure_distance(a_track, b_track):
    # Rearrange trajectories
    a_track_per_cam_no_str = {}
    for box in a_track:
        if type(box) is not str:
            if box[1] not in a_track_per_cam_no_str.keys():
                a_track_per_cam_no_str[box[1]] = []
            a_track_per_cam_no_str[box[1]].append(box)
    b_track_no_str = [box for box in b_track if type(box) is not str]

    dist = []
    for cam in a_track_per_cam_no_str.keys():
        # Find best object scores
        a_track_obj_scores = [box[14] for box in a_track_per_cam_no_str[cam]]
        a_track_best_obj_scores = sorted(a_track_obj_scores)[-max(-5, round(len(a_track_obj_scores)*0.2)):]
        b_track_obj_scores = [box[14] for box in b_track_no_str]
        b_track_best_obj_scores = sorted(b_track_obj_scores)[-max(-5, round(len(b_track_obj_scores)*0.2)):]

        for a_track_best_obj_score in a_track_best_obj_scores:
            # Get box and feature
            a_feat = a_track_per_cam_no_str[cam][a_track_obj_scores.index(a_track_best_obj_score)][15]
            for b_track_best_obj_score in b_track_best_obj_scores:
                # Get box and feature, Measure distance
                b_feat = b_track_no_str[b_track_obj_scores.index(b_track_best_obj_score)][15]
                dist.append(np.sqrt(np.sum((a_feat - b_feat) ** 2)))

    return np.min(dist)


# Generate pairwise distance matrix
def gen_dist_mat(a_mtmc, b_mtsc):
    # Create empty matrix
    con_mat = np.zeros((len(a_mtmc), len(b_mtsc)))
    dist_mat = np.ones((len(a_mtmc), len(b_mtsc))) * 1000

    # Overlap camera pairs (There are no overlapped cameras.)
    overlap_cam_pairs = []

    a_1 = []
    for a_track in a_mtmc:
        if a_track[-1] == 'to_next_cam':
            a_1.append(copy.deepcopy(a_track))
    a_1_diff = []
    a_1 = sorted(a_1, key=lambda track: track[-2][2])
    for i in range(len(a_1) - 1):
        a_1_diff = a_1[i+1][-2][2] - a_1[i][-2][2]
    a_1_max_diff = np.max(a_1_diff) * 1.5

    b_1 = []
    for b_track in b_mtsc:
        if b_track[-1] == 'to_previous_cam':
            b_1.append(copy.deepcopy(b_track))
    b_1_diff = []
    b_1 = sorted(b_1, key=lambda track: track[-2][2])
    for i in range(len(b_1) - 1):
        b_1_diff = b_1[i+1][-2][2] - b_1[i][-2][2]
    b_1_max_diff = np.max(b_1_diff) * 1.5

    # Post process the distance matrix with the prior constraints
    for idx, a_track in enumerate(a_mtmc):
        # Get minimum frame number and maximum frame number
        a_f_min = np.min([box[2] for box in a_track if type(box) is not str])
        a_f_max = np.max([box[2] for box in a_track if type(box) is not str])

        for jdx, b_track in enumerate(b_mtsc):
            # Get minimum frame number and maximum frame number
            b_f_min = np.min([box[2] for box in b_track if type(box) is not str])
            b_f_max = np.max([box[2] for box in b_track if type(box) is not str])

            # Disconnect if connection not available
            if a_track[-1] == 'to_next_cam' and b_track[0] == 'from_previous_cam':
                min_f_num_diff = utils.get_min_f_num_diff(a_track, b_track, 1)
                if a_f_max + min_f_num_diff < b_f_min < a_f_max + min_f_num_diff + a_1_max_diff:
                    dist_mat[idx, jdx] = measure_distance(a_track, b_track)
                    con_mat[idx, jdx] = 1
            elif a_track[0] == 'from_next_cam' and b_track[-1] == 'to_previous_cam':
                min_f_num_diff = utils.get_min_f_num_diff(a_track, b_track, -1)
                if b_f_max + min_f_num_diff + b_1_max_diff > a_f_min > b_f_max + min_f_num_diff:
                    dist_mat[idx, jdx] = measure_distance(a_track, b_track)
                    con_mat[idx, jdx] = -1

    # Post process dist mat
    for idx in range(dist_mat.shape[0]):
        for jdx in range(dist_mat.shape[1]):
            dist_mat[idx, jdx] = dist_mat[idx, jdx] if dist_mat[idx, jdx] <= 1.175 else 1000

    return dist_mat, con_mat


def hungarian():
    # Set merge order
    print('Start MTMC Hungarian\n')
    merge_order = ['c041', 'c042', 'c043', 'c044', 'c045', 'c046']

    # Start mtmc
    a_mtmc, result = copy.deepcopy(mtsc_results['S06'][merge_order[0]]), []
    for c_idx in range(1, len(merge_order)):
        # Get current mtsc results
        print('S06_%s starts' % merge_order[c_idx])
        b_mtsc = copy.deepcopy(mtsc_results['S06'][merge_order[c_idx]])

        # Generate distance matrix between trajectories
        print('Distance matrix pair: %d x %d' % (len(a_mtmc), len(b_mtsc)))
        dist_mat, con_mat = gen_dist_mat(a_mtmc, b_mtsc)
        print('Num connections: %d / %d\n' % (np.sum(con_mat != 0), len(a_mtmc) * len(b_mtsc)))

        # Hungarian algorithm
        row_ind, col_ind = opt.linear_sum_assignment(dist_mat)
        row_ind, col_ind = list(row_ind), list(col_ind)

        # Check distance between connections
        con_row_ind, con_col_ind = [], []
        for r_idx in range(len(row_ind)):
            if dist_mat[row_ind[r_idx], col_ind[r_idx]] < 1000:
                # Merge trajectories 'a' and 'b'
                if con_mat[row_ind[r_idx], col_ind[r_idx]] == 1:
                    a_mtmc[row_ind[r_idx]] = copy.deepcopy(a_mtmc[row_ind[r_idx]]) \
                                             + copy.deepcopy(b_mtsc[col_ind[r_idx]])
                elif con_mat[row_ind[r_idx], col_ind[r_idx]] == -1:
                    a_mtmc[row_ind[r_idx]] = copy.deepcopy(b_mtsc[col_ind[r_idx]])\
                                             + copy.deepcopy(a_mtmc[row_ind[r_idx]])

                # Record
                con_row_ind.append(row_ind[r_idx])
                con_col_ind.append(col_ind[r_idx])

        # Finish trajectories
        fin_idx = [r for r in range(len(a_mtmc)) if r not in con_row_ind]
        for idx, f_idx in enumerate(fin_idx):
            result.append(copy.deepcopy(a_mtmc.pop(f_idx - idx)))

        # Starting trajectories
        for c in range(len(b_mtsc)):
            if c not in con_col_ind:
                a_mtmc.append(copy.deepcopy(b_mtsc[c]))

    # Final merge
    result += copy.deepcopy(a_mtmc)

    # # Post process (Do not post process Recall become too low)
    # result_post = []
    # for track in result:
    #     cams = list(set([box[1] for box in track if type(box) is not str]))
    #     if 2 <= len(cams):
    #         result_post.append(track)

    return result


def map_obj_id(result):
    result_new_id = copy.deepcopy(result)
    for t_idx, track in enumerate(result):
        for b_idx, box in enumerate(track):
            if type(box) is not str:
                result_new_id[t_idx][b_idx][3] = t_idx
    print('Num ID: %d' % len(result_new_id))

    return result_new_id


def write_txt(result):
    # Open txt file, Write txt file, Close
    num_box = 0
    mtmc_txt = open(save_path + '.txt', 'w')
    for track in result:
        for box in track:
            if type(box) is not str:
                if 0.1 <= box[14]:
                    # Decode
                    left, top, w, h, img_w, img_h = box[4], box[5], box[6], box[7], box[8], box[9]

                    # Expand
                    new_w, new_h = w * 1.2, h * 1.2
                    # new_w, new_h = w, h

                    # Calculate new left and top
                    c_x, c_y = left + w / 2, top + h / 2
                    new_left, new_top = c_x - new_w / 2, c_y - new_h / 2
                    new_right, new_bot = new_left + new_w, new_top + new_h

                    # Threshold by image size
                    new_left, new_top = max(0, new_left), max(0, new_top)
                    new_right, new_bot = min(img_w, new_right), min(img_h, new_bot)
                    new_w, new_h = new_right - new_left, new_bot - new_top

                    # Write
                    mtmc_txt.write('%d %d %d %d %d %d %d %d %d\n'
                                   % (int(box[1][1:]), box[3], box[2], new_left, new_top, new_w, new_h, 0, 0))
                    num_box += 1
    mtmc_txt.close()
    print('Num Box: %d' % num_box)


if __name__ == "__main__":
    result = hungarian()
    result = map_obj_id(result)
    write_txt(result)
