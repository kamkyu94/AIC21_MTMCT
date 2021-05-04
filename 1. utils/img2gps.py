import numpy as np
from numpy.linalg import inv


def to_gps_coord(mode, scene, cam, bbox):
    # Read calibration information
    cal_file_path = '../../dataset/AIC21_Track3/%s/%s/%s/calibration.txt' % (mode, scene, cam)
    cal_file = open(cal_file_path, 'r')
    cal_lines = cal_file.readlines()

    # Create Homography matrix
    H = cal_lines[0].split(':')[-1]
    H = H.replace(';', ' ').split()
    H = [float(i) for i in H]
    H = np.reshape(H, (3, 3))
    H = inv(H)

    # Transform
    left, top, w, h = bbox
    center_pt = np.asarray([bbox[0] + 0.5 * w, bbox[1] + 0.5 * h, 1])
    center_pt = np.reshape(center_pt, (3, 1))
    gps_pt = np.matmul(H, center_pt)
    gps_pt = gps_pt / gps_pt[-1]

    # Close
    cal_file.close()

    return gps_pt[:2].squeeze()
