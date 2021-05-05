import copy
from utils import img2gps
from utils import kalman_filter as kf


class Track(object):
    def __init__(self, box):
        super(Track, self).__init__()
        self.track = [box]
        self.kalman = kf.KalmanFilter()
        self.mean, self.cov = self.kalman.initiate(self.ltwh2xywh(box[4:8]))

    def ltwh2xywh(self, ltwh):
        xywh = copy.deepcopy(ltwh)
        xywh[0] = xywh[0] + xywh[2] / 2
        xywh[1] = xywh[1] + xywh[3] / 2
        return xywh

    def update(self, box):
        self.track.append(copy.deepcopy(box))
        self.mean, self.cov = self.kalman.predict(self.mean, self.cov)
        self.mean, self.cov = self.kalman.update(self.mean, self.cov, self.ltwh2xywh(box[4:8]))

    def get_next_box(self):
        # Predict
        mean, cov = copy.deepcopy(self.mean), copy.deepcopy(self.cov)
        mean, cov = self.kalman.predict(mean, cov)

        # Form box
        next_box = copy.deepcopy(self.track[-1])
        next_box[4], next_box[5] = mean[0] - mean[2] / 2, mean[1] - mean[3] / 2
        next_box[6], next_box[7] = mean[2], mean[3]
        next_box[10], next_box[11] = next_box[4] + next_box[6] * 0.5, next_box[5] + next_box[7] * 0.8
        gps = img2gps.to_gps_coord('test', 'S06', next_box[1], [next_box[4], next_box[5], next_box[6], next_box[7]])
        next_box[12], next_box[13] = gps[0], gps[1]

        return next_box

    def get_future(self, num):
        # Predict
        mean, cov = copy.deepcopy(self.mean), copy.deepcopy(self.cov)
        for i in range(num):
            mean, cov = self.kalman.predict(mean, cov)

        # Calculate next position and box size
        next_left, next_top = mean[0] - mean[2] / 2, mean[1] - mean[3] / 2
        next_right, next_bot = mean[0] + mean[2] / 2, mean[1] + mean[3] / 2
        next_box_size = mean[2] * mean[3] if 0 < mean[2] and 0 < mean[3] else 0

        return next_left, next_top, next_right, next_bot, next_box_size
