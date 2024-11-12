import copy
import numpy as np
import utils.utils as utils
from tracking.kalman_filter import KalmanFilter
from utils.laneUtils import lane_mask_reader


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Finished = 3
    Removed = 3


class BaseTrack(object):
    _count = 0
    track_id = 0
    frame_id = -1
    is_activated = False
    state = TrackState.New

    confidence = 0
    start_frame = 0
    time_since_update = 0

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_finished(self):
        self.state = TrackState.Finished

    def mark_removed(self):
        self.state = TrackState.Removed

    @staticmethod
    def clear_count():
        BaseTrack._count = 0


class Track(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, cam, cxcywh, confidence, feat=None):
        # Initialize basics
        self.cam = cam
        self.global_id = None
        self.cxcywh = cxcywh
        self.confidence = confidence
        self.future_len = 5
        self.img_size = (1080, 1920)

        # Initialize about feature
        self.alpha = 0.9
        self.curr_feat = feat
        self.smooth_feat = None
        self.update_features(feat)

        # Initialize others
        self.obs_history = []
        self.is_activated = False
        self.kalman_filter = None
        self.mean, self.covariance = None, None

    def update_features(self, feat):
        if feat is not None:
            # Normalize
            feat /= np.linalg.norm(feat)

            # Update
            if self.smooth_feat is None:
                self.smooth_feat = feat
            else:
                self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat

            # Normalize
            self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()

        # Give zero to vw and vh
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(tracks):
        if len(tracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in tracks])
            multi_covariance = np.asarray([st.covariance for st in tracks])

            # Give zero to vw and vh
            for i, st in enumerate(tracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0

            multi_mean, multi_covariance = Track.shared_kalman.multi_predict(multi_mean, multi_covariance)

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                tracks[i].mean = mean
                tracks[i].covariance = cov

    def initiate(self, kalman_filter, frame_id):
        # Start a new track
        self.track_id = self.next_id()
        self.kalman_filter = kalman_filter

        # Update, Save
        self.mean, self.covariance = self.kalman_filter.initiate(self.cxcywh)
        self.obs_history = [[frame_id, self.cxcywh.copy(), self.confidence, self.curr_feat.copy(),
                             self.mean.copy(), self.covariance.copy(), []]]

        # Set
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.state = TrackState.Tracked
        self.is_activated = True if frame_id == 0 else self.is_activated

    def update(self, new_det, frame_id, lane_reader=lane_mask_reader):
        self.frame_id = frame_id

        # Update
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,
                                                               new_det.cxcywh, new_det.confidence)
        
        # 获取当前车道信息
        current_lanes = []
        if lane_reader is not None:
            bbox = [
                int(self.mean[0] - self.mean[2]/2),  # x1
                int(self.mean[1] - self.mean[3]/2),  # y1
                int(self.mean[0] + self.mean[2]/2),  # x2
                int(self.mean[1] + self.mean[3]/2)   # y2
            ]
            current_lanes = lane_reader.get_lanes_for_bbox(bbox, self.cam)
        
        # 在历史记录中加入车道信息
        self.obs_history.append([
            frame_id,
            new_det.cxcywh.copy(),
            new_det.confidence,
            copy.deepcopy(new_det.curr_feat),
            self.mean.copy(),
            self.covariance.copy(),
            current_lanes  # 新增字段
        ])

        if new_det.curr_feat is not None:
            self.update_features(new_det.curr_feat)

        # Set
        self.is_activated = True
        self.confidence = new_det.confidence
        self.state = TrackState.Tracked

    def re_activate(self, new_det, frame_id, new_id=False, lane_reader=lane_mask_reader):
        # Update
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,
                                                               new_det.cxcywh, new_det.confidence)
        
        # 获取当前车道信息
        current_lanes = []
        if lane_reader is not None:
            bbox = [
                int(self.mean[0] - self.mean[2]/2),  # x1
                int(self.mean[1] - self.mean[3]/2),  # y1
                int(self.mean[0] + self.mean[2]/2),  # x2
                int(self.mean[1] + self.mean[3]/2)   # y2
            ]
            current_lanes = lane_reader.get_lanes_for_bbox(bbox, self.cam)
        
        # 在历史记录中加入车道信息
        self.obs_history.append([
            frame_id,
            new_det.cxcywh.copy(),
            new_det.confidence,
            new_det.curr_feat.copy(),
            self.mean.copy(),
            self.covariance.copy(),
            current_lanes  # 新增字段
        ])

        if new_det.curr_feat is not None:
            self.update_features(new_det.curr_feat)

        # Set
        self.is_activated = True
        self.frame_id = frame_id
        self.confidence = new_det.confidence
        self.state = TrackState.Tracked
        self.track_id = self.next_id() if new_id else self.track_id

    def get_feature(self, mode='ema'):
        # Default use smoothed feature
        feat = self.smooth_feat

        # Other options
        if mode == 'best':
            feat = self.obs_history[np.argmax(np.array([o[2] for o in self.obs_history]))][3]
        elif mode == 'last':
            feat = self.obs_history[-1][3]
        elif mode == 'avg':
            feat = np.mean(np.array([o[3] for o in self.obs_history]), axis=0)
        elif mode == 'weighted_avg':
            feat = np.sum(np.array([o[3] * o[2] for o in self.obs_history]), axis=0)
            feat /= np.sum([o[2] for o in self.obs_history])
        elif mode == 'all':
            feat = np.array([o[3] for o in self.obs_history])
            feat = feat[np.newaxis] if len(feat.shape) == 1 else feat

        return feat

    @property
    def tlwh(self):
        """
            Get current position in bounding box format `(top left x, top left y, width, height)`.
        """
        if self.mean is None:
            x = self.cxcywh[0] - self.cxcywh[2] / 2
            y = self.cxcywh[1] - self.cxcywh[3] / 2
            w = self.cxcywh[2]
            h = self.cxcywh[3]
        else:
            x = self.mean[0] - self.mean[2] / 2
            y = self.mean[1] - self.mean[3] / 2
            w = self.mean[2]
            h = self.mean[3]

        return np.array([x, y, w, h])

    @property
    def x1y1x2y2(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})_{}_{}_{}'.format(self.track_id, self.start_frame, self.end_frame, self.state,self.global_id,self.cam)

    def get_main_lanes(self, recent_frames=None, start_threshold=0.5, min_threshold=0.1, step=0.1):
        """
        获取轨迹经过的主要车道，通过逐步降低阈值直到找到结果
        :param recent_frames: 只考虑最近的帧数，如果为None则考虑所有历史记录
        :param start_threshold: 开始的阈值，默认0.5
        :param min_threshold: 最低接受的阈值，默认0.1
        :param step: 每次降低的阈值步长，默认0.1
        :return: 主要车道集合
        """
        # 获取要分析的历史记录
        if recent_frames is not None:
            history = self.obs_history[-recent_frames:]
        else:
            history = self.obs_history
            
        # 统计每个车道出现的次数
        lane_counts = {}
        total_frames = len(history)
        
        for record in history:
            lanes = record[6]  # 第7个元素是车道信息
            for lane in lanes:
                lane_counts[lane] = lane_counts.get(lane, 0) + 1
        
        # 从高阈值开始逐步降低直到找到结果
        for threshold in np.arange(start_threshold, min_threshold - step/2, -step):
            main_lanes = {
                lane for lane, count in lane_counts.items() 
                if count / total_frames >= threshold
            }
            if main_lanes:  # 如果找到非空结果，立即返回
                return main_lanes
                
        return set()  # 如果所有阈值都没有结果，返回空集合

    @property
    def main_lanes(self):
        return self.get_main_lanes()

