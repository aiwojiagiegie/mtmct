import numpy as np
from tracking import matching
from tracking.kalman_filter import KalmanFilter
from tracking.track import TrackState, BaseTrack, Track
from scipy.spatial.distance import cdist


def joint_tracks(t_list_a, t_list_b):
    """
    合并两个track集合
    """
    exists = {}
    res = []

    for t in t_list_a:
        exists[t.track_id] = 1
        res.append(t)

    for t in t_list_b:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)

    return res


def sub_tracks(t_list_a, t_list_b):
    res = {}

    for t in t_list_a:
        res[t.track_id] = t

    for t in t_list_b:
        tid = t.track_id
        if res.get(tid, 0):
            del res[tid]

    return list(res.values())


def remove_duplicate_tracks(tracks_a, tracks_b):
    pdist = matching.iou_distance(tracks_a, tracks_b)
    pairs = np.where(pdist < 0.15)
    dup_a, dup_b = list(), list()

    for p, q in zip(*pairs):
        time_p = tracks_a[p].frame_id - tracks_a[p].start_frame
        time_q = tracks_b[q].frame_id - tracks_b[q].start_frame

        if time_p > time_q:
            dup_b.append(q)
        else:
            dup_a.append(p)

    res_a = [t for i, t in enumerate(tracks_a) if not i in dup_a]
    res_b = [t for i, t in enumerate(tracks_b) if not i in dup_b]

    return res_a, res_b


def calculate_iou(box1, box2):
    # 计算两个边界框的交并比（IOU）
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    # 计算相交面积
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算两个边界框面积
    area_box1 = box1[2] * box1[3]
    area_box2 = box2[2] * box2[3]

    # 计算并集面积
    union_area = area_box1 + area_box2 - intersection_area

    # 避免除以零的情况
    if union_area == 0:
        return 0

    return intersection_area / union_area


def find_adjacent_bbox(all_boxes, target_box):
    # 初始化最大IOU为0，以及对应的边界框
    max_iou = 0.6
    adjacent_index = None

    for index, box in enumerate(all_boxes):
        iou = calculate_iou(box, target_box)
        if iou > max_iou:
            max_iou = iou
            adjacent_index = index

    return adjacent_index


class BoTSORT(object):
    def __init__(self, opt ):
        # Initialize
        self.tracked = []
        self.lost = []
        self.finished = []
        self.kalman_filter = KalmanFilter()
        BaseTrack.clear_count()

        # Set parameters
        self.opt = opt
        self.frame_id = -1
        self.max_time_lost = int(opt.max_time_lost)
        self.temp_align = None

    def update(self, cam, detections, features, de_temp_align):
        """
        更新SCT的跟踪数据
        """
        # 初始化
        activated = []
        re_activated = []
        lost = []
        finished = []
        self.frame_id += 1

        if len(detections.shape) == 1:
            detections = detections[np.newaxis, :]

        # 初始化
        boxes = detections[:, 0:4].astype(np.float32)
        confidences = detections[:, 4].astype(np.float32)
        features = features.astype(np.float32)

        # 删除低于low_thresh的bbox和features
        indices_remain = confidences > self.opt.det_low_thresh
        boxes = boxes[indices_remain]
        confidences = confidences[indices_remain]
        features = features[indices_remain]

        # 找到高于high_thresh的bbox和features
        indices_high = confidences > self.opt.det_high_thresh
        boxes_first = boxes[indices_high]
        confidences_first = confidences[indices_high]
        features_first = features[indices_high]

        # 生成新的Track类
        if len(boxes_first) > 0:
            detections_first = [Track(cam, cxcywh, s, f)
                                for (cxcywh, s, f) in zip(boxes_first, confidences_first, features_first)]
        else:
            detections_first = []

        # 是否是红绿灯等待的车辆判断规则：1. 有相邻帧 2. 有bbox帧位置相近 IOU>0.6 3. reid 相似度距离小于0.3
        de_temp_align_cam_ = de_temp_align[cam]
        maybe_tracks = []
        if self.frame_id in de_temp_align_cam_:
            real_fdx = de_temp_align_cam_[self.frame_id]
            for finished_track in self.finished:
                if abs(real_fdx - de_temp_align_cam_[finished_track.end_frame]) < 30:  # 满足条件1
                    target_boxes_index = find_adjacent_bbox(boxes, finished_track.obs_history[-1][1])
                    if target_boxes_index is not None:
                        reid_dist = \
                            cdist([finished_track.obs_history[-1][3]], [features[target_boxes_index]], 'cosine')[0][0]
                        if reid_dist < 0.2:
                            maybe_tracks.append(finished_track)
                            finished_track.is_activated = True
                            finished_track.state = TrackState.Tracked
                            self.tracked.append(finished_track)
                            self.finished.remove(finished_track)
        # Split into unactivated (tracked only 1 beginning frame) and tracked (tracked more than 2 frames)
        # 将现有的track分成两部分tracked, unactivated
        tracked, unactivated = [], []
        for track in self.tracked:
            if not track.is_activated:
                unactivated.append(track)
            else:
                tracked.append(track)

        # Step 1 - First association, tracks & high confidence detection boxes =========================================
        # Merge tracked (tracked more than 2 frames) and lost (tracked more than 2 frames, lost tracks)
        # 合并tracked集合和lost集合到轨迹池pool中
        pool = joint_tracks(tracked, self.lost)

        # Predict the current location with KF
        # 用卡尔曼滤波去预测当前位置
        Track.multi_predict(pool)

        # Calculate cosine and IoU distances
        # 计算轨迹池pool与detections_first的IOU距离和ReId距离
        cos_dists = matching.embedding_distance(pool, detections_first)
        iou_dists = matching.iou_distance(pool, detections_first)

        # Distance
        dists = cos_dists.copy()
        dists[cos_dists > self.opt.cos_thr] = 1.
        dists[iou_dists > 1 - (1 - self.opt.iou_thr) / 2] = 1.

        # Associate
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.opt.cos_thr)

        # Update state
        for t, d in matches:
            track = pool[t]
            det = detections_first[d]

            if track.state == TrackState.Tracked:
                track.update(detections_first[d], self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                re_activated.append(track)

        # Step 2 - Second association, left tracks & low confidence detection boxes ====================================
        # Find low confidence detections
        indices_high = confidences < self.opt.det_high_thresh
        indices_low = confidences > self.opt.det_low_thresh
        indices_second = np.logical_and(indices_low, indices_high)
        boxes_second = boxes[indices_second]
        confidences_second = confidences[indices_second]
        features_second = features[indices_second]

        # Encode detections with Track
        if len(boxes_second) > 0:
            detections_second = [Track(cam, x1y1x2y2, s, f)
                                 for (x1y1x2y2, s, f) in zip(boxes_second, confidences_second, features_second)]
        else:
            detections_second = []

        # Calculate distances
        remained = [pool[i] for i in u_track if pool[i].state == TrackState.Tracked]
        iou_dists = matching.iou_distance(remained, detections_second)

        # Calculate cosine and IoU distances
        dists = iou_dists.copy()
        dists[iou_dists > self.opt.iou_thr] = 1.

        # Associate
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=self.opt.iou_thr)

        # Update state
        for t, d in matches:
            track = remained[t]
            det = detections_second[d]

            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                track.obs_history[-1][2] = self.opt.det_high_thresh+0.01
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                re_activated.append(track)

        # Find lost tracks
        for t in u_track:
            track = remained[t]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost.append(track)

        # Step 3 - Third association, unactivated tracks & left high confidence detection boxes ========================
        # Get left high confidence detections
        detections_first = [detections_first[i] for i in u_detection]

        # Calculate distances
        cos_dists = matching.embedding_distance(unactivated, detections_first)
        iou_dists = matching.iou_distance(unactivated, detections_first)

        # Distance
        dists = cos_dists.copy()
        dists[cos_dists > self.opt.cos_thr] = 1.
        dists[iou_dists > 1 - (1 - self.opt.iou_thr) / 2] = 1.

        # Associate
        matches, u_unactivated, u_detection = matching.linear_assignment(dists, thresh=self.opt.cos_thr)

        # Update state
        for t, d in matches:
            unactivated[t].update(detections_first[d], self.frame_id)
            activated.append(unactivated[t])

        # Update state
        for it in u_unactivated:
            track = unactivated[it]
            track.mark_removed()

        # Initiate new tracks
        for n in u_detection:
            track = detections_first[n]
            if track.confidence >= self.opt.det_high_thresh:
                # Exclude detection with small box size, Since gt does not include small boxes
                w, h = track.tlwh[2:]
                img_h, img_w = self.opt.img_ori_size
                if h * w <= img_h * img_w * self.opt.min_box_size:
                    continue

                # Initiate new track
                track.initiate(self.kalman_filter, self.frame_id)
                activated.append(track)

        # Update state
        for track in self.lost:
            # Finish track temporal distance
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_finished()
                finished.append(track)

            # Finish lost track with small box size, Since gt does not include small boxes
            w, h = track.tlwh[2:]
            img_h, img_w = self.opt.img_ori_size
            if h * w <= img_h * img_w * self.opt.min_box_size:
                track.mark_finished()
                finished.append(track)

        # Merge 1
        self.tracked = [t for t in self.tracked if t.state == TrackState.Tracked]
        self.tracked = joint_tracks(self.tracked, activated)
        self.tracked = joint_tracks(self.tracked, re_activated)

        # Merge 2
        self.lost.extend(lost)
        self.finished.extend(finished)

        # Merge 3
        self.lost = sub_tracks(self.lost, self.tracked)
        self.lost = sub_tracks(self.lost, self.finished)
        self.tracked, self.lost = remove_duplicate_tracks(self.tracked, self.lost)

        return self.tracked
