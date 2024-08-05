import lap
import numpy as np
from scipy.spatial.distance import cdist
from cython_bbox import bbox_overlaps as bbox_ious


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])

    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)

    return matches, unmatched_a, unmatched_b


def ious(a_x1y1x2y2s, b_x1y1x2y2s):
    ious = np.zeros((len(a_x1y1x2y2s), len(b_x1y1x2y2s)), dtype=np.float64)
    if ious.size == 0:
        return ious

    ious = bbox_ious(np.ascontiguousarray(a_x1y1x2y2s, dtype=np.float64),
                     np.ascontiguousarray(b_x1y1x2y2s, dtype=np.float64))

    return ious


def iou_distance(a_tracks, b_tracks):
    a_x1y1x2y2s = [track.x1y1x2y2 for track in a_tracks]
    b_x1y1x2y2s = [track.x1y1x2y2 for track in b_tracks]

    _ious = ious(a_x1y1x2y2s, b_x1y1x2y2s)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix

    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float64)
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float64)

    # Normalized features
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))

    return cost_matrix
if __name__ == '__main__':
    # onker-Volgenant 算法，也称为拉普拉斯修正匈牙利算法（LAPJV），
    # 是一种用于解决线性分配问题的高效算法。
    # 线性分配问题通常描述为：
    # 有等量的工作和工人，每个工人完成每项工作的成本不同，如何分配工作使得总成本最低。
    # 这个问题可以通过一个成本矩阵来表示，其中矩阵的每个元素表示对应工人完成对应工作的成本。
    cost_matrix=np.asarray([[9,2,7,8],[6,4,3,7],[5,8,1,8],[7,6,9,4]])
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=8)
    print(1)
