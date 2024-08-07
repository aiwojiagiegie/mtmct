import numpy as np
import pandas as pd
from tqdm import tqdm


def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1[3], box1[4], box1[5], box1[6]
    x2, y2, w2, h2 = box2[3], box2[4], box2[5], box2[6]

    # 计算两个矩形的交集坐标
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    # 计算交集的宽高
    wi = max(0, xi2 - xi1)
    hi = max(0, yi2 - yi1)

    # 计算交集和并集面积
    intersection = wi * hi
    union = w1 * h1 + w2 * h2 - intersection

    return intersection / union if union != 0 else 0


def getData(fpath, sep='\s+|\t+|,'):
    names = ['CameraId', 'Id', 'FrameId', 'X', 'Y', 'Width', 'Height', 'Xworld', 'Yworld']
    try:
        df = pd.read_csv(
            fpath,
            sep=sep,
            index_col=None,
            skipinitialspace=True,
            header=None,
            names=names,
            engine='python'
        )
        return df
    except Exception as e:
        raise ValueError("Could not read input from %s. Error: %s" % (fpath, repr(e)))


def calculate_iou_matrix(pred_boxes, gt_boxes):
    num_pred = len(pred_boxes)
    num_gt = len(gt_boxes)
    ious = np.zeros((num_pred, num_gt), dtype=np.float32)

    for i in tqdm(range(num_pred), desc='计算IOU中'):
        pred_x1 = pred_boxes[i, 3]
        pred_y1 = pred_boxes[i, 4]
        pred_w = pred_boxes[i, 5]
        pred_h = pred_boxes[i, 6]
        pred_x2 = pred_x1 + pred_w
        pred_y2 = pred_y1 + pred_h

        gt_x1 = gt_boxes[:, 3]
        gt_y1 = gt_boxes[:, 4]
        gt_w = gt_boxes[:, 5]
        gt_h = gt_boxes[:, 6]
        gt_x2 = gt_x1 + gt_w
        gt_y2 = gt_y1 + gt_h

        # 计算交集的坐标
        inter_x1 = np.maximum(pred_x1, gt_x1)
        inter_y1 = np.maximum(pred_y1, gt_y1)
        inter_x2 = np.minimum(pred_x2, gt_x2)
        inter_y2 = np.minimum(pred_y2, gt_y2)

        # 计算交集的宽度和高度
        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)

        # 计算交集面积
        intersection = inter_w * inter_h

        # 计算并集面积
        area_pred = pred_w * pred_h
        area_gt = gt_w * gt_h
        union = area_pred + area_gt - intersection

        # 计算IOU
        ious[i, :] = np.where(union > 0, intersection / union, 0)

    return ious


class Track():
    def __init__(self):
        self.gt_car_id = None  # gt文件中的车辆idx
        self.pred_car_id = None  # 预测结果中的车辆id
    def is_same_car(self, other_track):
        if self.gt_car_id is not None and self.pred_car_id is not None:
            return False
        if other_track.gt_car_id is not None and other_track.pred_car_id is not None:
            return False
        


def process_tracks(data_df, id_column_name='Id', is_pred=True):
    # 添加行索引列
    data_df['RowIndex'] = data_df.index.tolist()

    # 获取唯一的ID集合
    unique_ids = set(data_df[id_column_name])

    tracks = []
    for id in unique_ids:
        # 使用 .loc 根据 Id 列的值来选择行
        matching_rows = data_df.loc[data_df[id_column_name] == id]

        # 获取行索引
        row_indices = matching_rows.index.tolist()

        # 创建Track对象
        track = Track()

        if is_pred:
            track.pred_car_id = id
        else:
            track.gt_car_id = id

        # 将行索引添加为新的一列
        matching_rows['RowIndex'] = row_indices

        # 将DataFrame转换为NumPy数组
        track.matching_rows = matching_rows.to_numpy()

        tracks.append(track)

    return tracks


if __name__ == '__main__':
    detection_path = 'debug_sort/results.txt'
    pred_data_df = getData(detection_path)
    pred_data = pred_data_df.values

    gt_path = 'debug_sort/gt.txt'
    gt_data_df = getData(gt_path)
    gt_data = gt_data_df.values
    global ious
    ious = calculate_iou_matrix(pred_data, gt_data)

    pred_track_list = process_tracks(pred_data_df)

    gt_track_list = process_tracks(gt_data_df)


    for i, pred_track in enumerate(pred_track_list):
        for j, gt_track in enumerate(gt_track_list):
            pred_track.is_same_car(gt_track)
