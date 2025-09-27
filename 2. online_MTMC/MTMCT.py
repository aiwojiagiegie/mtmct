import argparse
import os
import random

import dill as pickle

# Import numpy compatibility patch BEFORE any package that uses motmetrics
from monkeypatch_numpy import *

import cv2
import time
import copy
import torch
import numpy as np
from tqdm import tqdm

from output_HST.将视频取帧放入训练yolo的数据集中 import output_path
from output_HST.计算单文件的指标 import calculate_results
from opts import opt
from torchvision import transforms
from utils.sklearn_dunn import dunn
from tracking.bot_sort import BoTSORT
from utils.datasets import LoadImages
from models.experimental import attempt_load
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.cluster.hierarchy import linkage, fcluster
from models.feature_extractor import FeatureExtractor
from utils.scipy_linear_assignment import linear_assignment
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.utils import letterbox, class_agnostic_nms, pairwise_tracks_dist
from ultralytics import YOLO
from ReId import ReId
import torchvision.ops

class Cluster:
    def __init__(self):
        super(Cluster, self).__init__()
        self.tracks = []
        self.feat = np.zeros((0, 2048))

    def add_track(self, track):
        if track not in self.tracks:
            self.tracks.append(track)

    def get_feature(self):
        if opt.get_feat_mode != 'all':
            feat = np.array([track.get_feature(opt.get_feat_mode) for track in self.tracks])
            feat = feat[np.newaxis, :] if len(feat.shape) == 1 else feat
        else:
            feat = np.concatenate([track.get_feature(opt.get_feat_mode) for track in self.tracks], axis=0)
        return feat

    @property
    def end_frame(self):
        return np.max([track.end_frame for track in self.tracks])

    @property
    def cam_list(self):
        return [track.cam for track in self.tracks]

def prepare_align(cams, f_nums):
    temp_align = {}
    for cam in cams:
        temp_align[cam] = {}
        for i in range(0, np.max(f_nums) + 1):
            # Default
            # temp_align[cam][i] = 0
            # temp_align[cam][i] = i + 1
            # continue
            # Set for each camera
            temp_align[cam][i] = i
    return temp_align
# def prepare_align(cams, f_nums):
#     temp_align = {}
#     reference_frame = 207  # 摄像头 41 在 15:30:00 时的帧数
#     # 定义每个摄像头在 15:30:00 时的帧数
#     sync_frames = {
#         '41': 109,
#         '42': 79,
#         '43': 77,
#         '44': 207,
#         '45': 119,
#         '46': 118
#     }
#     # 定义每个摄像头视频的帧数量
#     all_frames = {
#         '41': 2341,
#         '42': 2378,
#         '43': 2530,
#         '44': 2703,
#         '45': 2694,
#         '46': 2863
#     }
#     i=0
#     for cam in cams:
#         offset = sync_frames[cam] - reference_frame
#         f_nums[i] += abs(offset)
#         i+=1
#     for cam in cams:
#         temp_align[cam] = {}
#         offset = sync_frames[cam] - reference_frame
#         for i in range(0, np.max(f_nums) + 1):
#             # 对齐帧，使所有摄像头在 15:30:00 时的帧号都为 109
#             aligned_frame = i + offset

#             # 确保对齐后的帧号不小于 0
#             temp_align[cam][i] = aligned_frame

#     return temp_align


class MTMCT(object):
    def __init__(self, opt,YOLOv10_detect_model_path=None):
        self.result_path = None
        self.opt = opt
        self.start = time.time()

        # Load models ====================================================================================================
        # Load detection model
        # self.det_model = attempt_load(opt.det_weights + opt.det_name + '.pt')
        # self.det_model = self.det_model.cuda().eval().half()
        if YOLOv10_detect_model_path is None:
            self.YOLOv10_detect_model = YOLO(opt.yolo10_model, verbose=False)
        else:
            self.YOLOv10_detect_model = YOLO(YOLOv10_detect_model_path, verbose=False)
        # For time measurement
        self.total_times = {'Det': 0, 'Ext': 0, 'MTSC': 0, 'MTMC': 0}
        self.cams = sorted(os.listdir(opt.data_dir))
        self.stride = 32
        # self.stride = int(self.det_model.stride.max())
        self.img_size = opt.img_size.copy()
        # self.img_size[0] = check_img_size(opt.img_size[0], s=self.stride)
        # self.img_size[1] = check_img_size(opt.img_size[1], s=self.stride)
        # Load feature extraction model
        if opt.baseline_reid:
            print("现在使用baseline的reid")
            feat_ext_model = FeatureExtractor(opt.feat_ext_name, opt.avg_type, opt.feat_ext_weights)
            self.feat_ext_model = feat_ext_model.cuda().eval().half()
        else:
            print("现在使用新的的reid")
            self.ReId = ReId(opt.reid_path)
        # For feature extraction model
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # 添加 lane_img 成员变量
        self.lane_images = self.load_lane_images()
        # Prepare ========================================================================================================
        # Prepare output folder
        self.output_dir = opt.output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.result_path = self.output_dir + f'{mtmct_version}.txt'
        self.debug_mtmct_pkl = self.output_dir + f'{mtmct_version}/mtmct.pkl'
        # Prepare others
        self.datasets, self.trackers, self.f_nums = {}, {}, []
        self.roi_masks, self.overlap_regions_cam2cam = {}, {}
        self.img_h, self.img_w = opt.img_ori_size
        for cam in self.cams:
            # Prepare 1
            img_dir = os.path.join(opt.data_dir, cam) + '/frame/*'
            self.datasets[cam] = iter(LoadImages(img_dir, img_size=self.img_size, stride=self.stride))
            self.trackers[cam] = BoTSORT(opt)
            self.f_nums.append(self.datasets[cam].nf)

            # Prepare 2
            self.roi_masks[cam] = cv2.imread(f'../dataset/HST/real/{cam}/{cam}.png', cv2.IMREAD_GRAYSCALE)
            self.overlap_regions_cam2cam[cam] = {}
            for cam_ in self.cams:
                if cam_ == cam:
                    self.overlap_regions_cam2cam[cam][cam_] = None
                    continue
                overlap_file = f'./preliminary/overlap_zones/HST/{cam}_{cam_}.png'
                if os.path.exists(overlap_file):
                    self.overlap_regions_cam2cam[cam][cam_] = cv2.imread(overlap_file, cv2.IMREAD_GRAYSCALE)
                else:
                    # 创建一个与原图像大小相同的全黑图片
                    self.overlap_regions_cam2cam[cam][cam_] = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        self.temp_align = prepare_align(self.cams, self.f_nums)
        for tracker in self.trackers.values():
            tracker.temp_align = self.temp_align
        if opt.baseline_reid:
            with torch.autocast('cuda'):
                for _ in range(10):
                    self.feat_ext_model(
                        torch.rand((10, 3, opt.patch_size[0], opt.patch_size[1]), device='cuda').half())

        # Temporal alignment 时间对齐的序列
        self.next_global_id, self.dunn_index_prev = 0, -1e5
        self.clusters_dict = {}
        self.result = []
        self.result_set = set()
        # 在线特征记忆（EMA）相关
        self.id_memory = {}  # global_id -> { 'feat': np.ndarray, 'count': int }
        # 直接使用 self.opt 中的参数，避免在此处重复缓存配置

        # 加载并存储背景图像
        # self.background_images = self.load_background_images()
        # self.background_images = np.transpose(self.background_images, (0, 3, 1, 2))
        # self.background_images = torch.from_numpy(self.background_images).cuda().half()

        # 初始化视频捕获对象和当前帧计数器
        self.video_captures = {}
        self.current_frame = {}
        self.total_frames = {}  # 新增：存储每个摄像头的总帧数
        for cam in self.cams:
            video_path = f'../dataset/HST/real/{cam}/{cam}.mp4'
            self.video_captures[cam] = cv2.VideoCapture(video_path)
            self.current_frame[cam] = 0
            self.total_frames[cam] = int(self.video_captures[cam].get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数

    def load_background_images(self):
        background_images = {}
        base_path = '../dataset/HST/real/'
        for cam in self.cams:
            img_ori_path = os.path.join(base_path, cam, f'{cam}.png')
            if os.path.exists(img_ori_path):
                background_images[cam] = cv2.imread(img_ori_path)
            else:
                raise ValueError(f'图像{img_ori_path}不存在')
        return background_images

    def run_mtmct(self):
        # 初始化视频写入器字典
        video_writers = {}

        # Run
        for fdx in tqdm(range(0, np.max(self.f_nums))):
            # 准备图像数据
            batch_img, valid_cam = self.generate_image_info(fdx)

            # 目标检测
            preds = self.detect(batch_img, valid_cam)

            # REID
            feat, detection = self.reid(batch_img, preds)

            # 单摄像头跟踪
            online_tracks_raw = self.MTSCT_online(feat, detection)

            # 根据detection把bbox绘制到图片中
            self.draw_debug_video(batch_img, detection, valid_cam, video_writers)

            # 跨摄像头跟踪
            if 'n' in opt.version:
                self.new_mtmct_online(fdx, online_tracks_raw)
            else:
                self.mtmct_online(fdx, online_tracks_raw)

        # 释放视频写入器
        for writer in video_writers.values():
            writer.release()

        directory = os.path.dirname(self.result_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(self.result_path, 'w') as result_txt:
            for r in self.result:
                print(r, file=result_txt)
        print(f'结果已经写入到{self.result_path}')
        # Logging
        track_t, total_t = 0, 0
        print('%s_%s_%s' % (opt.det_name, opt.feat_ext_name, opt.avg_type))
        for key in self.total_times.keys():
            print('%s: %05f' % (key, self.total_times[key] / (np.max(self.f_nums) + 1)))
            track_t += self.total_times[key] / (np.max(self.f_nums) + 1) if key == 'MTSC' or key == 'MTMC' else 0
            total_t += self.total_times[key] / (np.max(self.f_nums) + 1)
        print('Tracking Time: %05f' % track_t)
        print('Total Time: %05f' % total_t)

    def draw_debug_video(self, batch_img, detection, valid_cam, video_writers):
        if opt.draw_debug:
            base_path = '../dataset/HST/real'
            output_path = './output_HST/debug生成视频文件 新训练的模型'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            for idx, det in enumerate(self.cams):
                if valid_cam[det]:
                    # 初始化视频写入器
                    if det not in video_writers:
                        video_path = os.path.join(output_path, f'{det}.mp4')
                        if not os.path.exists(os.path.dirname(video_path)):
                            os.makedirs(os.path.dirname(video_path))
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writers[det] = cv2.VideoWriter(video_path, fourcc, 30, (self.img_w, self.img_h))

                    if valid_cam[det]:
                        img = cv2.resize(batch_img[idx], (self.img_w, self.img_h))
                        for box in detection[det]:
                            # 左上角xy和长宽 置信度
                            center_x, center_y, width, height, conf = box
                            left = int(center_x - width / 2)
                            top = int(center_y - height / 2)
                            width = int(width)
                            height = int(height)
                            # 绘制边界框
                            cv2.rectangle(img, (left, top), (left + width, top + height), (0, 255, 0), 2)
                            # 添加标签
                            cv2.putText(img, f'{conf:.2f}', (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                        2)

                        video_writers[det].write(img)

    def generate_image_info(self, fdx):
        """
        从视频中准备图像数据
        """
        batch_img = []
        valid_cam = {}

        for cam in self.cams:
            cap = self.video_captures[cam]

            # 如果当前帧不是我们需要的帧，进行寻帧操作
            if fdx != self.current_frame[cam]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fdx)
                self.current_frame[cam] = fdx

            # 读取帧
            ret, frame = cap.read()

            if ret:
                img = cv2.resize(frame, (self.img_size[1], self.img_size[0]))
                batch_img.append(img)
                valid_cam[cam] = True
                self.current_frame[cam] += 1
            else:
                # 如果读取失败，使用空白图像并标记为无效
                batch_img.append(np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8))
                valid_cam[cam] = False

                if fdx >= self.total_frames[cam]:
                    pass
                    # print(f"警告：尝试读取的帧{fdx}超过了摄像头{cam}的总帧数{self.total_frames[cam]}")
                else:
                    print(f"从摄像头{cam}读取帧{fdx}失败")

        return batch_img, valid_cam

    def _normalize_feat(self, feat):
        if feat is None:
            return None
        feat = np.asarray(feat)
        if feat.ndim > 1:
            feat = feat.reshape(-1)
        norm = np.linalg.norm(feat) + 1e-12
        return feat / norm

    def _compute_track_quality(self, track):
        # 基于检测置信与框面积的简单质量估计，范围约 [min_quality, 1]
        try:
            conf = float(track.obs_history[-1][2])
        except Exception:
            conf = 0.5
        try:
            tlwh = track.obs_history[-1][1]
            w = float(tlwh[2])
            h = float(tlwh[3])
        except Exception:
            w, h = 0.0, 0.0
        area_ratio = (w * h) / (self.img_w * self.img_h + 1e-12)
        # 相对最小框面积做归一，避免超小框影响
        base = max(getattr(self.opt, 'min_box_size', 0.0005) * 2.0, 1e-6)
        area_norm = np.clip(area_ratio / base, 0.0, 1.0)
        quality = 0.6 * np.clip(conf, 0.0, 1.0) + 0.4 * area_norm
        return float(np.clip(quality, getattr(self.opt, 'min_quality', 0.2), 1.0))

    def _get_memory_feature(self, track):
        """
        基于 opt.memory_mode 返回外观特征：
        - current: 仅使用当前帧特征
        - memory: 优先使用记忆（若无则回退当前）
        - hybrid: 记忆与当前的加权融合（简单平均）
        """
        cur = track.get_feature(mode=self.opt.get_feat_mode)
        cur = self._normalize_feat(cur)
        gid = getattr(track, 'global_id', None)
        mem = None
        if gid is not None and gid in self.id_memory:
            mem = self.id_memory[gid]['feat']

        mode = self.opt.memory_mode
        if mode == 'current' or mem is None:
            return cur
        if mode == 'memory':
            return mem
        # hybrid
        return self._normalize_feat(0.5 * mem + 0.5 * cur)

    def _update_id_memory(self, gid, new_feat, quality):
        if gid is None or new_feat is None:
            return
        new_feat = self._normalize_feat(new_feat)
        if new_feat is None:
            return
        if gid not in self.id_memory:
            self.id_memory[gid] = {'feat': new_feat.copy(), 'count': 1}
            return
        old = self.id_memory[gid]['feat']
        # 质量调制的 EMA 更新：beta 越大越依赖新特征
        if self.opt.use_quality_ema:
            ema_mom = float(self.opt.ema_momentum)
            beta_base = (1.0 - ema_mom)
            beta = np.clip(beta_base * float(quality), 0.0, 1.0)
        else:
            # 固定步长 EMA（不使用质量调制）
            ema_mom = float(self.opt.ema_momentum)
            beta = np.clip(1.0 - ema_mom, 0.0, 1.0)
        updated = (1.0 - beta) * old + beta * new_feat
        self.id_memory[gid]['feat'] = self._normalize_feat(updated)
        self.id_memory[gid]['count'] += 1

    def new_mtmct_online(self, fdx, online_tracks_raw):
        """
        使用“分组匈牙利一对一匹配”替换原贪心匹配：
        - 逐相邻摄像头对、逐车道分组构建代价矩阵（ReID 余弦距离）
        - 对每组使用匈牙利算法进行一对一匹配
        - 阈值门控后赋 global_id，未匹配则新建 ID

        Args:
            fdx: 当前帧 id
            online_tracks_raw: {cam: List[track]}
        """
        from scipy.spatial.distance import cdist

        # 若未启用分组匈牙利路径，回退到旧的 mtmct_online（用于消融）
        if not self.opt.use_grouped_hungarian:
            return self.mtmct_online(fdx, online_tracks_raw)

        camera_order = ['41', '42', '43', '44', '45', '46']

        # 收集历史轨迹池（用于上一相邻摄像头的候选）
        target_tracks = {cam: tracker.all_tracks for cam, tracker in self.trackers.items()}

        # 过滤当前帧在线轨迹
        online_tracks_filtered = self.filter_online_tracks(online_tracks_raw)

        # 将所有在线轨迹合并，用于最终写结果
        online_tracks_all = []
        for cam in self.cams:
            online_tracks_all += online_tracks_filtered[cam]

        # 先给首个摄像头的在线轨迹分配全局 ID（若尚未分配）
        if camera_order:
            first_cam = camera_order[0]
            for tr in online_tracks_filtered.get(first_cam, []):
                if tr.global_id is None:
                    tr.global_id = self.next_global_id
                    self.next_global_id += 1
                # 初始化后立即用当前观测更新记忆
                q = self._compute_track_quality(tr)
                f = tr.get_feature(mode=opt.get_feat_mode)
                self._update_id_memory(tr.global_id, f, q)

        # 生成需要处理的摄像头对
        if self.opt.topology_adjacent_only:
            cam_pairs = [(camera_order[i - 1], camera_order[i]) for i in range(1, len(camera_order))]
        else:
            cam_pairs = []
            for i in range(len(camera_order)):
                for j in range(len(camera_order)):
                    if i == j:
                        continue
                    cam_pairs.append((camera_order[i], camera_order[j]))

        # 逐对摄像头进行（可选）分组匈牙利匹配
        for (pre_cam, cur_cam) in cam_pairs:

            # 候选集合：pre 是最近窗口内出现的历史轨迹；cur 是当前帧在线轨迹
            pre_candidates = [t for t in target_tracks.get(pre_cam, []) if (
                t.cam == pre_cam and len(t.obs_history) > 0 and (fdx - t.obs_history[-1][0] <= opt.max_time_lost)
            )]
            cur_candidates = [t for t in online_tracks_filtered.get(cur_cam, []) if t.cam == cur_cam]

            if len(pre_candidates) == 0 or len(cur_candidates) == 0:
                continue

            # 车道分组（可开关）
            if self.opt.lane_grouping:
                lanes = set([t.current_lane for t in pre_candidates] + [t.current_lane for t in cur_candidates])
            else:
                lanes = {'__ALL__'}

            # 当前摄像头已使用的 global_id（避免重复）
            cur_used_gids = set([t.global_id for t in cur_candidates if t.global_id is not None])

            for lane in lanes:
                if self.opt.lane_grouping:
                    pre_lane = [t for t in pre_candidates if t.current_lane == lane]
                    cur_lane = [t for t in cur_candidates if t.current_lane == lane and t.global_id is None]
                else:
                    pre_lane = list(pre_candidates)
                    cur_lane = [t for t in cur_candidates if t.global_id is None]

                if len(pre_lane) == 0 or len(cur_lane) == 0:
                    continue

                # 构建 ReID 余弦距离代价矩阵
                pre_feats = np.array([self._get_memory_feature(t) for t in pre_lane])
                cur_feats = np.array([self._get_memory_feature(t) for t in cur_lane])

                # 保护：若任一侧为空（理论上前面判断已排除）
                if pre_feats.size == 0 or cur_feats.size == 0:
                    continue

                dist_mat = cdist(pre_feats, cur_feats, metric='cosine')
                dist_mat = np.clip(dist_mat, 0.0, 1.0)

                # 若未分组，则按车道不一致施加惩罚或硬掩膜
                if not self.opt.lane_grouping:
                    pre_lanes = [t.current_lane for t in pre_lane]
                    cur_lanes = [t.current_lane for t in cur_lane]
                    mismatch = np.not_equal.outer(pre_lanes, cur_lanes)
                    if self.opt.lane_mask_strict:
                        dist_mat = dist_mat.copy()
                        dist_mat[mismatch] = 1e6
                    else:
                        penalty = float(self.opt.lane_penalty)
                        dist_mat = dist_mat + (mismatch.astype(np.float32) * penalty)

                # 阈值门控：超过阈值的对置为大数，避免被匹配
                thr = float(self.opt.gating_app_thr)
                gated_cost = dist_mat.copy()
                gated_cost[gated_cost > thr] = 1e6

                # 运行匈牙利匹配
                indices = linear_assignment(gated_cost)

                # 赋 ID（仅接受未超阈值的匹配）
                for row, col in indices:
                    if row < 0 or row >= len(pre_lane) or col < 0 or col >= len(cur_lane):
                        continue
                    cost_val = gated_cost[row, col]
                    if cost_val > thr:
                        continue

                    pre_t = pre_lane[row]
                    cur_t = cur_lane[col]

                    # 目标：将 pre 的 global_id 传播给 cur；若 pre 没有，则新建一个同时赋给二者
                    gid = pre_t.global_id
                    if gid is None:
                        # 如果 cur 已有 id（极少见，因为上面限定了 cur.global_id is None），优先复用 cur 的；否则新建
                        gid = cur_t.global_id if cur_t.global_id is not None else self.next_global_id
                        if gid == self.next_global_id:
                            self.next_global_id += 1
                        pre_t.global_id = gid

                    # 避免同摄像头重复使用同一个 gid
                    if gid in cur_used_gids and (cur_t.global_id is None or cur_t.global_id != gid):
                        continue

                    cur_t.global_id = gid
                    cur_used_gids.add(gid)

                    # 用当前观测对记忆做质量加权更新（pre 与 cur 均更新）
                    q_pre = self._compute_track_quality(pre_t)
                    f_pre = pre_t.get_feature(mode=opt.get_feat_mode)
                    self._update_id_memory(gid, f_pre, q_pre)

                    q_cur = self._compute_track_quality(cur_t)
                    f_cur = cur_t.get_feature(mode=opt.get_feat_mode)
                    self._update_id_memory(gid, f_cur, q_cur)

        # 对仍未分配 ID 的在线轨迹分配新 ID
        for cam in self.cams:
            for tr in online_tracks_filtered.get(cam, []):
                if tr.global_id is None:
                    tr.global_id = self.next_global_id
                    self.next_global_id += 1
                # 确保每个在线轨迹的记忆被及时刷新
                q = self._compute_track_quality(tr)
                f = tr.get_feature(mode=opt.get_feat_mode)
                self._update_id_memory(tr.global_id, f, q)

        # 记录结果
        self.record_result(fdx, online_tracks_all)
        return

    def record_result(self, fdx, online_tracks_raw):
        # 记录结果
        for track in online_tracks_raw:
            left, top, w, h = track.tlwh

            # 扩展边界框,因为gt边界框并不是紧密地围绕对象,而是比对象大得多
            cx, cy = left + w / 2, top + h / 2
            # w, h = w * 1.45, h * 1.45
            left, top = cx - w / 2, cy - h / 2

            # 根据大小过滤,因为gt不包括小边界框
            # if w * h / self.img_w / self.img_h < 0.003 or 0.3 < w * h / self.img_w / self.img_h:
            #     continue
            format = '%d %d %d %d %d %d %d -1 -1' % (
                int(track.cam[-2:]), track.global_id, self.temp_align[track.cam][fdx], int(left), int(top),
                int(w),
                int(h))
            if format not in self.result_set:
                self.result.append(format)
                self.result_set.add(format)

    def mtmct_online(self, fdx, online_tracks_raw):
        start = time.time()
        # 过滤原始跟踪结果,去除不符合条件的跟踪
        online_tracks_filtered = self.filter_online_tracks(online_tracks_raw)

        # 合并所有摄像头的跟踪结果
        online_tracks = []
        for cam in self.cams:
            online_tracks += online_tracks_filtered[cam]

        # 收集当前已分配的全局ID
        online_global_ids = {'41': [], '42': [], '43': [], '44': [], '45': [], '46': []}
        for track in online_tracks:
            if track.global_id is not None:
                online_global_ids[track.cam].append(track.global_id)

        # 获取特征数据并计算特征距离
        online_feats = np.array([track.get_feature(mode=opt.get_feat_mode) for track in online_tracks])
        if len(online_feats) < 2:
            return
        p_dists = pdist(online_feats, metric='cosine')
        p_dists = np.clip(p_dists, 0, 1)  # 归一化距离到[0,1]区间

        # 应用约束条件,过滤不符合条件的跟踪对
        self.filter_tracks_by_overlap(online_tracks, p_dists)

        # 聚类 =================================================================================================
        # 使用层次聚类生成链接矩阵
        linkage_matrix = linkage(p_dists, method='complete')
        ranked_dists = np.sort(list(set(list(linkage_matrix[:, 2]))), axis=None)

        # 通过调整距离阈值观察聚类结果,并计算Dunn指数
        clusters, dunn_indices, c_dists = [], [], squareform(p_dists)
        for rdx in range(2, ranked_dists.shape[0] + 1):
            if ranked_dists[-rdx] <= opt.mtmc_match_thr:
                clusters.append(fcluster(linkage_matrix, ranked_dists[-rdx] + 1e-5, criterion='distance') - 1)
                dunn_indices.append(dunn(clusters[-1], c_dists))

        if len(clusters) == 0:
            cluster = fcluster(linkage_matrix, ranked_dists[0] - 1e-5, criterion='distance') - 1
        else:
            # 选择最连通的聚类结果,排除不适当的配对
            # 获取Dunn指数突然跳跃的索引
            dunn_indices.insert(0, 0)
            pos = np.argmax(np.diff(dunn_indices))
            cluster = clusters[pos]

        # 运行多目标多摄像头跟踪 =====================================================================
        # 初始化
        num_cluster = len(list(set(list(cluster))))

        # 使用同一聚类中的其他跟踪为新跟踪分配全局ID
        for cam_index in range(num_cluster):
            track_idx = np.where(cluster == cam_index)[0]

            # 检查同一聚类中跟踪的索引和全局ID
            infos = []
            for tdx in track_idx:
                if online_tracks[tdx].global_id is not None:
                    infos.append([tdx, online_tracks[tdx].global_id])

            # 如果聚类中的某些跟踪已经有全局ID,则为新踪分配相同的全局ID
            if len(infos) > 0:
                # 分配全局ID,收集跟踪,更新特征
                for tdx in track_idx:
                    if online_tracks[tdx].global_id is None:
                        # 排序并获取具有最小距离的节点的全局ID
                        sorted_infos = sorted(copy.deepcopy(infos), key=lambda x: c_dists[tdx, x[0]])
                        for info in sorted_infos:
                            if online_tracks[info[0]].global_id not in online_global_ids[online_tracks[tdx].cam]:
                                # 分配全局ID,收集
                                online_tracks[tdx].global_id = info[1]
                                if info[1] in self.clusters_dict:
                                    self.clusters_dict[info[1]].add_track(online_tracks[tdx])
                                break

        # 获取剩余的当前跟踪
        remain_tracks = [track for track in online_tracks if track.global_id is None]

        # 计算先前聚类和当前聚类之间的成对距离
        dists = pairwise_tracks_dist(self.clusters_dict, remain_tracks, fdx, metric='cosine' , opt=opt)

        # 运行匈牙利算法
        indices = linear_assignment(dists)

        # 使用阈值进行匹配
        for row, col in indices:
            if dists[row, col] <= opt.mtmc_match_thr \
                    and list(self.clusters_dict.keys())[row] not in online_global_ids[remain_tracks[col].cam]:
                # 分配全局ID,收集跟踪
                remain_tracks[col].global_id = list(self.clusters_dict.keys())[row]
                self.clusters_dict[list(self.clusters_dict.keys())[row]].add_track(remain_tracks[col])

        # 如果未匹配,则新建跟踪
        for remain_track in remain_tracks:
            if remain_track.global_id is None:
                # 分配全局ID,收集跟踪
                remain_track.global_id = self.next_global_id
                self.clusters_dict[self.next_global_id] = Cluster()
                self.clusters_dict[self.next_global_id].add_track(remain_track)

                # 增加全局ID计数
                self.next_global_id += 1

        # 删除过旧的聚类
        del_key = [key for key in self.clusters_dict.keys() if
                   fdx - self.clusters_dict[key].end_frame > opt.max_time_differ]
        for key in del_key:
            del self.clusters_dict[key]

        self.total_times['MTMC'] += time.time() - start

        self.record_result(fdx, online_tracks)
    def filter_tracks_by_overlap(self, online_tracks, p_dists):
        # 定义摄像头拓扑结构
        camera_topology = {
            '41': ['42'],
            '42': ['43'],
            '43': ['44'],
            '44': ['45'],
            '45': ['46'],
            '46': []
        }
        for i in range(len(online_tracks)):
            for j in range(i + 1, len(online_tracks)):
                # 转换索引
                idx = len(online_tracks) * i + j - ((i + 2) * (i + 1)) // 2

                # 如果是同一个摄像头，过滤
                if online_tracks[i].cam == online_tracks[j].cam:
                    p_dists[idx] = 10
                    continue

                # 如果两个轨迹不在同一个车道上，过滤
                if online_tracks[i].current_lane != online_tracks[j].current_lane:
                    p_dists[idx] = 10
                    continue

                # 检查摄像头是否相邻 i 在前 j在后
                cam_i = online_tracks[i].cam
                cam_j = online_tracks[j].cam

                # 如果两个轨迹都有global_id或都没有global_id，跳过
                # if (online_tracks[i].global_id is None and online_tracks[j].global_id is None) or \
                #    (online_tracks[i].global_id is not None and online_tracks[j].global_id is not None):
                #     p_dists[idx] = 10
                #     continue

                # if online_tracks[i].global_id is not None and online_tracks[j].global_id is not None:
                #     p_dists[idx] = 10
                #     continue

                # # # 确保cam_i是有global_id的轨迹所在的摄像头
                # if online_tracks[i].global_id is None:
                #     cam_i, cam_j = cam_j, cam_i
                #
                # # 如果两个摄像头不相邻，过滤
                # topology_cams = camera_topology[cam_i]
                # if cam_j not in topology_cams:
                #     p_dists[idx] = 10
                #     continue

                # If the objects are not in overlapping region (i -> j)
                # 根据摄像头与摄像头之间的overlap区域过滤
                overlap_region = self.overlap_regions_cam2cam[online_tracks[i].cam][online_tracks[j].cam]
                x1, y1, x2, y2 = online_tracks[i].x1y1x2y2.astype(np.int32)
                y2 = y2 if y2 < 1080 else 1079
                o1 = overlap_region[y2, (x1 + x2) // 2]
                if o1 == 0:
                    p_dists[idx] = 10
                    continue

                # If the objects are not in overlapping region (j -> i)
                overlap_region = self.overlap_regions_cam2cam[online_tracks[j].cam][online_tracks[i].cam]
                x1, y1, x2, y2 = online_tracks[j].x1y1x2y2.astype(np.int32)
                y2 = y2 if y2 < 1080 else 1079
                o2 = overlap_region[y2, (x1 + x2) // 2]
                if o2 == 0:
                    p_dists[idx] = 10
                    continue

    def filter_online_tracks(self, online_tracks_raw):
        online_tracks_filtered = {}
        for cam in self.cams:
            online_tracks_filtered[cam] = []
            if cam not in online_tracks_raw:
                continue
            for track in online_tracks_raw[cam]:
                # If not activated
                if not track.is_activated:
                    continue

                # If it has low confidence score
                if track.obs_history[-1][2] <= opt.det_high_thresh:
                    continue

                # Filter detection with small box size, Since gt does not include small boxes
                w, h = int(track.obs_history[-1][1][2]),int(track.obs_history[-1][1][3])
                min_box_size = self.img_h * self.img_w * opt.min_box_size
                if h * w <= min_box_size:
                    continue

                # Filter detections around border, Since gt does not include boxes around border
                x1, y1, x2, y2 = track.x1y1x2y2
                if x1 <= 4 or y1 <= 4 or x2 >= self.img_w - 5 or y2 >= self.img_h - 5:
                    continue
                online_tracks_filtered[cam].append(track)

            # Class agnostic NMS, Since gt does not include overlapped boxes
            if 2 <= len(online_tracks_filtered[cam]):
                online_tracks_filtered[cam] = class_agnostic_nms(online_tracks_filtered[cam])
        return online_tracks_filtered

    def MTSCT_online(self, feat, detection):
        """
        返回出四个跟踪器在更新新的帧之后，仍在跟踪的轨迹集合
        """
        start = time.time()
        online_tracks_raw = {}
        for cam in self.cams:
            online_tracks_raw[cam] = self.trackers[cam].update(cam, detection[cam], feat[cam], self.temp_align,self.lane_images[cam])
        self.total_times['MTSC'] += time.time() - start
        return online_tracks_raw

    def reid(self, batch_img, preds):
        self.start = time.time()
        # 创建新的转置数组，而不修改原始数组
        transposed_batch_img = np.transpose(batch_img, (0, 3, 1, 2))
        # transposed_batch_img_ori = np.transpose(batch_img_ori, (0, 3, 1, 2))

        # 然后转换 CUDA 张量
        tensor_batch_img = torch.from_numpy(transposed_batch_img).cuda().half()
        # tensor_batch_img_ori = torch.from_numpy(self.background_images).cuda().half()
        batch_patch = torch.zeros((100 * len(self.cams), 3, opt.patch_size[0], opt.patch_size[1]), device='cuda').half()
        det_count, detection = 0, {}
        for pdx, pred in enumerate(preds):
            # Prepare dictionary to store detection results
            detection[self.cams[pdx]] = np.zeros((0, 5))

            # If there are valid predictions
            if len(pred) > 0:
                # Rescale boxes from img_size to im0s size
                debug_pred = pred[:, :4]
                pred[:, :4] = scale_coords(tensor_batch_img.shape[2:], pred[:, :4], [1080,1920])
                debug_pred = pred[:, :4]
                # Post-process detections xyxy应该分别是bbox的左上角和右下角的横纵坐标？
                for *xyxy, conf, _ in reversed(pred):
                    # Convert to integerz
                    x1, y1 = round(xyxy[0].item()), round(xyxy[1].item())
                    x2, y2 = round(xyxy[2].item()), round(xyxy[3].item())

                    # 根据ROI过滤掉不在ROI内的检测
                    target = self.roi_masks[self.cams[pdx]][
                        min(y2 + 1, self.img_h) - 1, (max(x1, 0) + min(x2 + 1, self.img_w)) // 2]
                    if target == 0:
                        continue

                    # 过滤掉小数据
                    # if (x2 - x1) * (y2 - y1) <= self.img_h * self.img_w * opt.min_box_size / 2:
                    #     continue

                    # Add detections
                    new_box = np.array([(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1), conf.item()])
                    new_box = new_box[np.newaxis, :]
                    detection[self.cams[pdx]] = np.concatenate([detection[self.cams[pdx]], new_box], axis=0)

                    # Get patch
                    patch = tensor_batch_img[pdx][:, max(y1, 0):min(y2 + 1, self.img_h),
                            max(x1, 0):min(x2 + 1, self.img_w)]
                    patch = self.normalize(letterbox(patch))
                    # 如果是c008则水平翻转
                    batch_patch[det_count] = torch.fliplr(patch) if self.cams[pdx] == 'c008' else patch
                    det_count += 1
        if det_count == 0:
            feat = {}
            for cam in self.cams:
                feat[cam] = np.zeros((0, 2048))
            return feat , detection
        # 提取特征
        if not opt.baseline_reid:
            batch_patch = batch_patch[:det_count]
            batch_feat = self.ReId.reid(batch_patch)
            batch_feat = batch_feat.squeeze().cpu().numpy()
        else:
            with torch.autocast('cuda'):
                batch_patch = batch_patch[:det_count]
                batch_feat = self.feat_ext_model(batch_patch)
            # 将batch_feat转换为NumPy数组
            batch_feat = batch_feat.cpu().numpy()

        # 当batch_feat为一维的时候，给它改成二维
        if batch_feat.ndim == 1:
            batch_feat = batch_feat[np.newaxis, :]
        self.total_times['Ext'] += time.time() - self.start

        # 根据cam id 拆分出需要的特性值
        feat_count, feat = 0, {}
        for cam in self.cams:
            feat[cam] = batch_feat[feat_count:feat_count + len(detection[cam])]
            feat_count += len(detection[cam])

        return feat, detection

    def detect(self, batch_img, valid_cam):
        self.start = time.time()
        # 目标检测阶段
        # Detect =====================================================================================================
        # with torch.autocast('cuda'):
        #     preds = self.det_model(batch_img[list(valid_cam.values())], augment=opt.augment)[0]
        # # NMS之后是最终的检结果
        # preds = non_max_suppression(preds, opt.conf_thres, opt.iou_thres,
        #                             classes=opt.classes, agnostic=opt.agnostic_nms)
        # for cam_index, cam in enumerate(self.cams):
        #     if not valid_cam[cam]:
        #         preds.insert(cam_index, torch.zeros((0, 6)).cuda().half())
        # preds_result = []
        # for single_img in batch_img:
        #     # 确保single_img是一个4D张量 [1, C, H, W]
        #     if single_img.ndim == 3:
        #         single_img = single_img.unsqueeze(0)
        #
        #     # 对单个图像进行预测
        #     single_pred = self.YOLOv10_detect_model(single_img)
        #
        #     # 将单个预测结果添加到列表中
        #     preds_result.append(single_pred)
        preds_result = self.YOLOv10_detect_model(batch_img, verbose=False)

        # 如果需要,可以将preds_result组合成一个批量结果
        # 具体的组合方式取决于YOLOv10_detect_model的输出格式
        # 这里假设输出是一个列表,每个元素对应一张图像的预测结果
        # preds_result = [item for sublist in preds_result for item in sublist]

        ans = []
        for result in preds_result:
            boxes = result.boxes.data.tolist()
            temp_add = []
            for obj in boxes:
                if int(obj[5]) in [2, 5, 7]:
                    temp_add.append(obj)
                    
            if len(temp_add) == 0:
                tensor_temp_add = torch.zeros((0, 6), dtype=torch.float16).cuda().half()
            else:
                # 转换为tensor
                tensor_temp_add = torch.tensor(temp_add, dtype=torch.float16).cuda().half()
                
                # 应用NMS
                # 获取boxes、scores和classes
                boxes = tensor_temp_add[:, :4]  # xyxy格式
                scores = tensor_temp_add[:, 4]  # confidence scores
                classes = tensor_temp_add[:, 5]  # class ids
                
                # 应用NMS,iou_threshold可以根据需要调整
                keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.45)
                tensor_temp_add = tensor_temp_add[keep]
                
            ans.append(tensor_temp_add)
            
        self.total_times['Det'] += time.time() - self.start
        return ans

    def caculate_result(self):
        ans = []
        for cam_name in self.trackers:
            for tracker in self.trackers[cam_name].finished:
                # tracker = boTSORT[cam_name]
                if tracker.global_id is None:
                    continue
                for item in tracker.obs_history:
                    left, top, w, h = item[1]

                    # Expand box, Since gt boxes are not tightly annotated around objects and quite larger than objects
                    cx, cy = left + w / 2, top + h / 2
                    w, h = w * 1.45, h * 1.45
                    left, top = cx - w / 2, cy - h / 2

                    # Filter with size, Since gt does not include small boxes
                    if w * h / self.img_w / self.img_h < 0.003 or 0.3 < w * h / self.img_w / self.img_h:
                        continue
                    format_string = f"{int(tracker.cam[-1])} {tracker.global_id} {self.temp_align[tracker.cam][tracker.frame_id]} " \
                                    f"{int(left)} {int(top)} {int(w)} {int(h)} -1 -1"
                    ans.append(
                        '%d %d %d %d %d %d %d -1 -1' % (
                            int(tracker.cam[-1]), tracker.global_id, item[0],
                            int(left), int(top), int(w), int(h)))
        return ans

    def debug_finished(self):
        with open(finished_txt, 'w') as output_file:
            for online_track in self.trackers.values():
                for track in online_track.all_tracks:
                    for info in track.obs_history:
                        left, top, w, h = caculate_tlwh(info[1])

                        # Expand box, Since gt boxes are not tightly annotated around objects and quite larger than objects
                        cx, cy = left + w / 2, top + h / 2
                        # w, h = w * 1.45, h * 1.45
                        left, top = cx - w / 2, cy - h / 2

                        if left < 0 or top < 0:
                            continue

                        # Filter with size, Since gt does not include small boxes
                        if w * h / self.img_w / self.img_h < 0.003 or 0.3 < w * h / self.img_w / self.img_h:
                            continue
                        if track.global_id is None:
                            continue
                        print(
                            '%d %d %d %d %d %d %d -1 -1' % (
                                int(track.cam[-2:]), track.global_id, self.temp_align[track.cam][info[0]],
                                int(left), int(top), int(w), int(h)), file=output_file)

    def cleanup_for_pickle(self):
        # 删除 ReId 属性
        if hasattr(self, 'ReId'):
            del self.ReId
        if hasattr(self, 'YOLOv10_detect_model'):
            del self.YOLOv10_detect_model
        # 在程序结束时释放视频捕获对象
        for cap in self.video_captures.values():
            cap.release()
        if hasattr(self, 'video_captures'):
            del self.video_captures

    def restore_after_pickle(self):
        # 如果有保存的路径，重新加载 ReId 模型
        if hasattr(self, 'reid_model_path') and self.reid_model_path:
            self.ReId = ReId(self.reid_model_path)

    def load_lane_images(self):
        lane_images = {}
        for cam in self.cams:
            image_path = f'./preliminary/devied_zones/{cam}.png'
            lane_images[cam] = cv2.imread(image_path)
        return lane_images

    def find_best_match(self, track, candidate_tracks, cur_tracks, match_threshold):
        if not candidate_tracks:
            return None, float('inf')

        track_feat = track.get_feature(mode=opt.get_feat_mode)
        candidate_feats = np.array([t.get_feature(mode=opt.get_feat_mode) for t in candidate_tracks])

        # 计算距离
        dists = cdist([track_feat], candidate_feats, metric='cosine')[0]
        dists = np.clip(dists, 0, 1)  # 归一化距离[0,1]区间

        min_dist = float('inf')
        best_match = None

        for i, dist in enumerate(dists):
            if dist <= match_threshold and dist < min_dist:
                # 检查candidate_tracks[i]的global_id是否已经在cur_tracks中出现
                if candidate_tracks[i].global_id not in [t.global_id for t in cur_tracks if t.global_id is not None]:
                    min_dist = dist
                    best_match = candidate_tracks[i]

        return best_match, min_dist


def caculate_tlwh(cxcywh):
    x = cxcywh[0] - cxcywh[2] / 2
    y = cxcywh[1] - cxcywh[3] / 2
    w = cxcywh[2]
    h = cxcywh[3]
    return np.array([x, y, w, h])


def run():
    mtmct = MTMCT(opt)
    with torch.no_grad():
        mtmct.run_mtmct()
    if not os.path.exists(mtmct.debug_mtmct_pkl):
        os.makedirs(os.path.dirname(mtmct.debug_mtmct_pkl), exist_ok=True)

    # 在保存之前清理 ReID 模型
    mtmct.cleanup_for_pickle()

    with open(mtmct.debug_mtmct_pkl, 'wb') as f:
        pickle.dump(mtmct, f)
    return mtmct


def read_pkl_from_file(outputs_mtmct_pkl):
    with open(outputs_mtmct_pkl, 'rb') as file:
        mtmct = pickle.load(file)
    return mtmct


finished_txt = 'debug.txt'
result_dir = 'results'


def debug():
    from tracking import matching
    from scipy.spatial.distance import cdist

    mtmct = read_pkl_from_file(opt.output_dir+outputs_mtmct_pkl)
    mtmct.debug_finished()
    # track14 = mtmct.trackers['c008'].finished[60]
    # reid14 = track14.obs_history[-1][3]
    # track94 = mtmct.trackers['c008'].finished[65]
    # reid94 = track94.obs_history[0][3]
    # reid_dis = cdist([reid14], [reid94], 'cosine')
    # reid_dis = reid_dis[0][0]
    # print(reid_dis)
    # calculate_results('outputs/ground_truth_validation.txt', finished_txt)
    pass

def main():
    mtmct = run()
    # mtmct = read_pkl_from_file(outputs_mtmct_pkl)
    calculate_results('./output_HST/test_gt.txt', mtmct.result_path)
    return mtmct

# 模型配置文件
model_yaml_path = "./yolov10/ultralytics/cfg/models/v10/yolov10n.yaml"
# 数据集配置文件
data_yaml_path = './yolov10/datasets/multi_class/data.yaml'
# 预训练模型

if __name__ == '__main__':
    if opt.train:
        pretrain_type = opt.pretrain_type
        pre_model_name = f'./yolov10/models/yolov10{pretrain_type}.pt'
        # 加载预训练模型
        # model = YOLOv10(model_yaml_path).load(pre_model_name)
        model = YOLO(model_yaml_path)
        pre_model_name_last = pre_model_name.split('/')[-1]
        # 训练模型
        epochs = opt.epoch
        batch = opt.batch
        # 生成一个六的随机数
        random_suffix = random.randint(100000, 999999)
        save_path = f'UA-DETRAC_pre/model_name_{pre_model_name_last}/epochs_{epochs}/batch_{batch}/{random_suffix}'
        results = model.train(data=data_yaml_path,
                              epochs=epochs,
                              batch=batch,
                              name=f"{save_path}", device=opt.gpu)
        outputs_mtmct_pkl = f'./yolov10/runs/detect/{save_path}/mtmct.pkl'
        mtmct_version = f'version/{save_path}/v1'
        mtmct = MTMCT(opt,f'./yolov10/runs/detect/{save_path}/weights/best.pt')
        with torch.no_grad():
            mtmct.run_mtmct()
        with open(outputs_mtmct_pkl, 'wb') as f:
            pickle.dump(mtmct, f)
        calculate_results('outputs/ground_truth_validation.txt', mtmct.result_path)
    else:
        # opt.draw_debug = True
        # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        mtmct_version = f'version/v{opt.version}'
        outputs_mtmct_pkl = f'{mtmct_version}/mtmct.pkl'
        main()
        # debug()







