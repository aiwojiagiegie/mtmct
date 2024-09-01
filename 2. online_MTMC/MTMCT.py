import os
import dill as pickle

import cv2
import time
import copy
import torch
import numpy as np
from tqdm import tqdm
from outputs.eval import calculate_results
from opts import opt
from torchvision import transforms
from utils.sklearn_dunn import dunn
from tracking.bot_sort import BoTSORT
from utils.datasets import LoadImages
from models.experimental import attempt_load
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from models.feature_extractor import FeatureExtractor
from utils.scipy_linear_assignment import linear_assignment
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.utils import letterbox, class_agnostic_nms, pairwise_tracks_dist
from yolov10.ultralytics import YOLOv10


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
            temp_align[cam][i] = 0
            # temp_align[cam][i] = i + 1
            # continue
            # Set for each camera
            if cam == 'c006':
                temp_align[cam][i] = i

            elif cam == 'c007':
                if i <= 1037:
                    temp_align[cam][i] = i + 1
                elif 1040 <= i <= 1309:
                    temp_align[cam][i] = i - 1
                elif 1320 <= i <= 1339:
                    temp_align[cam][i] = i - 11
                elif 1350 <= i <= 1379:
                    temp_align[cam][i] = i - 21
                elif 1400 <= i <= 1449:
                    temp_align[cam][i] = i - 41
                elif 1460 <= i <= 1499:
                    temp_align[cam][i] = i - 51
                elif 1510 <= i <= 1537:
                    temp_align[cam][i] = i - 61
                elif 1540 <= i <= 1542:
                    temp_align[cam][i] = i - 63
                elif 1560 <= i <= 1609:
                    temp_align[cam][i] = i - 80
                elif 1620 <= i <= 1639:
                    temp_align[cam][i] = i - 90
                elif 1650 <= i <= 1864:
                    temp_align[cam][i] = i - 100
                elif 1870 <= i <= 1893:
                    temp_align[cam][i] = i - 105
                elif 1901 <= i <= 1920:
                    temp_align[cam][i] = i - 112
                elif 1927 <= i <= 1933:
                    temp_align[cam][i] = i - 118
                elif 1940 <= i <= 1989:
                    temp_align[cam][i] = i - 124
                elif 2000 <= i <= 2049:
                    temp_align[cam][i] = i - 134
                elif 2060 <= i:
                    temp_align[cam][i] = i - 144

            elif cam == 'c008':
                if 7 <= i <= 421:
                    temp_align[cam][i] = i - 6
                elif 439 <= i <= 472:
                    temp_align[cam][i] = i - 23
                elif 479 <= i <= 548:
                    temp_align[cam][i] = i - 29
                elif 603 <= i <= 685:
                    temp_align[cam][i] = i - 83
                elif 728 <= i <= 925:
                    temp_align[cam][i] = i - 125
                elif 934 <= i <= 1397:
                    temp_align[cam][i] = i - 133
                elif 1401 <= i <= 1612:
                    temp_align[cam][i] = i - 136
                elif 1621 <= i <= 1752:
                    temp_align[cam][i] = i - 144
                elif 1763 <= i <= 1920:
                    temp_align[cam][i] = i - 154
                elif 1958 <= i:
                    temp_align[cam][i] = i - 191

            elif cam == 'c009':
                temp_align[cam][i] = i - 9
    return temp_align


class MTMCT(object):
    def __init__(self, opt):
        self.result_path = None
        self.opt = opt
        self.start = time.time()

        # Load models ====================================================================================================
        # Load detection model
        self.det_model = attempt_load(opt.det_weights + opt.det_name + '.pt')
        self.det_model = self.det_model.cuda().eval().half()
        self.YOLOv10_detect_model = YOLOv10(opt.det_weights + opt.det_name + '.pt')

        # For time measurement
        self.total_times = {'Det': 0, 'Ext': 0, 'MTSC': 0, 'MTMC': 0}
        self.cams = os.listdir(opt.data_dir)
        self.stride = int(self.det_model.stride.max())
        self.img_size = opt.img_size.copy()
        self.img_size[0] = check_img_size(opt.img_size[0], s=self.stride)
        self.img_size[1] = check_img_size(opt.img_size[1], s=self.stride)
        # Load feature extraction model
        feat_ext_model = FeatureExtractor(opt.feat_ext_name, opt.avg_type, opt.feat_ext_weights)
        self.feat_ext_model = feat_ext_model.cuda().eval().half()
        # For feature extraction model
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Prepare ========================================================================================================
        # Prepare output folder
        self.output_dir = opt.output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.result_path = self.output_dir + f'{mtmct_version}.txt'
        # Prepare others
        self.datasets, self.trackers, self.f_nums = {}, {}, []
        self.roi_masks, self.overlap_regions_cam2cam = {}, {}
        self.temp_align = prepare_align(self.cams, self.f_nums)
        for cam in self.cams:
            # Prepare 1
            img_dir = os.path.join(opt.data_dir, cam) + '/frame/*'
            self.datasets[cam] = iter(LoadImages(img_dir, img_size=self.img_size, stride=self.stride))
            self.trackers[cam] = BoTSORT(opt,self.temp_align)
            self.f_nums.append(self.datasets[cam].nf)

            # Prepare 2
            self.roi_masks[cam] = cv2.imread('./preliminary/rois/%s.png' % cam, cv2.IMREAD_GRAYSCALE)
            self.overlap_regions_cam2cam[cam] = {}
            for cam_ in self.cams:
                self.overlap_regions_cam2cam[cam][cam_] = cv2.imread(
                    './preliminary/overlap_zones/%s_%s.png' % (cam, cam_),
                    cv2.IMREAD_GRAYSCALE) if cam_ != cam else None
        # Warm-up models
        with torch.autocast('cuda'):
            for _ in range(10):
                self.det_model(torch.rand((4, 3, self.img_size[0], self.img_size[1]), device='cuda').half())
                self.feat_ext_model(
                    torch.rand((10, 3, opt.patch_size[0], opt.patch_size[1]), device='cuda').half())

        # Temporal alignment 时间对齐的序列
        self.img_h, self.img_w = opt.img_ori_size
        self.next_global_id, self.dunn_index_prev = 0, -1e5
        self.clusters_dict = {}
        self.result = []

    def run_mtmct(self):
        # Run
        for fdx in tqdm(range(0, np.max(self.f_nums) + 1)):
            # 准备图像数据
            batch_img, batch_img_ori, valid_cam = self.generate_image_info(fdx)

            # 目标检测
            preds = self.detect(batch_img, valid_cam)

            # REID
            batch_feat, detection = self.reid(batch_img, batch_img_ori, preds)

            # 单摄像头跟踪
            online_tracks_raw = self.MTSCT_online(batch_feat, detection)

            # 跨摄像头跟踪
            self.mtmct_online(fdx, online_tracks_raw)
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

    def generate_image_info(self, fdx):
        """
        准备图像数据
        """
        # Generate empty batches
        batch_img = torch.zeros((len(self.cams), 3, self.img_size[0], self.img_size[1]), device='cuda').half()
        batch_img_ori = torch.zeros((len(self.cams), 3, opt.img_ori_size[0], opt.img_ori_size[1]),
                                    device='cuda').half()
        # 准备图像数据
        valid_cam = {}
        for cdx, cam in enumerate(self.cams):
            # Read
            valid_cam[cam] = True
            temp_align_cam_fdx_ = self.temp_align[cam][fdx]
            path, img, img_ori, _ = self.datasets[cam].__next__(cam, temp_align_cam_fdx_)
            # Check 这里检查是否有这样的图片存在文件夹中，如果没有则跳过然后到下一个摄像头中
            if img is None:
                valid_cam[cam] = False
                continue

            # Store 把读取到的图片存到batch_img中并且序列化
            batch_img[cdx] = torch.tensor(img / 255.0, device='cuda').half()
            batch_img_ori[cdx] = torch.tensor(img_ori.transpose((2, 0, 1)) / 255.0, device='cuda').half()
        return batch_img, batch_img_ori, valid_cam

    def mtmct_online(self, fdx, online_tracks_raw):
        start = time.time()
        online_tracks_filtered = self.filter_online_tracks(online_tracks_raw)
        # Merge
        online_tracks = []
        for cam in self.cams:
            online_tracks += online_tracks_filtered[cam]
        # Gather current tracking global ids
        online_global_ids = {'c006': [], 'c007': [], 'c008': [], 'c009': []}
        for track in online_tracks:
            if track.global_id is not None:
                online_global_ids[track.cam].append(track.global_id)
        # 获取feat数据并计算特征距离
        online_feats = np.array([track.get_feature(mode=opt.get_feat_mode) for track in online_tracks])
        p_dists = pdist(online_feats, metric='cosine')
        p_dists = np.clip(p_dists, 0, 1)  # 归一化
        # Apply constraints
        self.filter_tracks_by_overlap(online_tracks, p_dists)
        # Clustering =================================================================================================
        # Generate linkage matrix with hierarchical clustering
        linkage_matrix = linkage(p_dists, method='complete')
        ranked_dists = np.sort(list(set(list(linkage_matrix[:, 2]))), axis=None)
        # Observe clusters with adjusting a distance threshold and calculate dunn index
        clusters, dunn_indices, c_dists = [], [], squareform(p_dists)
        for rdx in range(2, ranked_dists.shape[0] + 1):
            if ranked_dists[-rdx] <= opt.mtmc_match_thr:
                clusters.append(fcluster(linkage_matrix, ranked_dists[-rdx] + 1e-5, criterion='distance') - 1)
                dunn_indices.append(dunn(clusters[-1], c_dists))
        if len(clusters) == 0:
            cluster = fcluster(linkage_matrix, ranked_dists[0] - 1e-5, criterion='distance') - 1
        else:
            # Choose the most connected cluster except inappropriate pairs
            # Get the index of the dunn indices where the values suddenly jump.
            dunn_indices.insert(0, 0)
            pos = np.argmax(np.diff(dunn_indices))
            cluster = clusters[pos]
        # Run Multi-target Multi-Camera Tracking =====================================================================
        # Initialize
        num_cluster = len(list(set(list(cluster))))
        # Assign global id to new tracks using other tracks in the same cluster
        for cam_index in range(num_cluster):
            track_idx = np.where(cluster == cam_index)[0]

            # Check index and global id of tracks in same cluster
            infos = []
            for tdx in track_idx:
                if online_tracks[tdx].global_id is not None:
                    infos.append([tdx, online_tracks[tdx].global_id])

            # If some tracks in the cluster already has global id, assign same global id to new tracks
            if len(infos) > 0:
                # Assign global id, Collect tracks, Update feature
                for tdx in track_idx:
                    if online_tracks[tdx].global_id is None:
                        # Sort and get global id with the node with minimum distance
                        sorted_infos = sorted(copy.deepcopy(infos), key=lambda x: c_dists[tdx, x[0]])
                        for info in sorted_infos:
                            if online_tracks[info[0]].global_id not in online_global_ids[online_tracks[tdx].cam]:
                                # Assign global id, Collect
                                online_tracks[tdx].global_id = info[1]
                                if info[1] in self.clusters_dict:
                                    self.clusters_dict[info[1]].add_track(online_tracks[tdx])
                                break
        # Get remaining current tracks
        remain_tracks = [track for track in online_tracks if track.global_id is None]
        # Calculate pairwise distance between previous clusters and current clusters
        dists = pairwise_tracks_dist(self.clusters_dict, remain_tracks, fdx, metric='cosine')
        # Run Hungarian algorithm
        indices = linear_assignment(dists)
        # Match with thresholding
        for row, col in indices:
            if dists[row, col] <= opt.mtmc_match_thr \
                    and list(self.clusters_dict.keys())[row] not in online_global_ids[remain_tracks[col].cam]:
                # Assign global id, Collect track
                remain_tracks[col].global_id = list(self.clusters_dict.keys())[row]
                self.clusters_dict[list(self.clusters_dict.keys())[row]].add_track(remain_tracks[col])
        # If not matched newly starts
        for remain_track in remain_tracks:
            if remain_track.global_id is None:
                # Assign global id, Collect track
                remain_track.global_id = self.next_global_id
                self.clusters_dict[self.next_global_id] = Cluster()
                self.clusters_dict[self.next_global_id].add_track(remain_track)

                # Increase
                self.next_global_id += 1
        # Delete too old cluster
        del_key = [key for key in self.clusters_dict.keys() if
                   fdx - self.clusters_dict[key].end_frame > opt.max_time_differ]
        for key in del_key:
            del self.clusters_dict[key]
        self.total_times['MTMC'] += time.time() - start
        # Logging
        for track in online_tracks:
            left, top, w, h = track.tlwh

            # Expand box, Since gt boxes are not tightly annotated around objects and quite larger than objects
            cx, cy = left + w / 2, top + h / 2
            w, h = w * 1.45, h * 1.45
            left, top = cx - w / 2, cy - h / 2

            # Filter with size, Since gt does not include small boxes
            if w * h / self.img_w / self.img_h < 0.003 or 0.3 < w * h / self.img_w / self.img_h:
                continue
            format = '%d %d %d %d %d %d %d -1 -1' % (
            int(track.cam[-1]), track.global_id, self.temp_align[track.cam][fdx], int(left), int(top), int(w), int(h))
            self.result.append(format)

    def filter_tracks_by_overlap(self, online_tracks, p_dists):
        for i in range(len(online_tracks)):
            for j in range(i + 1, len(online_tracks)):
                # Covert index
                idx = len(online_tracks) * i + j - ((i + 2) * (i + 1)) // 2

                # If same camera
                if online_tracks[i].cam == online_tracks[j].cam:
                    p_dists[idx] = 10
                    continue

                # If the objects are not in overlapping region (i -> j)
                # 根据摄像头与摄像头之间的overlap区域过滤
                overlap_region = self.overlap_regions_cam2cam[online_tracks[i].cam][online_tracks[j].cam]
                x1, y1, x2, y2 = online_tracks[i].x1y1x2y2.astype(np.int32)
                y2 = y2 if y2 < 1080 else 1079
                if overlap_region[y2, (x1 + x2) // 2] == 0:
                    p_dists[idx] = 10
                    continue

                # If the objects are not in overlapping region (j -> i)
                overlap_region = self.overlap_regions_cam2cam[online_tracks[j].cam][online_tracks[i].cam]
                x1, y1, x2, y2 = online_tracks[j].x1y1x2y2.astype(np.int32)
                y2 = y2 if y2 < 1080 else 1079
                if overlap_region[y2, (x1 + x2) // 2] == 0:
                    p_dists[idx] = 10
                    continue

    def filter_online_tracks(self, online_tracks_raw):
        online_tracks_filtered = {}
        for cam in self.cams:
            online_tracks_filtered[cam] = []
            for track in online_tracks_raw[cam]:
                # If not activated
                if not track.is_activated:
                    continue

                # If it has low confidence score
                if track.obs_history[-1][2] <= opt.det_high_thresh:
                    continue

                # Filter detection with small box size, Since gt does not include small boxes
                w, h = track.tlwh[2:]
                if h * w <= self.img_h * self.img_w * opt.min_box_size:
                    continue

                # Filter detections around border, Since gt does not include boxes around border
                x1, y1, x2, y2 = track.x1y1x2y2
                if x1 <= 5 or y1 <= 5 or x2 >= self.img_w - 5 or y2 >= self.img_h - 5:
                    continue

                # Append
                online_tracks_filtered[cam].append(track)

            # Class agnostic NMS, Since gt does not include overlapped boxes
            if 2 <= len(online_tracks_filtered[cam]):
                online_tracks_filtered[cam] = class_agnostic_nms(online_tracks_filtered[cam])
        return online_tracks_filtered

    def MTSCT_online(self, batch_feat, detection):
        """
        返回出四个跟踪器在更新新的帧之后，仍在跟踪的轨迹集合
        """
        start = time.time()
        feat_count, feat = 0, {}
        for cam in self.cams:
            feat[cam] = batch_feat[feat_count:feat_count + len(detection[cam])]
            feat_count += len(detection[cam])
        # Run Multi-target Single-Camera Tracking and online tracks
        online_tracks_raw = {}
        for cam in self.cams:
            online_tracks_raw[cam] = self.trackers[cam].update(cam, detection[cam], feat[cam],self.temp_align)
        self.total_times['MTSC'] += time.time() - start
        return online_tracks_raw

    def reid(self, batch_img, batch_img_ori, preds):
        self.start = time.time()
        batch_patch = torch.zeros((100 * len(self.cams), 3, opt.patch_size[0], opt.patch_size[1]), device='cuda').half()
        det_count, detection = 0, {}
        for pdx, pred in enumerate(preds):
            # Prepare dictionary to store detection results
            detection[self.cams[pdx]] = np.zeros((0, 5))

            # If there are valid predictions
            if len(pred) > 0:
                # Rescale boxes from img_size to im0s size
                debug_pred = pred[:, :4]
                pred[:, :4] = scale_coords(batch_img.shape[2:], pred[:, :4], batch_img_ori.shape[2:4])
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
                    if (x2 - x1) * (y2 - y1) <= self.img_h * self.img_w * opt.min_box_size / 2:
                        continue

                    # Add detections
                    new_box = np.array([(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1), conf.item()])
                    new_box = new_box[np.newaxis, :]
                    detection[self.cams[pdx]] = np.concatenate([detection[self.cams[pdx]], new_box], axis=0)

                    # Get patch
                    patch = batch_img_ori[pdx][:, max(y1, 0):min(y2 + 1, self.img_h),
                            max(x1, 0):min(x2 + 1, self.img_w)]
                    patch = self.normalize(letterbox(patch))
                    # 如果是c008则水平翻转
                    batch_patch[det_count] = torch.fliplr(patch) if self.cams[pdx] == 'c008' else patch
                    det_count += 1
        # Extract features
        with torch.autocast('cuda'):
            batch_patch = batch_patch[:det_count]
            batch_feat = self.feat_ext_model(batch_patch)
        batch_feat = batch_feat.squeeze().cpu().numpy()
        self.total_times['Ext'] += time.time() - self.start
        return batch_feat, detection

    def detect(self, batch_img, valid_cam):
        self.start = time.time()
        # 目标检测阶段
        # Detect =====================================================================================================
        # with torch.autocast('cuda'):
        #     preds = self.det_model(batch_img[list(valid_cam.values())], augment=opt.augment)[0]
        # # NMS之后是最终的检测结果
        # preds = non_max_suppression(preds, opt.conf_thres, opt.iou_thres,
        #                             classes=opt.classes, agnostic=opt.agnostic_nms)
        # for cam_index, cam in enumerate(self.cams):
        #     if not valid_cam[cam]:
        #         preds.insert(cam_index, torch.zeros((0, 6)).cuda().half())
        preds_result = self.YOLOv10_detect_model(batch_img)
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
                tensor_temp_add = torch.tensor(temp_add, dtype=torch.float16).cuda().half()
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
                for track in online_track.finished:
                    for info in track.obs_history:
                        left, top, w, h = caculate_tlwh(info[1])

                        # Expand box, Since gt boxes are not tightly annotated around objects and quite larger than objects
                        cx, cy = left + w / 2, top + h / 2
                        w, h = w * 1.45, h * 1.45
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
                                int(track.cam[-1]), track.global_id, self.temp_align[track.cam][info[0]],
                                int(left), int(top), int(w), int(h)), file=output_file)


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
    with open(outputs_mtmct_pkl, 'wb') as f:
        pickle.dump(mtmct, f)
    return mtmct


def read_pkl_from_file(outputs_mtmct_pkl):
    with open(outputs_mtmct_pkl, 'rb') as file:
        mtmct = pickle.load(file)
    return mtmct


finished_txt = 'outputs/result/finished.txt'
result_dir = 'results'


def debug():
    from tracking import matching
    from scipy.spatial.distance import cdist

    mtmct = read_pkl_from_file(outputs_mtmct_pkl)
    # mtmct.debug_finished()
    track14 = mtmct.trackers['c008'].finished[60]
    reid14 = track14.obs_history[-1][3]
    track94 = mtmct.trackers['c008'].finished[65]
    reid94 = track94.obs_history[0][3]
    reid_dis = cdist([reid14], [reid94], 'cosine')
    reid_dis = reid_dis[0][0]
    print(reid_dis)
    # calculate_results('outputs/ground_truth_validation.txt', finished_txt)


def main():
    mtmct = run()
    # mtmct = read_pkl_from_file(outputs_mtmct_pkl)
    calculate_results('outputs/ground_truth_validation.txt', mtmct.result_path)
    return mtmct


if __name__ == '__main__':
    opt.version = 4
    mtmct_version = f'v{opt.version}'
    outputs_mtmct_pkl = 'outputs/mtmct.pkl'
    main()
    # debug()
