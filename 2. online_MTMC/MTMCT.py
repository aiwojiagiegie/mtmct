import argparse
import os
import random

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

from utils.laneUtils import LaneMaskReader
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

    @property 
    def main_lanes(self):
        # 初始化为空集
        common_lanes = set()
        
        # 遍历所有轨迹
        for i, track in enumerate(self.tracks):
            # 获取当前轨迹的车道集合 - 修改这里，直接访问属性而不是调用
            lanes = track.main_lanes
            if not lanes:  # 如果当前轨迹没有车道,跳过
                continue
                
            # 如果是第一个有效的车道集合,直接赋值
            if not common_lanes:
                common_lanes = lanes
            else:  # 否则取交集
                common_lanes.intersection_update(lanes)
                
        return common_lanes

    @property
    def main_zone(self) -> set:
        """
        获取轨迹簇中所有轨迹的区域类型集合
        :return: 区域类型集合，例如 {'entry', 'middle'}
        """
        zones = set()
        
        # 遍历所有轨迹
        for track in self.tracks:
            zone = track.main_zone
            if zone:  # 如果有区域信息，添加到集合中
                zones.add(zone)
                
        return zones

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
    def __init__(self, opt,YOLOv10_detect_model_path=None):
        self.result_path = None
        self.opt = opt
        self.start = time.time()

        # Load models ====================================================================================================
        # Load detection model
        # self.det_model = attempt_load(opt.det_weights + opt.det_name + '.pt')
        # self.det_model = self.det_model.cuda().eval().half()
        if YOLOv10_detect_model_path is None:
            self.YOLOv10_detect_model = YOLOv10(opt.yolo10_model)
        else:
            self.YOLOv10_detect_model = YOLOv10(YOLOv10_detect_model_path)
        # For time measurement
        self.total_times = {'Det': 0, 'Ext': 0, 'MTSC': 0, 'MTMC': 0}
        self.cams = sorted(os.listdir(opt.data_dir))
        self.stride = 32
        # self.stride = int(self.det_model.stride.max())
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
        for cam in self.cams:
            # Prepare 1
            img_dir = os.path.join(opt.data_dir, cam) + '/frame/*'
            self.datasets[cam] = iter(LoadImages(img_dir, img_size=self.img_size, stride=self.stride))
            self.trackers[cam] = BoTSORT(opt)
            self.f_nums.append(self.datasets[cam].nf)

            # Prepare 2
            self.roi_masks[cam] = cv2.imread('./preliminary/rois/%s.png' % cam, cv2.IMREAD_GRAYSCALE)
            self.overlap_regions_cam2cam[cam] = {}
            for cam_ in self.cams:
                self.overlap_regions_cam2cam[cam][cam_] = cv2.imread(
                    './preliminary/overlap_zones/%s_%s.png' % (cam, cam_),
                    cv2.IMREAD_GRAYSCALE) if cam_ != cam else None
        self.temp_align = prepare_align(self.cams, self.f_nums)
        for tracker in self.trackers.values():
            tracker.temp_align = self.temp_align
        # Warm-up models
        with torch.autocast('cuda'):
            for _ in range(10):
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
            feat, detection = self.reid(batch_img, batch_img_ori, preds)
            # 单摄像头跟踪
            online_tracks_raw = self.MTSCT_online(feat, detection)

            # 跨摄像头跟踪
            self.mtmct_online(fdx, online_tracks_raw)
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
        # 使用完全链接法(complete linkage)构建层次聚类的连接矩阵
        # 输入p_dists: 压缩格式的距离矩阵，例如对于5个轨迹A,B,C,D,E之间的距离:
        # p_dists = [0.1,   0.8,   0.5,   0.9,   0.7,   0.3,   0.6,   0.4,   0.2,   0.5]
        #            A->B   A->C   A->D   A->E   B->C   B->D   B->E   C->D   C->E   D->E
        
        # 返回linkage_matrix: (n-1)×4的矩阵，每行代表一次合并操作
        # 格式: [簇1编号, 簇2编号, 合并距离, 新簇中的样本数]
        # 例如对于上述距离矩阵，返回结果可能是:
        # [[0, 1, 0.1, 2],    # 第一次：合并A和B(距离0.1)，形成簇AB(2个样本)
        #  [2, 4, 0.2, 2],    # 第二次：合并C和E(距离0.2)，形成簇CE(2个样本)
        #  [5, 3, 0.4, 3],    # 第三次：合并簇CE和D(距离0.4)，形成簇CDE(3个样本)
        #                      # 选择CE和D合并是因为:
        #                      # 1. CE到D的距离 = max(C到D, E到D) = max(0.4, 0.5) = 0.4
        #                      # 2. AB到D的距离 = max(A到D, B到D) = max(0.5, 0.3) = 0.5
        #                      # 3. 0.4 < 0.5，所以优先合并CE和D
        #  [6, 7, 0.8, 5]]    # 第四次：合并簇AB和簇CDE(距离0.8)，形成最终簇ABCDE(5个样本)
        
        # 编号说明:
        # - 原始轨迹编号：A=0, B=1, C=2, D=3, E=4
        # - 新簇编号：从N开始(N为原始样本数)，即从5开始
        # - 第一次合并后AB簇编号为5
        # - 第二次合并后CE簇编号为6
        # - 第三次合并后CDE簇编号为7
        #
        # 距离计算（完全链接法）：
        # - 两个簇之间的距离 = 簇间最远两点的距离
        # - 例如：CE到D的距离 = max(C到D的距离, E到D的距离) = max(0.4, 0.5) = 0.4

        # method='complete'表示使用完全链接法：
        # 两个簇之间的距离定义为它们中最远的两个点之间的距离
        # 例如：当合并簇AB和簇CDE时，距离为max(A->C,A->D,A->E,B->C,B->D,B->E)=0.8
        linkage_matrix = linkage(p_dists, method='complete')
        # 从linkage_matrix中提取所有唯一的距离值并排序
        # 例如，对于5个轨迹的linkage_matrix:
        # [[0, 1, 0.1, 2],    # 第一次合并：A和B
        #  [2, 4, 0.2, 2],    # 第二次合并：C和E
        #  [5, 3, 0.4, 3],    # 第三次合并：CE和D
        #  [6, 7, 0.8, 5]]    # 第四次合并：AB和CDE
        
        # 1. linkage_matrix[:, 2] 提取第3列(即所有合并距离):
        #    [0.1, 0.2, 0.4, 0.8]
        
        # 2. list()将numpy数组转换为Python列表:
        #    [0.1, 0.2, 0.4, 0.8]
        
        # 3. set()创建集合去除重复值(本例中没有重复值，但其他情况可能有):
        #    {0.1, 0.2, 0.4, 0.8}
        
        # 4. 再次list()将集合转回列表(因为set无序):
        #    [0.1, 0.2, 0.4, 0.8]
        
        # 5. np.sort()对列表进行排序，axis=None确保返回一维数组:
        #    [0.1, 0.2, 0.4, 0.8]
        
        # 最终得到ranked_dists：所有可能的合并距离阈值，从小到大排序
        # 这些阈值将用于后续尝试不同的聚类方案：
        # - 0.1: 只合并非常相似的轨迹
        # - 0.2: 合并相似度稍低的轨迹
        # - 0.4: 合并相似度中等的轨迹
        # - 0.8: 合并相似度较低的轨迹
        ranked_dists = np.sort(list(set(list(linkage_matrix[:, 2]))), axis=None)
        # Observe clusters with adjusting a distance threshold and calculate dunn index
        # 初始化三个列表/矩阵
        clusters = []          # 存储不同阈值下的聚类结果
        dunn_indices = []      # 存储每个聚类结果的Dunn指数(评估聚类质量)
        c_dists = squareform(p_dists)  # 将压缩距离矩阵转换为完整的方阵形式
        # 示例: 如果ranked_dists = [0.1, 0.2, 0.4, 0.8]
        # range(2, 5) = [2, 3, 4]，即从后往前尝试不同的阈值:
        # rdx=2: ranked_dists[-2] = 0.4
        # rdx=3: ranked_dists[-3] = 0.2
        # rdx=4: ranked_dists[-4] = 0.1
        for rdx in range(2, ranked_dists.shape[0] + 1):
            # 只考虑小于匹配阈值的距离
            # 例如：如果opt.mtmc_match_thr = 0.5
            # 则只考虑距离<=0.5的合并方案
            if ranked_dists[-rdx] <= opt.mtmc_match_thr:
                # fcluster: 使用给定阈值对层次聚类结果进行切割，获得聚类分配结果
                # ranked_dists[-rdx] + 1e-5: 添加小量以确保稳定性
                # criterion='distance': 基��距离进行切割
                # -1: 将聚类ID从1-based转换为0-based
                # fcluster根据距离阈值对层次聚类结果进行切割，返回每个样本的簇标签
                # 示例: 假设有5个轨迹A,B,C,D,E，linkage_matrix为:
                # [[0, 1, 0.1, 2],    # 合并A和B，距离0.1
                #  [2, 4, 0.2, 2],    # 合并C和E，距离0.2 
                #  [5, 3, 0.4, 3],    # 合并CE和D，距离0.4
                #  [6, 7, 0.8, 5]]    # 合并AB和CDE，距离0.8
                # 
                # 当ranked_dists[-rdx]=0.3时(介于0.2和0.4之间):
                # 返回[0,0,1,2,1] -> AB一组(id=0)，CE一组(id=1)，D单独一组(id=2)
                #
                # 当ranked_dists[-rdx]=0.5时(介于0.4和0.8之间):
                # 返回[0,0,1,1,1] -> AB一组(id=0)，CDE一组(id=1)
                #
                # 当ranked_dists[-rdx]=0.9时(大于0.8):
                # 返回[0,0,0,0,0] -> 所有轨迹在一组(id=0)
                new_cluster = fcluster(linkage_matrix, ranked_dists[-rdx] + 1e-5, criterion='distance') - 1
                clusters.append(new_cluster)
                # 计算当前聚类结果的Dunn指数并存储
                # Dunn指数衡量聚类的紧密度和分离度：
                # - 更高的值表示更好的聚类结果
                # - 计算方式：最小簇间距离 / 最大簇内距离
                # 计算当前聚类结果的Dunn指数
                # 示例: 对聚类结果[0,0,1,2,1] (AB一组，CE一组，D单独组)的Dunn指数计算
                #
                # 假设c_dists距离矩阵为:
                # [    A    B    C    D    E
                #  A [0.0, 0.1, 0.7, 0.6, 0.8],  # A的距离
                #  B [0.1, 0.0, 0.7, 0.5, 0.7],  # B的距离
                #  C [0.7, 0.7, 0.0, 0.4, 0.2],  # C的距离
                #  D [0.6, 0.5, 0.4, 0.0, 0.3],  # D的距离
                #  E [0.8, 0.7, 0.2, 0.3, 0.0]   # E的距离
                # ]
                #
                # 1. 计算簇内距离(intra-cluster distances):
                #    簇0(AB): dist(A,B) = 0.1
                #    簇1(CE): dist(C,E) = 0.2
                #    簇2(D):  没有簇内距离(单点)
                #    最大簇内距离 = max(0.1, 0.2) = 0.2
                #
                # 2. 计算簇间距离(inter-cluster distances):
                #    簇0到簇1: min(dist(A,C), dist(A,E), dist(B,C), dist(B,E))
                #             = min(0.7, 0.8, 0.7, 0.7) = 0.7
                #    簇0到簇2: min(dist(A,D), dist(B,D))
                #             = min(0.6, 0.5) = 0.5
                #    簇1到簇2: min(dist(C,D), dist(E,D))
                #             = min(0.4, 0.3) = 0.3
                #    最小簇间距离 = min(0.7, 0.5, 0.3) = 0.3
                #
                # 3. Dunn指数计算:
                #    Dunn = 最小簇间距离 / 最大簇内距离
                #         = 0.3 / 0.2 
                #         = 1.5
                #
                # 这个Dunn指数表明:
                # - 簇内距离最大为0.2，说明簇内对象相对紧密
                # - 簇间距离最小为0.3，说明簇间有一定分离度
                # - 指数大于1说明聚类结果可以接受，但分离度不是特别理想
                new_dunn = dunn(clusters[-1], c_dists)
                dunn_indices.append(new_dunn)
        # 示例输出：
        # 假设ranked_dists = [0.1, 0.2, 0.4, 0.8]，mtmc_match_thr = 0.5
        # 则可能得到：
        # clusters = [
        #   [0,0,1,1,1],  # 使用阈值0.4的聚类结果：AB在一组，CDE在一组
        #   [0,0,1,2,1],  # 使用阈值0.2的聚类结果：AB一组，CE一组，D单独一组
        #   [0,0,1,2,3]   # 使用阈值0.1的聚类结果：AB一组，其他各自一组
        # ]
        # dunn_indices = [0.4, 0.6, 0.8]  # 每个聚类结果对应的Dunn指数
        # 选择最佳聚类结果
        if len(clusters) == 0:
            # 如果没有找到任何有效的聚类结果(所有距离都大于mtmc_match_thr)
            # 则使用最小的距离阈值(ranked_dists[0])进行聚类
            # 减去一个很小的值(1e-5)是为了确保稳定性
            # criterion='distance'表示基于距离进行切割
            # -1是将聚类ID从1-based转换为0-based
            cluster = fcluster(linkage_matrix, ranked_dists[0] - 1e-5, criterion='distance') - 1
        else:
            # 如果有多个有效的聚类结果，选择Dunn指数突变点对应的聚类方案
            
            # 在dunn_indices开头插入0作为基准点
            # 例如：原始dunn_indices = [0.4, 0.6, 0.8]
            # 插入后变为：[0, 0.4, 0.6, 0.8]
            dunn_indices.insert(0, 0)
            
            # np.diff()计算相邻元素的差值
            # 例如：[0, 0.4, 0.6, 0.8] -> [0.4, 0.2, 0.2]
            # np.argmax()找出差值最大的位置
            # 这表示Dunn指数突然显著提升的位置，即聚类质量有明显改善的阈值
            pos = np.argmax(np.diff(dunn_indices))
            
            # 使用该位置对应的聚类结果
            # 例如：如果pos=1，表示选择clusters[1]对应的聚类方案
            # 这种方案在保持聚类质量的同时，避免过度分割
            cluster = clusters[pos]
        # Run Multi-target Multi-Camera Tracking =====================================================================
        # Initialize
        num_cluster = len(list(set(list(cluster))))

        # 遍历每个聚类，为新轨迹分配全局ID
        for cam_index in range(num_cluster):
            # 获取当前聚类中所有轨迹的索引
            # 例如：cluster=[0,0,1,1,2] 当cam_index=1时
            # track_idx将包含值为1的所有索引位置：[2,3]
            track_idx = np.where(cluster == cam_index)[0]

            # 收集当前聚类中已有全局ID的轨迹信息
            infos = []
            for tdx in track_idx:
                # 如果轨迹已经有全局ID，将其索引和全局ID保存
                if online_tracks[tdx].global_id is not None:
                    infos.append([tdx, online_tracks[tdx].global_id])

            # 如果当前聚类中存在已有全局ID的轨迹
            if len(infos) > 0:
                # 为聚类中没有全局ID的轨迹分配ID
                for tdx in track_idx:
                    if online_tracks[tdx].global_id is None:
                        # 根据与当前轨迹的距离对已有全局ID的轨迹进行排序
                        # c_dists[tdx, x[0]]表示当前轨迹与已有ID轨迹的距离
                        # 距离越小表示越可能是同一个目标
                        sorted_infos = sorted(copy.deepcopy(infos), key=lambda x: c_dists[tdx, x[0]])
                        
                        # 遍历排序后的轨迹信息
                        for info in sorted_infos:
                            # 检查该全局ID是否已在当前摄像头中使用
                            # 避免在同一个摄像头中出现相同的全局ID
                            if online_tracks[info[0]].global_id not in online_global_ids[online_tracks[tdx].cam]:
                                # 分配全局ID
                                online_tracks[tdx].global_id = info[1]
                                # 如果该全局ID对应的轨迹簇存在，则添加当前轨迹
                                if info[1] in self.clusters_dict:
                                    self.clusters_dict[info[1]].add_track(online_tracks[tdx])
                                break
        # 获取所有还未分配全局ID的轨迹
        # 使用列表推导式筛选出global_id为None的轨迹
        remain_tracks = [track for track in online_tracks if track.global_id is None]

        # 计算已有轨迹簇(clusters_dict)和剩余未分配轨迹(remain_tracks)之间的成对距离
        # clusters_dict: 包含所有已分配全局ID的轨迹簇的字典，key为全局ID
        # remain_tracks: 待分配全局ID的轨迹列表
        # fdx: 当前帧号，用于时间相关的距离计算
        # metric='cosine': 使用余弦距离作为度量标准
        # 返回一个距离矩阵 dists[i,j]表示第i个已有簇与第j个待分配轨迹的距离
        dists = pairwise_tracks_dist(self.clusters_dict, remain_tracks, fdx, metric='cosine')

        # 使用匈牙利算法(Hungarian Algorithm)求解最优匹配问题
        # 该算法能在O(n^3)时间内找到最优的一对一匹配
        # 返回最优匹配的索引对列表 indices = [[cluster_idx1, track_idx1], [cluster_idx2, track_idx2], ...]
        # 每个匹配对表示一个已有簇和一个待分配轨迹的最优匹配
        indices = linear_assignment(dists)
        # Match with thresholding
        for row, col in indices:
            # 1. 检查距离阈值
            distance_match = dists[row, col] <= opt.sec_mtmc_match_thr
            
            # 2. 检查全局ID是否已在该摄像头中使用
            cluster_keys = list(self.clusters_dict.keys())
            current_cluster_id = cluster_keys[row]
            current_cam = remain_tracks[col].cam
            id_not_in_cam = current_cluster_id not in online_global_ids[current_cam]
            
            # 3. 组合条件判断
            if distance_match and id_not_in_cam:
                # 条件满足时:
                # 1. 将已有簇的全局ID分配给新轨迹
                remain_tracks[col].global_id = current_cluster_id
                # 2. 将新轨迹添加到已有簇中
                self.clusters_dict[current_cluster_id].add_track(remain_tracks[col])
        # If not matched newly starts
        # 如果新轨迹没有匹配到已有簇，则为其分配新的全局ID
        for remain_track in remain_tracks:
            if remain_track.global_id is None:
                # 1. 为新轨迹分配新的全局ID
                remain_track.global_id = self.next_global_id
                # 2. 创建新的簇
                self.clusters_dict[self.next_global_id] = Cluster()
                # 3. 将新轨迹添加到新簇中
                self.clusters_dict[self.next_global_id].add_track(remain_track)
                # 4. 增加全局ID计数器
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
            confidence  = track.confidence
            # Expand box, Since gt boxes are not tightly annotated around objects and quite larger than objects
            cx, cy = left + w / 2, top + h / 2
            w, h = w * 1.45, h * 1.45
            left, top = cx - w / 2, cy - h / 2

            # Filter with size, Since gt does not include small boxes
            if w * h / self.img_w / self.img_h < 0.003 or 0.3 < w * h / self.img_w / self.img_h:
                continue
            format = '%d %d %d %d %d %d %d %.2f -1' % (
                int(track.cam[-1]), track.global_id, self.temp_align[track.cam][fdx], int(left), int(top), int(w),
                int(h) , confidence)
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
                pass

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

    def MTSCT_online(self, feat, detection):
        """
        返回出四个跟踪器在更新新的帧之后，仍在跟踪的轨迹集合
        """
        start = time.time()
        online_tracks_raw = {}
        for cam in self.cams:
            online_tracks_raw[cam] = self.trackers[cam].update(cam, detection[cam], feat[cam], self.temp_align)
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
    # with open(outputs_mtmct_pkl, 'wb') as f:
    #     pickle.dump(mtmct, f)
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

# 模型配置文件
model_yaml_path = "./yolov10/ultralytics/cfg/models/v10/yolov10n.yaml"
# 数据集配置文件
data_yaml_path = './yolov10/datasets/multi_class/data.yaml'
# 预训练模型

if __name__ == '__main__':
    if opt.train:
        pretrain_type = opt.pretrain_type
        pre_model_name = f'/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/2. online_MTMC/yolov10/models/yolov10{pretrain_type}.pt'
        # 加载预训练模型
        # model = YOLOv10(model_yaml_path).load(pre_model_name)
        model = YOLOv10(model_yaml_path)
        pre_model_name_last = pre_model_name.split('/')[-1]
        # 训练模型
        epochs = opt.epoch
        batch = opt.batch
        # 生成一个六位的随机数
        random_suffix = random.randint(100000, 999999)
        save_path = f'UA-DETRAC_pre/model_name_{pre_model_name_last}/epochs_{epochs}/batch_{batch}/{random_suffix}'
        results = model.train(data=data_yaml_path,
                              epochs=epochs,
                              batch=batch,
                              name=f"{save_path}", device=opt.gpu)
        outputs_mtmct_pkl = f'../../yolov10/runs/detect/{save_path}/mtmct.pkl'
        mtmct_version = f'version/{save_path}/v1'
        mtmct = MTMCT(opt,f'../../yolov10/runs/detect/{save_path}/weights/best.pt')
        with torch.no_grad():
            mtmct.run_mtmct()
        with open(outputs_mtmct_pkl, 'wb') as f:
            pickle.dump(mtmct, f)
        calculate_results('outputs/ground_truth_validation.txt', mtmct.result_path)
    else:
        # opt.version = 6
        mtmct_version = f'version/v{opt.version}'
        outputs_mtmct_pkl = 'outputs/mtmct.pkl'
        main()
        # debug()
