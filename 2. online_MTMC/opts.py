import argparse


class Opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # Options for detection
        self.parser.add_argument('--det_name', type=str, default='best')
        self.parser.add_argument('--det_weights', type=str, default='./preliminary/det_weights/my/')
        self.parser.add_argument('--img_size', type=int, default=[1080, 1920], help='inference size (pixels)')
        self.parser.add_argument('--classes', type=int, default=[2, 5, 7], help='filter by class')
        self.parser.add_argument('--conf_thres', type=float, default=0.1, help='object confidence threshold')
        self.parser.add_argument('--iou_thres', type=float, default=0.7, help='IoU threshold for NMS')
        self.parser.add_argument('--agnostic_nms', default=True, action='store_true', help='class-agnostic NMS')
        self.parser.add_argument('--augment', default=False, action='store_true', help='augmented inference')

        # Options for feature extraction
        self.parser.add_argument('--feat_ext_name', type=str, default='resnet50_ibn_a')
        self.parser.add_argument('--feat_ext_weights', type=str, default='./preliminary/feat_ext_weights/')
        self.parser.add_argument('--avg_type', type=str, default='gap')
        self.parser.add_argument('--patch_size', type=int, default=[384, 384], help='inference size (pixels)')

        # Options for MTSC
        self.parser.add_argument("--det_high_thresh", type=float, default=0.4)
        self.parser.add_argument("--det_low_thresh", type=float, default=0.1)
        self.parser.add_argument("--cos_thr", type=float, default=0.6)
        self.parser.add_argument("--iou_thr", type=float, default=0.6)
        self.parser.add_argument("--max_time_lost", type=int, default=200)

        # Options for MTMC
        self.parser.add_argument('--get_feat_mode', type=str, default='best')
        self.parser.add_argument("--max_time_differ", type=int, default=2500)
        self.parser.add_argument("--mtmc_match_thr", type=float, default=0.65)
        
        # Options for grouped Hungarian MTMC (adjacent-cam, per-lane matching)
        self.parser.add_argument("--camera_order", type=str, default="41,42,43,44,45,46", 
                                help="相邻摄像头顺序，逗号分隔")
        self.parser.add_argument("--hungarian_gate_cost", type=float, default=1e6,
                                help="匈牙利匹配门控后的大代价值")
        self.parser.add_argument("--reid_metric", type=str, default="cosine",
                                help="ReID特征距离度量：cosine 或 euclidean")
        self.parser.add_argument("--use_lane_grouping", type=bool, default=True,
                                help="是否按车道分组进行匈牙利匹配")
        self.parser.add_argument("--assign_first_cam_id", type=bool, default=True,
                                help="是否为首个摄像头在线轨迹先行分配全局ID")
        
        
        # ============================
        # MTMC 外观记忆（在 enable_mtmct_memory=True 时生效）的可调参数
        # 作用：稳定外观表征、防漂移、提升遮挡恢复；用于论文消融与工程调参
        # ============================

        # 质量与步长相关
        self.parser.add_argument('--min_quality', type=float, default=0.2,
                                 help='轨迹质量下限（用于质量感知EMA步长裁剪）')
        self.parser.add_argument('--ema_momentum', type=float, default=0.9,
                                 help='EMA动量（越大越信任历史，更新越慢）')
        self.parser.add_argument('--update_skip_thr', type=float, default=0.4,
                                 help='若新特征与记忆的余弦距离>阈值，则跳过本次更新')
        self.parser.add_argument('--update_every_n', type=int, default=1,
                                 help='每N次触发时才执行一次记忆更新（限制更新频率）')
        self.parser.add_argument('--quality_conf_weight', type=float, default=0.6,
                                 help='质量分数中检测置信度的权重；面积权重=1-该值')

        # 冷启动与鲁棒更新
        self.parser.add_argument('--bootstrap_frames', type=int, default=2,
                                 help='ID创建后的前K次更新跳过（防冷启动噪声）')
        self.parser.add_argument('--clip_update_beta_max', type=float, default=0.5,
                                 help='单次EMA步长beta的最大值（抗异常大步长）')
        self.parser.add_argument('--robust_update', type=bool, default=False,
                                 help='是否对(new-old)做鲁棒裁剪（近似Huber）后再EMA')
        self.parser.add_argument('--huber_delta', type=float, default=1.0,
                                 help='鲁棒裁剪阈值（元素级 |Δ|>delta时进行截断）')
        self.parser.add_argument('--dual_update', type=str, default='pre_and_cur',
                                 help='pre_and_cur | cur_only：仅更新当前轨迹或同时更新两侧')
        self.parser.add_argument('--memory_reset_thr', type=float, default=0.85,
                                 help='若余弦距离>阈值，直接用新特征重置记忆（应对剧变）')

        # Others
        self.parser.add_argument('--data_dir', type=str, default='../dataset/HST/real')
        self.parser.add_argument('--output_dir', type=str, default='./output_HST/result/')
        self.parser.add_argument('--min_box_size', type=int, default=0.0005, help='minimum box size')
        self.parser.add_argument('--img_ori_size', type=int, default=[1080, 1920], help='original image size (pixels)')
        self.parser.add_argument('--version', type=str, default='n4', help='original image size (pixels)')
        # self.parser.add_argument('--version', type=str, default='26', help='original image size (pixels)')
        self.parser.add_argument("--train", help="是否训练新的模型", dest="train", type=bool, default="")
        self.parser.add_argument("--draw_debug", help="是否绘制debug视频", dest="draw_debug", type=bool, default=False)
        self.parser.add_argument("--epoch", help="训练轮次", dest="epoch", type=int, default="300")
        self.parser.add_argument("--batch", help="训练batch", dest="batch", type=int, default="2")
        self.parser.add_argument("--gpu", help="训练用的卡id", dest="gpu", type=int, default="0")
        self.parser.add_argument("--yolo10_model", help="yolo模型权重", dest="yolo10_model", type=str, default="runs\\best.pt")
        # self.parser.add_argument("--yolo10_model", help="yolo模型权重", dest="yolo10_model", type=str, default="runs\detect\yolo_train\model_name_yolov10s.pt\epochs_200\\batch_322\weights\\best.pt")
        self.parser.add_argument("--pretrain_type", help="yolo预训练模型类型", dest="pretrain_type", type=str, default="s")
        self.parser.add_argument("--reid_path", help="reid模型地址", dest="reid_path", type=str, default="../1. train_feat_ext/logs/")
        self.parser.add_argument("--baseline_reid", help="是否使用baseline的reid", dest="baseline_reid", type=bool, default=True)
        # ============================
        # 消融实验控制参数 (Ablation Study Switches)
        # ============================
        self.parser.add_argument('--use_topology', type=bool, default=False,
                                help='【创新点1】是否使用拓扑感知的分层分组匹配策略（摄像头顺序+车道分组+时间窗口）')
        self.parser.add_argument('--use_hungarian', type=bool, default=False,
                                help='【创新点2】是否使用匈牙利算法（True=匈牙利, False=贪心）')
        # Innovation toggles (coarse on/off)
        self.parser.add_argument("--enable_mtmct_memory", help="是否启用在线特征记忆与质量加权更新", dest="enable_mtmct_memory", type=bool, default=False)

        # ============================
        # 消融实验可调参数 (Ablation Hyperparameters)
        # ============================
        self.parser.add_argument('--mtmc_temporal_window', type=int, default=200,
                                 help='MTMC层候选时间窗口（帧数），独立于SCT层的max_time_lost；'
                                      '仅控制跨摄像头匹配时向前回溯多少帧寻找候选')
        self.parser.add_argument('--lane_penalty_weight', type=float, default=1e6,
                                 help='车道不匹配惩罚权重：0=无车道约束，'
                                      '中间值(0.05~0.3)=软惩罚，1e6=等效硬约束')
        self.parser.add_argument('--speed_tolerance', type=float, default=float('inf'),
                                 help='速度可行性容差（帧数）：相邻摄像头对的期望行驶帧差±tolerance范围内不惩罚，'
                                      '超出范围则在代价矩阵上叠加惩罚；inf=不启用速度约束')
        self.parser.add_argument('--speed_penalty_weight', type=float, default=0.5,
                                 help='速度可行性惩罚系数：对超出tolerance部分的帧差偏差乘以该权重')
        self.parser.add_argument('--expected_travel_frames', type=int, default=100,
                                 help='相邻摄像头对间车辆的期望行驶帧差（默认值，所有相邻对共用）')

    def parse(self):
        return self.parser.parse_args()


opt = Opts().parse()
