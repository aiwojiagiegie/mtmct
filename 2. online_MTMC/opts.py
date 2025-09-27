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

        # ============================
        # MTMC（跨摄在线关联）消融与分组相关参数
        # 这些开关便于一键对比：贪心 vs 分组匈牙利、是否只在相邻摄像头匹配、是否按车道分组、
        # 门控阈值、质量感知记忆等。用于论文消融与工程复现实验。
        # ============================

        # 是否启用“相邻摄像头 × 车道分组 × 匈牙利一对一”匹配主路径
        self.parser.add_argument('--use_grouped_hungarian', type=bool, default=True,
                                 help='Enable grouped Hungarian (adjacent cameras × lane grouping) for MTMCT')

        # 异常时是否回退到贪心匹配（例如代价矩阵退化或外部错误）
        self.parser.add_argument('--greedy_fallback', type=bool, default=False,
                                 help='Fallback to greedy matching when Hungarian path fails')

        # 仅在拓扑上相邻的摄像头对之间进行跨摄匹配，强门控搜索空间
        self.parser.add_argument('--topology_adjacent_only', type=bool, default=True,
                                 help='Restrict cross-camera matching to adjacent camera pairs by topology')

        # 相机拓扑/邻接图配置文件路径（可为空）。如提供，可包含方向信息以进一步门控
        self.parser.add_argument('--topology_graph', type=str, default='',
                                 help='Optional path to camera adjacency/topology file (with optional direction)')

        # 是否启用“按车道分组”以拆分大匹配为多个小问题，降低跨车道串扰
        self.parser.add_argument('--lane_grouping', type=bool, default=True,
                                 help='Enable lane-based grouping inside each adjacent camera pair')

        # 严格屏蔽跨车道候选（True：直接不可达；False：通过惩罚增大代价）
        self.parser.add_argument('--lane_mask_strict', type=bool, default=False,
                                 help='Hard-mask cross-lane candidates instead of only penalizing them')

        # 车道不一致惩罚系数/大代价（当非严格屏蔽时生效）
        self.parser.add_argument('--lane_penalty', type=float, default=5.0,
                                 help='Penalty (large cost) for lane inconsistency when not hard-masked')

        # 时序/速度/外观门控阈值：用于预筛掉显著不合理的候选，降低计算量与误配
        self.parser.add_argument('--gating_time_thr', type=float, default=2.0,
                                 help='Temporal gating threshold (e.g., seconds or frame-normalized)')
        self.parser.add_argument('--gating_speed_thr', type=float, default=30.0,
                                 help='Speed/direction gating threshold (domain-specific units)')
        self.parser.add_argument('--gating_app_thr', type=float, default=0.65,
                                 help='Appearance distance threshold for gating')

        # 是否启用“质量感知的 EMA 记忆更新”，抑制低质观测导致的外观漂移
        self.parser.add_argument('--use_quality_ema', type=bool, default=True,
                                 help='Enable quality-aware EMA update for global ID memory')

        # EMA 动量（越大越重视旧记忆），建议与实现中的 self.ema_momentum 对齐
        self.parser.add_argument('--ema_momentum', type=float, default=0.9,
                                 help='EMA momentum for memory update (higher = trust history more)')

        # 外观来源模式：current（仅当前帧）、memory（仅记忆）、hybrid（混合使用）
        self.parser.add_argument('--memory_mode', type=str, default='hybrid',
                                 help='Appearance source mode: current | memory | hybrid')

        # Others
        self.parser.add_argument('--data_dir', type=str, default='../dataset/HST/real')
        self.parser.add_argument('--output_dir', type=str, default='./output_HST/result/')
        self.parser.add_argument('--min_box_size', type=int, default=0.0005, help='minimum box size')
        self.parser.add_argument('--img_ori_size', type=int, default=[1080, 1920], help='original image size (pixels)')
        self.parser.add_argument('--version', type=str, default='n6', help='original image size (pixels)')
        # self.parser.add_argument('--version', type=str, default='26', help='original image size (pixels)')
        self.parser.add_argument("--train", help="是否训练新的模型", dest="train", type=bool, default="")
        self.parser.add_argument("--draw_debug", help="是否绘制debug视频", dest="draw_debug", type=bool, default=False)
        self.parser.add_argument("--epoch", help="训练轮次", dest="epoch", type=int, default="300")
        self.parser.add_argument("--batch", help="训练batch", dest="batch", type=int, default="2")
        self.parser.add_argument("--gpu", help="训练用的卡id", dest="gpu", type=int, default="0")
        self.parser.add_argument("--yolo10_model", help="yolo模型权重", dest="yolo10_model", type=str, default="runs\detect\yolo_train\model_name_best_multiple3.pt\epochs_200\\batch_326\weights\\best.pt")
        # self.parser.add_argument("--yolo10_model", help="yolo模型权重", dest="yolo10_model", type=str, default="runs\detect\yolo_train\model_name_yolov10s.pt\epochs_200\\batch_322\weights\\best.pt")
        self.parser.add_argument("--pretrain_type", help="yolo预训练模型类型", dest="pretrain_type", type=str, default="s")
        self.parser.add_argument("--reid_path", help="reid模型地址", dest="reid_path", type=str, default="../1. train_feat_ext/logs/")
        self.parser.add_argument("--baseline_reid", help="是否使用baseline的reid", dest="baseline_reid", type=bool, default=True)
        self.parser.add_argument('--use_topology', type=bool, default=True,
                                help='Whether to use topology-based matching strategy')

    def parse(self):
        return self.parser.parse_args()


opt = Opts().parse()
