import argparse

dataset_hst_default = '/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/dataset/HST/real'


class Opts:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # Options for detection
        self.parser.add_argument('--det_name', type=str, default='best')
        self.parser.add_argument('--det_weights', type=str, default='/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/2. online_MTMC/preliminary/det_weights/my/')
        self.parser.add_argument('--img_size', type=int, default=[1080, 1920], help='inference size (pixels)')
        self.parser.add_argument('--classes', type=int, default=[2, 5, 7], help='filter by class')
        self.parser.add_argument('--conf_thres', type=float, default=0.1, help='object confidence threshold')
        self.parser.add_argument('--iou_thres', type=float, default=0.7, help='IoU threshold for NMS')
        self.parser.add_argument('--agnostic_nms', default=True, action='store_true', help='class-agnostic NMS')
        self.parser.add_argument('--augment', default=False, action='store_true', help='augmented inference')

        # Options for feature extraction
        self.parser.add_argument('--feat_ext_name', type=str, default='resnet50_ibn_a')
        self.parser.add_argument('--feat_ext_weights', type=str, default='/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/2. online_MTMC/preliminary/feat_ext_weights/')
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

        # Others
        self.parser.add_argument('--data_dir', type=str, default=dataset_hst_default)
        self.parser.add_argument('--output_dir', type=str, default='/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/2. online_MTMC/output_HST/result/')
        self.parser.add_argument('--min_box_size', type=int, default=0.0005, help='minimum box size')
        self.parser.add_argument('--img_ori_size', type=int, default=[1080, 1920], help='original image size (pixels)')
        self.parser.add_argument('--version', type=str, default='6', help='original image size (pixels)')
        # self.parser.add_argument('--version', type=str, default='26', help='original image size (pixels)')
        self.parser.add_argument("--train", help="是否训练新的模型", dest="train", type=bool, default="")
        self.parser.add_argument("--draw_debug", help="是否绘制debug视频", dest="draw_debug", type=bool, default=False)
        self.parser.add_argument("--epoch", help="训练轮次", dest="epoch", type=int, default="300")
        self.parser.add_argument("--batch", help="训练batch", dest="batch", type=int, default="2")
        self.parser.add_argument("--gpu", help="训练用的卡id", dest="gpu", type=int, default="0")
        self.parser.add_argument("--yolo10_model", help="yolo模型权重", dest="yolo10_model", type=str, default="/home/chatmindai/project/zhangkun/yolov10/runs/detect/UA-DETRAC_pre/model_name_yolov10s.pt/epochs_200/batch_327/weights/best.pt")
        self.parser.add_argument("--pretrain_type", help="yolo预训练模型类型", dest="pretrain_type", type=str, default="s")
        self.parser.add_argument("--reid_path", help="reid模型地址", dest="reid_path", type=str, default="/home/chatmindai/project/zhangkun/vehicle_reid_itsc2023/logs/Veri776/MBR_4G/10/")
        self.parser.add_argument("--baseline_reid", help="是否使用baseline的reid", dest="baseline_reid", type=bool, default=True)
        self.parser.add_argument('--use_topology', type=bool, default=True,
                                help='Whether to use topology-based matching strategy')
        self.parser.add_argument('--database_name', type=str, default='AIC19',
                                 help='是否使用AIC22数据集进行测试')

    def parse(self):
        opt = self.parser.parse_args()
        if opt.database_name == 'AIC19':
            # 检查data_dir是否使用默认值，如果是则修改为AIC22数据集路径
            if opt.data_dir == dataset_hst_default:
                opt.data_dir = '/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/dataset/AIC19/validation/S02/'
            opt.output_dir='/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/2. online_MTMC/outputs/result/'
            opt.yolo10_model='/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/2. online_MTMC/preliminary/det_weights/my/best_multiple3.pt'
        elif  opt.database_name == 'HST':
            pass
        else :
            raise ValueError('database_name must be AIC19 or HST')
        return opt

opt = Opts().parse()
