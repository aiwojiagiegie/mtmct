import argparse
import torch
from torchvision import transforms
from reid.data.triplet_sampler import *
import torch.multiprocessing
import yaml
from reid.models.models import MBR_model



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic= True
    torch.backends.cudnn.benchmark= False
class ReId(object):
    def __init__(self,path_weights):
        with open(path_weights + "config.yaml", "r") as stream:
            data = yaml.safe_load(stream)
        if data['half_precision']:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = False
        model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="LBS", n_groups=4, LAI=data['LAI'],
                          n_cams=data['n_cams'], n_views=data['n_views'])
        path_weights = path_weights + 'best_mAP.pt'
        model.load_state_dict(torch.load(path_weights, map_location='cpu'))
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f'Selected device: {device}')
        self.model = model.to(device)
        self.model.eval()
    def reid(self,image):
        if self.scaler:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, _, ffs, activations = self.model(image, None, None)
        else:
            _, _, ffs, activations = self.model(image, None, None)
        return torch.cat(ffs[:], dim=1)

if __name__ == "__main__":
    reid = ReId('./reid/logs/Veri776/MBR_4G/0/')

    # ### Just to ensure VehicleID 10-fold validation randomness is not random to compare different models training
    # set_seed(0)
    # parser = argparse.ArgumentParser(description='Reid train')
    #
    # parser.add_argument('--batch_size', default=None, type=int, help='an integer for the accumulator')
    # parser.add_argument('--dataset', default=None, help='Choose one of[Veri776, VERIWILD]')
    # parser.add_argument('--model_arch', default=None, help='Model Architecture')
    # parser.add_argument('--path_weights', default='./reid/logs/Veri776/MBR_4G/0/', help="Path to *.pth/*.pt loading weights file")
    # parser.add_argument('--re_rank', action="store_true", help="Re-Rank")
    # args = parser.parse_args()
    #
    # with open(args.path_weights + "config.yaml", "r") as stream:
    #     data = yaml.safe_load(stream)
    #
    # data['BATCH_SIZE'] = args.batch_size or data['BATCH_SIZE']
    # data['dataset'] = args.dataset or data['dataset']
    # data['model_arch'] = args.model_arch or data['model_arch']
    #
    #
    # teste_transform = transforms.Compose([
    #                 transforms.Resize((data['y_length'],data['x_length']), antialias=True),
    #                 transforms.Normalize(data['n_mean'], data['n_std']),
    #
    # ])
    #
    # if data['half_precision']:
    #     scaler = torch.cuda.amp.GradScaler()
    # else:
    #     scaler=False
    # model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="LBS", n_groups=4, LAI=data['LAI'],
    #                   n_cams=data['n_cams'], n_views=data['n_views'])
    # path_weights = args.path_weights + 'best_mAP.pt'
    # model.load_state_dict(torch.load(path_weights, map_location='cpu'))
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # print(f'Selected device: {device}')
    # model = model.to(device)
    # model.eval()
    # path = '/data2/zhangkun/reid/VeRi/image_test/'



