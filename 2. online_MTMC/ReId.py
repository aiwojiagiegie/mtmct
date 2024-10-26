import argparse
import cv2
import torch
import torch.nn.functional as F
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
    def reid_multiple(self, images):
        features = []
        for image in images:
            if self.scaler:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, _, ffs, _ = self.model(image, None, None)
            else:
                _, _, ffs, _ = self.model(image, None, None)
            features.append(torch.cat(ffs[:], dim=1))
        return features
def load_model(path_weights):
    with open(path_weights + "config.yaml", "r") as stream:
        data = yaml.safe_load(stream)
    model = MBR_model(class_num=data['n_classes'], n_branches=[], losses="LBS", n_groups=4, LAI=data['LAI'],
                      n_cams=data['n_cams'], n_views=data['n_views'])
    model.load_state_dict(torch.load(path_weights + 'best_mAP.pt', map_location='cpu'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval()

def process_image(model, image_path, crop_params):
    img = cv2.imread(image_path)
    x, y, w, h = crop_params
    img = img[y:y+h, x:x+w]
    img_tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().cuda()
    with torch.no_grad():
        _, _, ffs, _ = model(img_tensor, None, None)
    feat = torch.cat(ffs[:], dim=1)
    return feat / torch.norm(feat, dim=1, keepdim=True)

def compute_cosine_distance(feat1, feat2):
    similarity = F.cosine_similarity(feat1, feat2, dim=1)
    return (1 - similarity).item()
if __name__ == "__main__":
    model_path = './reid/logs/Veri776/MBR_4G/0/'
    reid_model = ReId(model_path)

    # 准备三张图片
    img_paths = [
        '/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/dataset/HST/real/41/frame/41_f0113.jpg',
        '/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/dataset/HST/real/43/frame/43_f0323.jpg',
        '/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/dataset/HST/real/43/frame/43_f1008.jpg'
    ]
    crop_params = [
        (772, 512, 337, 266),
        (1129, 217, 125, 105),
        (1357, 86, 54, 42)
    ]

    # 处理图片
    processed_images = []
    for img_path, crop_param in zip(img_paths, crop_params):
        img = cv2.imread(img_path)
        x, y, w, h = crop_param
        img = img[y:y+h, x:x+w]
        img_tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().cuda()
        processed_images.append(img_tensor)

    # 一次性进行ReId
    features = reid_model.reid_multiple(processed_images)

    # 计算余弦距离
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            distance = compute_cosine_distance(features[i], features[j])
            print(f"图片{i+1}和图片{j+1}之间的余弦距离: {distance}")
