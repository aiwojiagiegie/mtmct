from random import random

from yolov10.ultralytics.models import YOLOv10

model_yaml_path = "./yolov10/ultralytics/cfg/models/v10/yolov10n.yaml"
data_yaml_path = './yolov10/datasets2/multi_class/data.yaml'

if __name__ == '__main__':
    pre_model_name = f'/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/2. online_MTMC/preliminary/det_weights/my/best_multiple3.pt'
    # 加载预训练模型
    # model = YOLOv10(model_yaml_path).load(pre_model_name)
    model = YOLOv10(model_yaml_path)
    pre_model_name_last = pre_model_name.split('/')[-1]
    # 训练模型
    epochs = 200
    batch = 32
    # 生成一个六位的随机数
    save_path = f'UA-DETRAC_pre/model_name_{pre_model_name_last}/epochs_{epochs}/batch_{batch}/'
    results = model.train(data=data_yaml_path,
                          epochs=epochs,
                          batch=batch,
                          name=f"{save_path}", device='1')