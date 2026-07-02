# Fast_Online_MTMCT

面向车辆的**快速在线多目标多摄像头跟踪(MTMCT)**。本仓库在论文
*"Fast online multi-target multi-camera tracking for vehicles", Applied Intelligence, 2023*
([论文链接](https://link.springer.com/article/10.1007/s10489-023-05081-7)) 官方代码的基础上进行了扩展与改进:

- 检测器由 YOLOv7 迁移至 **YOLOv10**;
- 新增三个可插拔的跨摄像头匹配增强模块:**STCR**、**LPMT**、**RSAM**(见下文),可通过命令行开关独立开启/关闭以做消融;
- 兼容 **NumPy 2.0**,并优化了检测输出流程。

---

## 核心模块

跨摄像头关联在保持"分层拓扑门控 + 阈值门控匈牙利匹配"整体框架的同时,引入以下三个模块(默认全部开启,关闭后行为与原始版本一致):

| 模块 | 开关 | 作用 |
| --- | --- | --- |
| **STCR** | `--use_stcr` | 单摄像头内静止车辆轨迹恢复(在 `tracking/bot_sort.py` 中实现) |
| **LPMT** | `--use_lpmt` | 跨摄像头车道可达性剪枝,按拓扑/车道约束缩小匹配可行域 |
| **RSAM** | `--use_rsam` | 跨摄像头入口/中段/出口区域门控,过滤不合理的跨镜配对 |

> 关于分层全局匹配与自适应特征记忆的详细设计,参见
> [`2. online_MTMC/docs/`](2.%20online_MTMC/docs/) 下的说明文档。

---

## 环境

本项目在本地 conda 环境 **`mtmct_gpu`** 下开发与测试,主要依赖版本如下:

| 依赖 | 版本 |
| --- | --- |
| Python | 3.10.18 |
| PyTorch | 2.5.1 (cu121) |
| torchvision | 0.20.1 (cu121) |
| CUDA / cuDNN | 12.1 / 9.1 |
| NumPy | 2.1.2 |
| OpenCV | 4.11.0 |
| SciPy | 1.15.3 |
| scikit-learn | 1.7.0 |
| ultralytics (YOLOv10) | 8.3.160 |
| motmetrics | 1.4.0 |

> 推理使用半精度 `half()`,需支持 CUDA 的 GPU。YOLOv10 检测器位于 `2. online_MTMC/yolov10`。
> 建议直接创建同名环境:`conda create -n mtmct_gpu python=3.10`,再按上表安装依赖。

---

## 目录结构

```
Fast_Online_MTMCT
├── 1. train_feat_ext        # 训练 ReID 特征提取器
│   ├── gen_aic_veri_dataset.py
│   └── train.py
├── 2. online_MTMC           # 在线 MTMCT 主流程
│   ├── run_mtmc.py          # 主入口
│   ├── opts.py              # 所有超参数与模块开关
│   ├── tracking/bot_sort.py # 单摄像头跟踪(含 STCR)
│   ├── preliminary/         # 检测/特征权重、RoI、重叠区域掩膜
│   ├── yolov10/             # YOLOv10 检测器
│   └── outputs/             # 结果输出
└── dataset                  # 数据集(需自行准备)
    ├── AIC19
    └── VeRi-776
```

---

## 数据准备

将数据集按如下方式放置于仓库同级的 `dataset` 目录:

```
- dataset
  - AIC19        # CityFlow / AI City Challenge 2019
  - VeRi-776     # 用于 ReID 特征训练
```

在线跟踪默认读取 `../dataset/AIC19/validation/S02`(可用 `--data_dir` 修改)。

---

## 权重准备(Model Zoo)

**1. 检测权重**
将 YOLOv10 检测权重放到 `2. online_MTMC/preliminary/det_weights/`
(默认使用 `preliminary/det_weights/my/best_multiple3.pt`,可用 `--det_weights` / `--det_name` 修改)。

**2. 特征提取(ReID)权重**
将 AIC19 + VeRi-776 训练得到的权重放到 `2. online_MTMC/preliminary/feat_ext_weights/`:

- [resnet50_ibn_a_gap_120.t7](https://drive.google.com/file/d/1ZQspaimt2WfyXAeX6C1tSgAPtcBDfv0w/view?usp=sharing)
- [resnet50_ibn_a_gem_120.t7](https://drive.google.com/file/d/1A2ib3FNSFoaFdvbcSWay6JYD5AOHn8w0/view?usp=sharing)
- [resnet101_ibn_a_gap_120.t7](https://drive.google.com/file/d/1ZQ2SCrJEszhWsfUCmV8Jh1lv2apZctUG/view?usp=sharing)
- [resnet101_ibn_a_gem_120.t7](https://drive.google.com/file/d/1iQe4n0SiiPwF8z7HXyMpPuqH7-3aaoeO/view?usp=sharing)

---

## 训练特征提取器(可选)

如需自行训练 ReID 特征提取器,进入 `1. train_feat_ext/`:

1. 将 ImageNet 预训练权重放到 `./net`(来自 [IBN-Net](https://github.com/XingangPan/IBN-Net)):
   [resnet50_ibn_a.pth](https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth) /
   [resnet101_ibn_a.pth](https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth)
2. 运行 `gen_aic_veri_dataset.py` 生成 AIC19 + VeRi-776 训练集
3. 运行 `train.py`,权重保存在 `./outputs`

---

## 运行在线 MTMCT

进入 `2. online_MTMC/`:

1. 在 `opts.py` 中调整参数(数据路径、权重、匹配阈值等)
2. 运行主入口:
   ```bash
   python run_mtmc.py
   ```
3. 跟踪结果保存在 `./outputs/result/` 下

### 消融开关

三个增强模块均可通过命令行独立控制(取值 `true`/`false`),便于消融实验:

```bash
# 关闭 LPMT,仅评估其余模块
python run_mtmc.py --use_lpmt false

# 只保留基线(三个模块全部关闭)
python run_mtmc.py --use_stcr false --use_lpmt false --use_rsam false
```




