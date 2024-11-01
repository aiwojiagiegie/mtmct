#!/usr/bin/python3
"""
Evaluate submissions for the AI City Challenge.
"""
import os
import sys
import zipfile
import tarfile
import traceback
from marshal import version

import numpy as np
import pandas as pd
import scipy as sp
import motmetrics as mm
import pytrec_eval as trec
from PIL import Image
from collections import defaultdict
from argparse import ArgumentParser
import warnings
import threading
from prettytable import PrettyTable
from termcolor import colored

from tqdm import tqdm

from opts import opt

warnings.filterwarnings("ignore")


def get_args():
    parser = ArgumentParser(add_help=False, usage=usageMsg())
    parser.add_argument("data", nargs=2, help="Path to <test_labels> <predicted_labels>.")
    parser.add_argument('--help', action='help', help='Show this help message and exit')
    parser.add_argument('-m', '--mread', action='store_true', help="Print machine readable results (JSON).")
    parser.add_argument('-ds', '--dstype', type=str, default='train', help="Data set type: train, validation or test.")
    parser.add_argument('-rd', '--roidir', type=str, default='ROIs', help="Region of Interest images directory.")
    return parser.parse_args()


def usageMsg():
    return """  python3 eval.py <ground_truth> <prediction> --dstype <dstype>

Details for expected formats can be found at https://www.aicitychallenge.org/.

See `python3 eval.py --help` for more info.

"""


def getData(fh, fpath, names=None, sep='\s+|\t+|,'):
    """ Get the necessary track data from a file handle.
    
    Params
    ------
    fh : opened handle
        Steam handle to read from.
    fpath : str
        Original path of file reading from.
    names : list<str>
        List of column names for the data.
    sep : str
        Allowed separators regular expression string.
    Returns
    -------
    df : pandas.DataFrame
        Data frame containing the data loaded from the stream with optionally assigned column names.
        No index is set on the data.
    """

    try:
        df = pd.read_csv(
            fpath,
            sep=sep,
            index_col=None,
            skipinitialspace=True,
            header=None,
            names=names,
            engine='python'
        )

        return df

    except Exception as e:
        raise ValueError("Could not read input from %s. Error: %s" % (fpath, repr(e)))


def readData(fpath):
    """ Read test or pred data for a given track. 
    
    Params
    ------
    fpath : str
        Original path of file reading from.
    Returns
    -------
    df : pandas.DataFrame
        Data frame containing the data loaded from the stream with optionally assigned column names.
        No index is set on the data.
    Exceptions
    ----------
        May raise a ValueError exception if file cannot be opened or read.
    """
    names = ['CameraId', 'Id', 'FrameId', 'X', 'Y', 'Width', 'Height', 'Xworld', 'Yworld']

    if not os.path.isfile(fpath):
        raise ValueError("File %s does not exist." % fpath)
    # Gzip tar archive
    if fpath.lower().endswith("tar.gz") or fpath.lower().endswith("tgz"):
        tar = tarfile.open(fpath, "r:gz")
        members = tar.getmembers()
        if len(members) > 1:
            raise ValueError("File %s contains more than one file. A single file is expected." % fpath)
        if not members:
            raise ValueError("Missing files in archive %s." % fpath)
        fh = tar.extractfile(members[0])
        return getData(fh, tar.getnames()[0], names=names)
    # Zip archive
    elif fpath.lower().endswith(".zip"):
        with zipfile.ZipFile(fpath) as z:
            members = z.namelist()
            if len(members) > 1:
                raise ValueError("File %s contains more than one file. A single file is expected." % fpath)
            if not members:
                raise ValueError("Missing files in archive %s." % fpath)
            with z.open(members[0]) as fh:
                return getData(fh, members[0], names=names)
    # text file
    elif fpath.lower().endswith(".txt"):
        with open(fpath, "r") as fh:
            return getData(fh, fpath, names=names)
    else:
        raise ValueError("Invalid file type %s." % fpath)


def print_results(summary, mread=False):
    """Print a summary dataframe in a human- or machine-readable format.
    
    Params
    ------
    summary : pandas.DataFrame
        Data frame of evaluation results in motmetrics format.
    mread : bool
        Whether to print results in machine-readable format (JSON).
    Returns
    -------
    None
        Prints results to screen.
    """
    if mread:
        print('{"results":%s}' % summary.iloc[-1].to_json())
        return
    print(summary)

    formatters = {'idf1': '{:2.2f}'.format,
                  'idp': '{:2.2f}'.format,
                  'idr': '{:2.2f}'.format}

    summary = summary[['idf1', 'idp', 'idr']]
    summary['idp'] *= 100
    summary['idr'] *= 100
    summary['idf1'] *= 100
    print(mm.io.render_summary(summary, formatters=formatters, namemap=mm.io.motchallenge_metric_names))
    return

def compare_dataframes_mtmc(gts, ts,mh):
    """Compute ID-based evaluation metrics for multi-camera multi-object tracking.

    Params
    ------
    gts : pandas.DataFrame
        Ground truth data.
    ts : pandas.DataFrame
        Prediction/test data.
    Returns
    -------
    df : pandas.DataFrame
        Results of the evaluations in a df with only the 'idf1', 'idp', and 'idr' columns.
    """
    gtds = []
    tsds = []
    gtcams = gts['CameraId'].drop_duplicates().tolist()
    tscams = ts['CameraId'].drop_duplicates().tolist()
    maxFrameId = 0

    for k in tqdm(sorted(gtcams)):
        gtd = gts.query('CameraId == %d' % k)
        gtd = gtd[['FrameId', 'Id', 'X', 'Y', 'Width', 'Height']]
        # max FrameId in gtd only
        mfid = gtd['FrameId'].max()
        gtd['FrameId'] += maxFrameId
        gtd = gtd.set_index(['FrameId', 'Id'])
        gtds.append(gtd)

        if k in tscams:
            tsd = ts.query('CameraId == %d' % k)
            tsd = tsd[['FrameId', 'Id', 'X', 'Y', 'Width', 'Height']]
            # max FrameId among both gtd and tsd
            mfid = max(mfid, tsd['FrameId'].max())
            tsd['FrameId'] += maxFrameId
            tsd = tsd.set_index(['FrameId', 'Id'])
            tsds.append(tsd)

        maxFrameId += mfid

    # compute multi-camera tracking evaluation stats
    gtds_concat = pd.concat(gtds)
    tsds_concat = pd.concat(tsds)
    multiCamAcc = mm.utils.compare_to_groundtruth(gtds_concat, tsds_concat, 'iou')
    metrics = list(mm.metrics.motchallenge_metrics)
    metrics.extend(['num_frames', 'idfp', 'idfn', 'idtp'])
    summary = mh.compute(multiCamAcc, metrics=metrics, name='MultiCam')

    return summary
def removeRepetition(df):
    """Remove repetition to ensure that all objects are unique for every frame.

    Params
    ------
    df : pandas.DataFrame
        Data that should be filtered
    Returns
    -------
    df : pandas.DataFrame
        Filtered data that all objects are unique for every frame.
    """

    df = df.drop_duplicates(subset=['CameraId', 'Id', 'FrameId'], keep='first')

    return df
def removeOutliersSingleCam(df):
    """Remove outlier objects that appear in a single camera.

    Params
    ------
    df : pandas.DataFrame
        Data that should be filtered
    Returns
    -------
    df : pandas.DataFrame
        Filtered data with only objects that appear in 2 or more cameras.
    """
    # get unique CameraId/Id combinations, then count by Id
    cnt = df[['CameraId', 'Id']]
    cnt = cnt.drop_duplicates()[['Id']].groupby(['Id'])
    cnt = cnt.size()  # 这里获取到了哪些车辆id出现在摄像头的次数
    # keep only those Ids with a camera count > 1
    keep = cnt[cnt > 1]
    no_keep = cnt[cnt <= 1]
    # retrict the data to kept ids
    return df.loc[df['Id'].isin(keep.index)]
def removeOutliersROI(df, dstype='train', roidir='ROIs', cid=None):
    """ Remove outliers from the submitted test df that are outsize the region of interest for each camera.

    Params
    ------
    df : pandas.dfFrame
        df that should be filtered.
    dstype : str
        Data set type. One of 'train', 'validation' or 'test'. Defaults to 'train'.
    roidir : str
        Directory containing the ROI images. Images are stores in sub-directories <dstype>/c<camid%03d>/roi.jpg,
        where dstype is the dataset type, and camid is the camera number as a 3-digit 0-padded int.
        If the ROI data cannot be found, it will be downloaded and stored locally in the <roidir> directory
        relative to the execution of the eval script. Defaults to 'ROIs'.
    cid : int
        Optional camera ID for which to filter data. Defaults to None.
    Returns
    -------
    df : pandas.dfFrame
        Filtered df with only objects within the ROI retained.
    """

    def loadroi(cid):
        """Read the ROI image for a given camera.

        Params
        ------
        cid : int
            Camera ID whose ROI image should be retrieved.
        Returns
        -------
        im : numpy.ndarray
            Image stored as a 2-d ndarray.
        """

        imf = os.path.join(roidir, '%02d' % cid, '%02d.png' % cid)
        if not os.path.exists(imf):
            raise ValueError(f"缺少摄像头 %02d 的ROI图像。路径为{imf}" % cid)
        img = Image.open(imf, mode='r').convert('L')  # 转换为灰度图像
        img.load()
        if img.size[0] > img.size[1]:
            img = img.transpose(Image.TRANSPOSE)

        im = np.asarray(img, dtype="uint8")
        if im.shape[0] > im.shape[1]:
            im = im.T

        return im

    def isInROI(row, roi, height, width, threshold=0.2):
        """检查目标是否主要在ROI内。

        参数:
        row : 包含目标位置和大小信息的数据行
        roi : ROI图像
        height, width : ROI图像的高度和宽度
        threshold : 目标需要在ROI内的最小比例,默认为0.5

        返回:
        bool : 如果目标主要在ROI内则返回True,否则返回False
        """
        xmin = max(0, int(row['X']))
        ymin = max(0, int(row['Y']))
        xmax = min(width, int(row['X'] + row['Width']))
        ymax = min(height, int(row['Y'] + row['Height']))

        # 计算边界框内的像素总数和ROI内的像素数
        total_pixels = (xmax - xmin) * (ymax - ymin)
        roi_pixels = np.sum(roi[ymin:ymax, xmin:xmax] == 255)

        # 如果ROI内的像素比例大于阈值,则认为目标主要在ROI内
        return roi_pixels / total_pixels > threshold

    # Fetch the ROI data if necessary
    if not os.path.isdir(roidir):
        import zipfile
        import urllib.request
        import shutil
        import tempfile
        os.makedirs(roidir)
        url = 'https://drive.google.com/uc?export=download&id=1sHQqtzNaUJu1r3AJ8X0sODfe9TIu4C1M'
        # Download the file from `url` and save it locally under `file_name`:
        tmp_fname = next(tempfile._get_candidate_names())
        tmp_dir = tempfile._get_default_tempdir()
        fzip = os.path.join(tmp_dir, tmp_fname)
        with urllib.request.urlopen(url) as response, open(fzip, 'wb') as ofh:
            shutil.copyfileobj(response, ofh)
        zip_ref = zipfile.ZipFile(fzip, 'r')
        zip_ref.extractall(roidir)
        zip_ref.close()
        os.remove(fzip)

    # Store which rows are not ROI outliers
    df['InROI'] = True

    if cid is None:  # Process all cameras
        # Make sure df is sorted appropriately
        df.sort_values(['CameraId', 'FrameId'], inplace=True)
        # Load first ROI image
        tscams = df['CameraId'].unique()
        cid = tscams[0]
        roi = loadroi(cid)
        height, width = roi.shape
        # Loop over objects and check for outliers
        for i, row in df.iterrows():
            if row['CameraId'] != cid:
                cid = row['CameraId']
                roi = loadroi(cid)
                height, width = roi.shape
            df.at[i, 'InROI'] = isInROI(row, roi, height, width)

        return df[df['InROI']].drop(columns=['InROI'])

    df = df[df['CameraId'] == cid].copy()
    # Make sure df is sorted appropriately
    df.sort_values(['CameraId', 'FrameId'], inplace=True)
    # Load ROI image
    roi = loadroi(cid)
    height, width = roi.shape
    # Loop over objects and check for outliers
    for i, row in df.iterrows():
        df.at[i, 'InROI'] = isInROI(row, roi, height, width)

    return df[df['InROI']].drop(columns=['InROI'])
def eval(test, pred, **kwargs):
    """ Evaluate submission.

    Params
    ------
    test : pandas.DataFrame
        Labeled data for the test set. Minimum columns that should be present in the 
        data frame include ['CameraId','Id', 'FrameId', 'X', 'Y', 'Width', 'Height'].
    pred : pandas.DataFrame
        Predictions for the same frames as in the test data.
    Kwargs
    ------
    mread : bool
        Whether printed result should be machine readable (JSON). Defaults to False.
    dstype : str
        Data set type. One of 'train', 'validation' or 'test'. Defaults to 'train'.
    roidir : str
        Directory containing ROI images or where they should be stored.
    Returns
    -------
    df : pandas.DataFrame
        Results from the evaluation
    """
    if test is None:
        return None
    mread = kwargs.pop('mread', False)
    dstype = kwargs.pop('dstype', 'train')
    roidir = kwargs.pop('roidir', 'ROIs')

    mh = mm.metrics.create()

    # filter prediction data
    # 根据ROI进行过滤
    pred = removeOutliersROI(pred, dstype=dstype, roidir=roidir)
    # test = removeOutliersROI(test, dstype=dstype, roidir=roidir)
    # 过滤掉只出现在一个摄像头中的车辆
    pred = removeOutliersSingleCam(pred)
    # 过滤掉重复出现的数据 根据'CameraId', 'Id', 'FrameId'判断是否重复
    pred = removeRepetition(pred)

    # evaluate results
    return compare_dataframes_mtmc(test, pred,mh)


def usage(msg=None):
    """ Print usage information, including an optional message, and exit. """
    if msg:
        print("%s\n" % msg)
    print("\nUsage: %s" % usageMsg())
    exit()
info = {
    "idf1": "评估跟踪器和基准真实数据之间的一致性",
    "mota": "表示整体跟踪准确率。",
    "idp": "正确识别的目标比例。",
    "idr": "从所有真实目标中，被正确识别的比例。",
    "recall": "召回率",
    "precision": "正确识别的目标比例。",
    "num_unique_objects": "总共需要被跟踪的不同目标的数量。",
    "mostly_tracked": "被跟踪时间超过 80% 的目标数。",
    "partially_tracked": "被跟踪时间在 20% 到 80% 之间的目标数。",
    "mostly_lost": "被跟踪时间少于 20% 的目标数。",
    "num_false_positives": "错误识别的目标数。",
    "num_misses": "未被跟踪到的真实目标数。",
    "num_switches": "目标身份在跟踪过程中错误切换的次数。",
    "num_fragmentations": "目标跟踪过程中断然后重新开始的次数。",
    "motp": "表示跟踪的位置精度。",
    "num_frames": "数据中的帧数。",
    "idfp": "错误跟踪的目标数。",
    "idfn": "被错误丢失的目标数。",
    "idtp": "被正确跟踪的目标数。",
    "num_transfer": "目标在不同摄像机视角间转移的次数。",
    "num_ascend": "目标在场景中升级或升高的次数。",
    "num_migrate": "目标从一个区域迁移到另一个区域的次数。"
}


def eval_wrapper(test, pred, mread, dstype, roidir, result):
    result.append(eval(test, pred, mread=mread, dstype=dstype, roidir=roidir))

def calculate_results(test_path, pred_path, mread=False, dstype='validation', roidir='/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/dataset/HST/real'):
    test = readData(test_path)
    pred = readData(pred_path)
    try:
        pred_results = []
        eval_wrapper(test, pred, mread, dstype, roidir, pred_results)
        my_print_result(pred_results[0])
    except Exception as e:
        if mread:
            print('{"error": "%s"}' % repr(e))
        else:
            print("Error: %s" % repr(e))
        traceback.print_exc()

def my_print_result(pred_summary):
    pred_dict = pred_summary.to_dict(orient='records')[0]
    
    # 创建表格
    table = PrettyTable()
    table.field_names = ["指标", "预测结果", "指标解释"]
    
    # 定义要优先显示的指标
    priority_metrics = ['idf1', 'idfp', 'idfn', 'idtp', 'mota']
    
    # 先添加优先指标
    for key in priority_metrics:
        pred_value = pred_dict[key]
        # 格式化数值
        if isinstance(pred_value, float) and not pred_value.is_integer():
            formatted_pred = f"{pred_value:.6%}"
        else:
            formatted_pred = str(pred_value)
        table.add_row([key, formatted_pred, info[key]])
    
    # 添加其他指标
    for key in pred_dict.keys():
        if key not in priority_metrics:
            pred_value = pred_dict[key]
            # 格式化数值
            if isinstance(pred_value, float) and not pred_value.is_integer():
                formatted_pred = f"{pred_value:.6%}"
            else:
                formatted_pred = str(pred_value)
            table.add_row([key, formatted_pred, info[key]])
    
    # 设置表格样式
    table.align = "l"
    table.max_width = 120
    
    # 打印表格
    print(table)

if __name__ == '__main__':
    version=opt.version
    version = '14'
    pred_path = f'result/version/v{version}.txt'
    gt_path = './test_gt.txt'
    
    calculate_results(gt_path, pred_path)





