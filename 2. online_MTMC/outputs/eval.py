#!/usr/bin/python3
"""
Evaluate submissions for the AI City Challenge.
"""
import os
import sys
import zipfile
import tarfile
import traceback
import numpy as np
import pandas as pd
import requests
import scipy as sp
import motmetrics as mm
import pytrec_eval as trec
from PIL import Image
from collections import defaultdict
from argparse import ArgumentParser
import warnings

from tqdm import tqdm

from opts import opt
from utils.notify import send_feishu_message

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
    maxFrameId = 0;

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

        imf = os.path.join(roidir, 'c%03d' % cid, 'roi.jpg')
        if not os.path.exists(imf):
            raise ValueError("Missing ROI image for camera %03d." % cid)
        img = Image.open(imf, mode='r')
        img.load()
        if img.size[0] > img.size[1]:
            img = img.transpose(Image.TRANSPOSE)

        im = np.asarray(img, dtype="uint8")
        if im.shape[0] > im.shape[1]:
            im = im.T

        return im

    def isROIOutlier(row, roi, height, width):
        """Check whether item stored in row is outside the region of interest.

        Params
        ------
        row : pandas.Series
            Row of data including, at minimum, the 'X', 'Y', 'Width', and 'Height' columns.
        roi : numpy.ndarray
            ROI image for the camera with the same id as in row['CameraId'].
        height : int
            ROI image height.
        width : int
            ROI image width.
        Returns
        -------
        bool
            Return True if image is an outlier.
        """
        xmin = int(row['X'])
        ymin = int(row['Y'])
        xmax = int(row['X'] + row['Width'])
        ymax = int(row['Y'] + row['Height'])

        if xmin >= 0 and xmin < width:
            if ymin >= 0 and ymin < height and roi[ymin, xmin] < 255:
                return True
            if ymax >= 0 and ymax < height and roi[ymax, xmin] < 255:
                return True
        if xmax >= 0 and xmax < width:
            if ymin >= 0 and ymin < height and roi[ymin, xmax] < 255:
                return True
            if ymax >= 0 and ymax < height and roi[ymax, xmax] < 255:
                return True
        return False

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
    df['NotOutlier'] = True

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
            if isROIOutlier(row, roi, height, width):
                df.at[i, 'NotOutlier'] = False

        return df[df['NotOutlier']].drop(columns=['NotOutlier'])

    df = df[df['CameraId'] == cid].copy()
    # Make sure df is sorted appropriately
    df.sort_values(['CameraId', 'FrameId'], inplace=True)
    # Load ROI image
    roi = loadroi(cid)
    height, width = roi.shape
    # Loop over objects and check for outliers
    for i, row in df.iterrows():
        if isROIOutlier(row, roi, height, width):
            df.at[i, 'NotOutlier'] = False

    return df[df['NotOutlier']].drop(columns=['NotOutlier'])
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
    "mota": "表示整体跟踪准确率。",
    "motp": "表示跟踪的位置精度。",
    "num_frames": "数据中的帧数。",
    "idfp": "错误跟踪的目标数。",
    "idfn": "被错误丢失的目标数。",
    "idtp": "被正确跟踪的目标数。",
    "num_transfer": "目标在不同摄像机视角间转移的次数。",
    "num_ascend": "目标在场景中升级或升高的次数。",
    "num_migrate": "目标从一个区域迁移到另一个区域的次数。"
}
baseline = {'idf1': 0.7825229312555478, 'idp': 0.8095918367346939, 'idr': 0.7572055735827448, 'recall': 0.8324107654132468, 'precision': 0.89, 'num_unique_objects': 145, 'mostly_tracked': 105, 'partially_tracked': 35, 'mostly_lost': 5, 'num_false_positives': 2156, 'num_misses': 3512, 'num_switches': 43, 'num_fragmentations': 240, 'mota': 0.7274766176751288, 'motp': 0.2514508515359548, 'num_transfer': 46, 'num_ascend': 10, 'num_migrate': 18, 'num_frames': 7494, 'idfp': 3732.0, 'idfn': 5088.0, 'idtp': 15868.0}


def calculate_results(test_path, pred_path, mread=False, dstype='validation', roidir='/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/dataset/AIC19/validation/S02'):
    test = readData(test_path)
    pred = readData(pred_path)
    try:
        summary = eval(test, pred, mread=mread, dstype=dstype, roidir=roidir)
        # 将DataFrame转换为字典列表
        return my_print_result(summary)
        # print_results(summary, mread=mread)
    except Exception as e:
        if mread:
            print('{"error": "%s"}' % repr(e))
        else:
            print("Error: %s" % repr(e))
        traceback.print_exc()


def my_print_result(summary):
    dict_list = summary.to_dict(orient='records')
    format_str = "{:<20} {:<15} {:<12} {:10} {:4} {}"
    float_format = "{:<20} {:<15.6%} {:<12.6%} {:10.6%} {:4} {}"
    print(format_str.format("指标", "baseline", "当前结果", "差值", "好坏", "指标解释"))
    
    # 用于返回的关键指标
    key_metrics = {}
    
    for record in dict_list:
        for key, value in record.items():
            # 计算差值和符号
            difference = value - baseline[key]
            symbol = "↑" if difference > 0 else "↓" if difference < 0 else ""
            
            # 收集关键指标
            if key in ['idf1', 'mota', 'idp', 'idr']:
                key_metrics[key] = value
                
            if isinstance(value, float):
                if value.is_integer():
                    print(format_str.format(key, int(baseline[key]), int(value), difference, symbol, info[key]))
                else:
                    print(float_format.format(key, baseline[key], value, difference, symbol, info[key]))
            else:
                print(format_str.format(key, baseline[key], value, difference,symbol, info[key]))
    return key_metrics

if __name__ == '__main__':
    version = opt.version
    detection_path = f'./result/version/v{version}.txt'
    gt_path = './ground_truth_validation.txt'
    # calculate_results('ground_truth_validation.txt', f'ground_truth_validation.txt')
    ans = calculate_results(gt_path, detection_path)
    # 格式化消息内容
    message = "MTMCT评估结果:\n"
    message += f"IDF1(跨摄像头跟踪准确率): {ans['idf1']:.2%}\n"
    message += f"MOTA(整体跟踪准确率): {ans['mota']:.2%}\n"
    message += f"IDP(识别精确度): {ans['idp']:.2%}\n"
    message += f"IDR(识别召回率): {ans['idr']:.2%}\n"
    send_feishu_message(message)
    # calculate_results('test_gt.txt','test_pred.txt')
    # args = get_args();
    # if not args.data or len(args.data) < 2:
    #     usage("Incorrect number of arguments. Must provide paths for the test (ground truth) and predicitons.")
    #
    # test = readData(args.data[0])
    # pred = readData(args.data[1])
    # try:
    #     summary = eval(test, pred, mread=args.mread, dstype=args.dstype, roidir=args.roidir)
    #     print_results(summary, mread=args.mread)
    # except Exception as e:
    #     if args.mread:
    #         print('{"error": "%s"}' % repr(e))
    #     else:
    #         print("Error: %s" % repr(e))
    #     traceback.print_exc()
