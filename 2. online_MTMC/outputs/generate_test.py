import pandas as pd


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

if __name__ == '__main__':
    fpath = 'ground_truth_validation.txt'
    names = ['CameraId', 'Id', 'FrameId', 'X', 'Y', 'Width', 'Height', 'Xworld', 'Yworld']
    with open(fpath, "r") as fh:
        data = getData(fh, fpath, names=names)
    # 筛选出Id为96的数据
    data = data[data['Id'] == 97]
    # 写入到某个文件
    data.to_csv('test_gt.txt', index=False,header=None)

