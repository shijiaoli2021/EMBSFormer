import numpy as np
import pandas as pd
from datetime import datetime
import torch
import holidays as hy
import pickle

startDict = {'PEMS04': '20180101', 'PEMS07': '20170501', 'PEMS08': '20160701', 'METR-LA':'20120301'}
endDict = {'PEMS04': '20180228', 'PEMS07': '20170831', 'PEMS08': '20160831', 'METR-LA':'20120630'}
MinuteSize = 1440
DaySize = 7


'''PEM: distance_matrix'''
def get_distance_matirx_PEM(load_path, nodes_num):
    adj_data = np.array(pd.read_csv(load_path))
    distance_adj = np.zeros((nodes_num, nodes_num))
    for i in range(adj_data.shape[0]):
        distance_adj[int(adj_data[i, 0]), int(adj_data[i, 1])] = adj_data[i, 2]
        distance_adj[int(adj_data[i, 1]), int(adj_data[i, 0])] = adj_data[i, 2]
    return distance_adj

def normalization(train, val, test):
    '''
    :param train: np
    :param val: np
    :param test : np
    :return: stats, train_norm, val_norm, test_norm 
    '''
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    train_norm = (train-mean)/std
    test_norm = (test-mean)/std
    val_norm = (val-mean)/std
    stats = {"mean": mean, "std": std}
    return stats, train_norm, val_norm, test_norm

def normalization_cycle(train, val, test, feature_dim):
    mean = []
    std = []
    for i in range(train.shape[0]):
        cycle_stats_cash, train[i, :, :, :feature_dim, :], val[i, :, :, :feature_dim, :], test[i, :, :, :feature_dim, :] \
        = normalization(train[i, :, :, :feature_dim, :], val[i, :, :, :feature_dim, :],
                        test[i, :, :, :feature_dim, :])
        mean.append(cycle_stats_cash["mean"])
        std.append(cycle_stats_cash["std"])
    mean = np.concatenate([np.expand_dims(mean_cash, axis=0) for mean_cash in mean], axis=0)
    std = np.concatenate([np.expand_dims(std_cash, axis=0) for std_cash in std], axis=0)
    stats = {"mean": mean, "std": std}
    return stats


def datelist(beginDate, endDate):
    # beginDate, endDate是形如‘20160601’的字符串或datetime格式
    date_l=[datetime.strftime(x,'%Y-%m-%d') for x in list(pd.date_range(start=beginDate, end=endDate))]
    return date_l
# datalist = datelist(startDict['pem04'],endDict['pem04'])
# print(datetime.strptime(datalist[1], '%Y-%M-%d').weekday()+1)


'''
获取每天平均温度，风速信息，温度：HourlyDryBulbTemperature   风速：HourlyWindSpeed
'''
def read_tem_data(path, data_name, dtype=np.float32):
    dates = datelist(startDict[data_name], endDict[data_name])
    data = pd.read_csv(path)
    len, _ = data.shape
    all = []
    for date in dates:
        day = []
        tem = 0
        speed = 0
        sum = 0
        for i in range(len):
            rows = data.iloc[i]
            if rows['DATE'][: 10] == date:
                tem_data = str(rows['HourlyDryBulbTemperature']).replace(" ", "")
                wind_speed = str(rows['HourlyWindSpeed']).replace(" ", "")
                if tem_data == "nan" or wind_speed == "nan" or tem_data == "*" or wind_speed == "*":
                    continue
                tem += float(str(rows['HourlyDryBulbTemperature']).strip('s*'))
                speed += float(str(rows['HourlyWindSpeed']).strip('s*'))
                sum += 1
        if sum != 0:
            # print("温度和：{}，风速和：{}, sum:{}".format(tem, speed, sum))
            day.append(tem/sum)
            day.append(speed/sum)
            all.append(day)
    return np.array(all, dtype=dtype)

def load_features(feat_path, dtype=np.float32):
    feat_df = pd.read_csv(feat_path)
    feat = np.array(feat_df, dtype=dtype)
    feat = np.expand_dims(feat, axis=2)
    return feat

'''PEM : adj_matrix'''
def get_adj_matrix_PEM(load_path, nodes_num):
    adj_data = np.array(pd.read_csv(load_path))
    adj = np.zeros((nodes_num,nodes_num))
    for i in range(adj_data.shape[0]):
        adj[int(adj_data[i, 0]), int(adj_data[i, 1])] = 1
        adj[int(adj_data[i,1]), int(adj_data[i,0])] = 1
    return adj

'''METR-LA: adj_matrix'''
def get_adj_matrix_METR(load_path, nodes_num):
    with open(load_path, 'rb') as f:
        # print(np.array(pickle.load(f)[2]))
        adj_data = np.array(pickle.load(f)[2])
        adj = np.zeros((nodes_num, nodes_num))
        for i in range(nodes_num):
            for j in range(nodes_num):
                if adj_data[i, j] != 0 and i!=j:
                    adj[i, j] = 1
                else:
                    adj[i, j] = 0
        f.close()
        return adj

'''los or sz : adj_matrix'''
def get_adj_matrix_tgcn(load_path, data_name="sz", dtype= np.float32):
    adj_df = pd.read_csv(load_path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    return adj


'''D matrix'''
def get_d_matrix(adj_matrix):
    d_matrix = np.zeros((adj_matrix.shape[0], adj_matrix.shape[1]))
    for i in range(adj_matrix.shape[0]):
        d_matrix[i, i] = np.sum(adj_matrix[i, :])
    return d_matrix

'''Laplace matrix'''
def get_laplace_matrix(adj, d):
    d_turn = np.zeros((adj.shape[0], adj.shape[0]))
    for i in range(adj.shape[0]):
        if d_turn[i, i] == 0:
            continue
        d_turn[i, i] = np.power(d[i, i], -0.5)
    res = np.eye(adj.shape[0]) + d_turn @ adj @ d_turn
    return res


'''
get the max rou of matrix
'''
def getRou(x):
    lambdas, features = np.linalg.eig(x)
    return np.max(np.abs(lambdas))

'''
get scaled laplacian for chebyshev ploynomials
'''
def scaled_laplacian(L):
    res = (2*L)/getRou(L)-np.identity(L.shape[0])
    return res

'''
get chebyshev ploynomials 
'''
def chebyshev_ploynomials(scaled_laplacian, K):
    '''
    :param scaled_laplacian: [-1，1]的拉普拉斯矩阵
    :param K: 多项式项数
    :return: list
    '''
    res = [np.identity(scaled_laplacian.shape[0]), scaled_laplacian]
    if K>=2:
        for i in range(2, K+1):
            res.append(2*scaled_laplacian*res[i-1]-res[i-2])
    return res


'''get holiday'''
def get_holiday_data(dataset_name):
    us_holidays = hy.UnitedStates()
    dates = datelist(startDict[dataset_name], endDict[dataset_name])
    holidays = []
    for i in dates:
        is_holiday = i in us_holidays
        # weekday = datetime.strptime(i, "%Y-%M-%d").weekday()
        if is_holiday:
            holidays.append(1)
        else:
            holidays.append(0)
    return np.array(holidays)

'''
fix data
'''
def handle_data_for_fix(data):
    # print(data.shape)
    time_len, node_num = data.shape
    data_mean_for_day = np.zeros((24*12, node_num))
    for i in range(24*12):
        data_cash = data[i:time_len:288,:]
        exist = (data_cash!=0)
        data_cash_mean = data_cash.sum(axis = 0) / exist.sum(axis =0)
        data_mean_for_day[i, :] = data_cash_mean
    for i in range(node_num):
        zero_index = np.array(np.where(data[:, i] == 0))
        if (len(zero_index)) == 0:
            continue
        data[zero_index, i] = data_mean_for_day[zero_index % (24*12), i]
    return data
    
    

'''dataPre for PEM'''
def dataPre(load_path="../../data/PEMS/PEMS08/PEMS08.npz",
            tem_speed_path = "../../data/PEMS/PEMS08/TEMP_PEMS08",
            minute_offset = 0,
            add_time_in_day=True,
            add_day_in_week=True,
            add_tem_windspeed=False,
            add_holiday=False,
            clean_junk_data = False,
            datasetName="PEMS08",
            dtype=np.float32):
    print("minute_offset:{}".format(minute_offset))
    tem_speed_data = None
    holiday_data = None
    pems = np.load(load_path)
    pems_data = np.array(pems["data"])
    time_len, nodes_num, oringe_dim = pems_data.shape
    if clean_junk_data:
        pems_data[:, :, 0] = handle_data_for_fix(pems_data[:, :, 0])
    dim = oringe_dim
    feature_dim = oringe_dim
    if add_time_in_day:
        feature_dim += 1
    if add_day_in_week:
        feature_dim += 1
    if add_tem_windspeed:
        feature_dim += 2
        tem_speed_data = read_tem_data(tem_speed_path ,datasetName)
    if add_holiday:
        feature_dim += 1
        holiday_data = get_holiday_data(datasetName)

    res = np.zeros((time_len, nodes_num, feature_dim), dtype=dtype)
    res[:, :, :oringe_dim] = pems_data[:, :, :oringe_dim]
    if feature_dim == oringe_dim:
        return res
    dateList = datelist(startDict[datasetName],endDict[datasetName])
    for i in range(time_len):
        d_index = int((minute_offset + i*5) / int(MinuteSize))
        minute_in_day = (5 * i + minute_offset) % MinuteSize
        if add_tem_windspeed:
            res[i, :, oringe_dim] = tem_speed_data[d_index, 0]
            res[i, :, oringe_dim+1] = tem_speed_data[d_index, 1]
            oringe_dim += 2
        if add_time_in_day:
            res[i, :, oringe_dim] = minute_in_day
            oringe_dim += 1
        if add_day_in_week:
            res[i, :, oringe_dim] = datetime.strptime(dateList[d_index], "%Y-%M-%d").weekday()
            oringe_dim += 1
        if add_holiday:
            res[i, :, oringe_dim] = holiday_data[d_index]
        oringe_dim = dim

    return res



'''generate dataset'''
def generate_dataset(data,
                     seq_len,
                     pre_len,
                     feature_dim=1,
                     time_len=None,
                     split_ratio=0.7,
                     test_ratio = 0.2,
                     normalize=True):
    """
        :param data: feature matrix
        :param seq_len: length of the train data sequence
        :param pre_len: length of the prediction data sequence
        :param time_len: length of the time series in total
        :param split_ratio: proportion of the training set
        :param normalize: scale the data to (0, 1], divide by the maximum value in the data
        :return: train set (X, Y) and test set (X, Y)
    """
    if time_len is None:
        time_len = data.shape[0]
    val_len = int(time_len * (1-split_ratio-test_ratio))
    train_size = int(time_len * split_ratio)
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_len]
    test_data = data[train_size + val_len:time_len]
    train_X, train_Y, val_X, val_Y, test_X, test_Y = list(), list(), list(), list(), list(), list()
    if feature_dim == 1:
        for i in range(len(train_data) - seq_len - pre_len):
            train_X.append(np.array(train_data[i : i + seq_len, :, 0]))
            train_Y.append(np.array(train_data[i + seq_len: i + seq_len + pre_len, :, 0]))
        for i in range(len(val_data) - seq_len - pre_len):
            val_X.append(np.array(val_data[i : i + seq_len, :, 0]))
            val_Y.append(np.array(val_data[i + seq_len: i + seq_len + pre_len, :, 0]))
        for i in range(len(test_data) - seq_len - pre_len):
            test_X.append(np.array(test_data[i : i + seq_len, :, 0]))
            test_Y.append(np.array(test_data[i + seq_len: i + seq_len + pre_len, :, 0]))
    else:
        for i in range(len(train_data) - seq_len - pre_len):
            train_X.append(np.concatenate((train_data[i + seq_len: i + seq_len + pre_len, :, :feature_dim], train_data[i + seq_len: i + seq_len + pre_len, :, 3:]), axis=2))
            train_Y.append(np.array(train_data[i + seq_len: i + seq_len + pre_len, :, 0]))
        for i in range(len(test_data) - seq_len - pre_len):
            test_X.append(np.concatenate((test_data[i + seq_len: i + seq_len + pre_len, :, :feature_dim], test_data[i + seq_len: i + seq_len + pre_len, :, 3:]), axis=2))
            test_Y.append(np.array(test_data[i + seq_len : i + seq_len + pre_len, :, 0]))
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    val_X = np.array(val_X)
    val_Y = np.array(val_Y)
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)
    data_stats = None
    if normalize:
        data_stats, train_X[:, :, :feature_dim], val_X[:, :, :feature_dim], test_X[:, :, :feature_dim] = normalization(train_X[:, :, :feature_dim], val_X[:, :, :feature_dim], test_X[:, :, :feature_dim])
    return train_X, train_Y, test_X, test_Y, data_stats


def generate_torch_datasets(
                            data,
                            config,
                            time_len=None,
                            split_ratio=0.8,
                            val_ratio=0.1,
                            normalize=True,
):
    train_dataset = None
    test_dataset = None
    stats = None
    if config['Model']['model_name'] == "EMBSFormer":
        all_data = generate_dataset_AST(
            data,
            int(config['Training']['num_of_hours']),
            int(config['Training']['num_of_days']),
            int(config['Training']['num_of_weeks']),
            int(config['Data']['pre_len']),
            int(config['Data']['seq_len']),
            int(config['Data']['feature_dim']),
            split_ratio=split_ratio,
            val_ratio = val_ratio
        )
        train_recent = all_data['train']['recent']
        train_days = all_data['train']['day']
        train_weeks = all_data['train']['week']
        train_target = all_data['train']['target']
        val_recent = all_data['val']['recent']
        val_days = all_data['val']['day']
        val_week = all_data['val']['week']
        val_target = all_data['val']['target']
        test_recent = all_data['test']['recent']
        test_days = all_data['test']['day']
        test_week = all_data['test']['week']
        test_target = all_data['test']['target']
        stats = all_data['stats']
        train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_recent),
                                                       torch.FloatTensor(train_days),
                                                       torch.FloatTensor(train_weeks),
                                                       torch.FloatTensor(train_target))
        val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(val_recent),
                                                      torch.FloatTensor(val_days),
                                                      torch.FloatTensor(val_week),
                                                      torch.FloatTensor(val_target))
        test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(test_recent),
                                                      torch.FloatTensor(test_days),
                                                      torch.FloatTensor(test_week),
                                                      torch.FloatTensor(test_target))
    else:
        train_X, train_Y, test_X, test_Y, stats = generate_dataset(
            data,
            seq_len=int(config['Data']['seq_len']),
            pre_len=int(config['Data']['pre_len']),
            feature_dim=int(config['Data']['feature_dim']),
            time_len=time_len,
            split_ratio=split_ratio,
            normalize=normalize,
        )
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
        )
    return train_dataset, val_dataset, test_dataset, stats

def generate_torch_datasets2(
                            data,
                            config,
                            time_len=None,
                            split_ratio=0.8,
                            val_ratio=0.1,
                            normalize=True,
):
    train_dataset = None
    test_dataset = None
    stats = None
    if config['Model']['model_name'] == "EMBSFormer":
        all_data = generate_dataset_AST2(
            data,
            int(config['Training']['num_of_hours']),
            eval(config['Training']['cycle_matrix_config']),
            int(config['Data']['pre_len']),
            int(config['Data']['seq_len']),
            int(config['Data']['feature_dim']),
            split_ratio=split_ratio,
            val_ratio = val_ratio
        )
        train_recent, train_cycle, train_target = all_data['train']['recent'], all_data['train']['cycle'], all_data['train']['target']
        val_recent, val_cycle, val_target = all_data['val']['recent'], all_data['val']['cycle'], all_data['val']['target']
        test_recent, test_cycle, test_target = all_data['test']['recent'], all_data['test']['cycle'], all_data['test']['target']
        stats = all_data['stats']
        train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_recent),
                                                       torch.FloatTensor(train_cycle).transpose(0, 1),
                                                       torch.FloatTensor(train_target))
        val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(val_recent),
                                                      torch.FloatTensor(val_cycle).transpose(0, 1),
                                                      torch.FloatTensor(val_target))
        test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(test_recent),
                                                      torch.FloatTensor(test_cycle).transpose(0, 1),
                                                      torch.FloatTensor(test_target))
    else:
        train_X, train_Y, test_X, test_Y, stats = generate_dataset(
            data,
            seq_len=int(config['Data']['seq_len']),
            pre_len=int(config['Data']['pre_len']),
            feature_dim=int(config['Data']['feature_dim']),
            time_len=time_len,
            split_ratio=split_ratio,
            normalize=normalize,
        )
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
        )
    return train_dataset, val_dataset, test_dataset, stats

'''
dataset generation: recently, day, week
'''
'''
dataset generation: recently, day, week
'''
def generate_dataset_AST(data, num_of_hours, num_of_days, num_of_weeks, num_of_pre, num_of_seq=0, feature_dim=1, hour_points_num=12, merge=False, split_ratio=0.7, val_ratio=0.1):
    print(data.shape)
    seq_len, nodes_num, _ = data.shape
    all_sample = []
    for i in range(seq_len):
        sample = get_indices_data(data, num_of_hours, num_of_days, num_of_weeks, num_of_pre, i, num_of_seq, hour_points_num)
        if not sample:
            continue
        hours_sample, days_sample, weeks_sample, target_sample = sample
        all_sample.append(
            (np.expand_dims(hours_sample, axis=0).transpose(0, 2, 3, 1),
            np.expand_dims(days_sample, axis=0).transpose(0, 2, 3, 1),
            np.expand_dims(weeks_sample, axis=0).transpose(0, 2, 3, 1),
            np.expand_dims(target_sample, axis=0).transpose(0, 2, 3, 1)[:, :, 0, :])
        )
    train_len = int(len(all_sample)*split_ratio)
    val_len = int(len(all_sample)*val_ratio)
    train_set = [np.concatenate(data,axis=0) for data in zip(*all_sample[:train_len])]
    val_set = [np.concatenate(data,axis=0) for data in zip(*all_sample[train_len:train_len+val_len])]
    test_set = [np.concatenate(data, axis=0) for data in zip(*all_sample[train_len+val_len:])]
    train_hours, train_days, train_weeks, train_target = train_set
    val_hours, val_days, val_weeks, val_target = val_set
    test_hours, test_days, test_weeks, test_target = test_set
    

    print("train: hours:{}, days:{}, weeks:{}".format(train_hours.shape, train_days.shape, train_weeks.shape))
    print("val: hours:{}, days:{}, weeks:{}".format(val_hours.shape, val_days.shape, val_weeks.shape))
    print("test: hours:{}, days:{}, weeks:{}".format(test_hours.shape, test_days.shape, test_weeks.shape))
    hours_stats, train_hours[:, :, :feature_dim, :], val_hours[:, :, :feature_dim, :], test_hours[:, :, :feature_dim, :] \
        = normalization(train_hours[:, :, :feature_dim, :],val_hours[:, :, :feature_dim, :], test_hours[:, :, :feature_dim, :])
    days_stats, train_days[:, :, :feature_dim, :], val_days[:, :, :feature_dim, :], test_days[:, :, :feature_dim, :] \
        = normalization(train_days[:, :, :feature_dim, :], val_days[:, :, :feature_dim, :], test_days[:, :, :feature_dim, :])
    weeks_stats, train_weeks[:, :, :feature_dim, :], val_weeks[:, :, :feature_dim, :], test_weeks[:, :, :feature_dim, :] \
        = normalization(train_weeks[:, :, :feature_dim, :], val_weeks[:, :, :feature_dim, :], test_weeks[:, :, :feature_dim, :])

    all_data = {
        'train': {
            'week': train_weeks,
            'day': train_days,
            'recent': train_hours,
            'target': train_target,
        },
        'val': {
            'week': val_weeks,
            'day': val_days,
            'recent': val_hours,
            'target': val_target
        },
        'test': {
            'week': test_weeks,
            'day': test_days,
            'recent': test_hours,
            'target': test_target
        },
        'stats': {
            'week': weeks_stats,
            'day': days_stats,
            'recent': hours_stats
        }
    }

    return all_data



'''
dataset generation: recently, day, week
'''
def generate_dataset_AST2(data, num_of_hours, cycle_matrix_config, num_of_pre, num_of_seq=0, feature_dim=1, hour_points_num=12, merge=False, split_ratio=0.8, val_ratio=0.1):
    seq_len, nodes_num, _ = data.shape
    all_sample = []
    for i in range(seq_len):
        sample = get_indices_data2(data, num_of_hours, cycle_matrix_config, num_of_pre, i, num_of_seq,
                                  hour_points_num)
        if not sample:
            continue
        hours_sample, cycle_data, target_sample = sample
        all_sample.append(
            (np.expand_dims(hours_sample, axis=0).transpose(0, 2, 3, 1),
             np.concatenate([np.expand_dims(np.expand_dims(cycle_cash_data, axis=0).transpose(0, 2, 3, 1), axis=1) for cycle_cash_data in cycle_data], axis=1),
             np.expand_dims(target_sample, axis=0).transpose(0, 2, 3, 1)[:, :, 0, :])
        )
    train_len = int(len(all_sample) * split_ratio)
    val_len = int(len(all_sample) * val_ratio)
    train_set = [np.concatenate(data, axis=0) for data in zip(*all_sample[:train_len])]
    val_set = [np.concatenate(data, axis=0) for data in zip(*all_sample[train_len:train_len + val_len])]
    test_set = [np.concatenate(data, axis=0) for data in zip(*all_sample[train_len + val_len:])]
    train_hours, train_cycle, train_target = train_set
    val_hours, val_cycle, val_target = val_set
    test_hours, test_cycle, test_target = test_set

    # 对周期历史数据置换位置
    train_cycle, val_cycle, test_cycle = (train_cycle.transpose(1, 0, 2, 3, 4),
                                          val_cycle.transpose(1, 0, 2, 3, 4),
                                          test_cycle.transpose(1, 0, 2, 3, 4))

    print("train: hours:{}, train_cycle:{}".format(train_hours.shape, train_cycle.shape))
    print("val: hours:{}, train_cycle:{}".format(val_hours.shape, val_cycle.shape))
    print("test: hours:{}, train_cycle:{}".format(test_hours.shape, test_cycle.shape))
    # 对特征归一化
    hours_stats, train_hours[:, :, :feature_dim, :], val_hours[:, :, :feature_dim, :], test_hours[:, :, :feature_dim, :] \
        = normalization(train_hours[:, :, :feature_dim, :], val_hours[:, :, :feature_dim, :],
                        test_hours[:, :, :feature_dim, :])
    cycle_stats = normalization_cycle(train_cycle, val_cycle, test_cycle, feature_dim)

    all_data = {
        'train': {
            'cycle': train_cycle,
            'recent': train_hours,
            'target': train_target,
        },
        'val': {
            'cycle': val_cycle,
            'recent': val_hours,
            'target': val_target
        },
        'test': {
            'cycle': test_cycle,
            'recent': test_hours,
            'target': test_target
        },
        'stats': {
            'cycle': cycle_stats,
            'recent': hours_stats
        }
    }

    return all_data



def get_indices_data(data, num_of_hours, num_of_days, num_of_weeks, num_of_pre, label_start_idx, num_of_seq=0, hour_points_num=12):

    hours_idx = search_data_idx(data.shape[0], num_of_hours, label_start_idx, num_of_pre, 1, 0, hour_points_num)
    if not hours_idx:
        return None

    days_idx = search_data_idx(data.shape[0], num_of_days, label_start_idx, num_of_pre, 24, num_of_seq, hour_points_num)
    if not days_idx:
        return None

    weeks_idx = search_data_idx(data.shape[0], num_of_weeks, label_start_idx, num_of_pre, 24*7, num_of_seq, hour_points_num)
    if not weeks_idx:
        return None

    # num_pre*(num_of_hours or num_of_days or num_of_weeks), n, features
    hours_data = np.concatenate([data[i:j] for i,j in hours_idx], axis=0)
    days_data = np.concatenate([data[i:j] for i,j in days_idx], axis=0)
    weeks_data = np.concatenate([data[i:j] for i,j in weeks_idx], axis=0)
    target = data[label_start_idx:label_start_idx+num_of_pre]
    return hours_data, days_data, weeks_data,target

def get_indices_data2(data, num_of_hours, cycle_matrix_config, num_of_pre, label_start_idx, num_of_seq=0, hour_points_num=12):
    '''
    data
    num_of_hours: int
    cycle_matrix_config: [{"cycle": 24, "num": 1},...,]
    num_of_pre: int
    label_start_idx: int
    '''
    
    cycle_num = len(cycle_matrix_config)
    cycle_data = []
    for i in range(cycle_num):
        cycle_cash_data_idx = search_data_idx(data.shape[0], cycle_matrix_config[i]["num"], label_start_idx, num_of_pre, int(cycle_matrix_config[i]["cycle"]), num_of_seq, hour_points_num)
        if not cycle_cash_data_idx:
            return None
        cycle_cash_data = np.concatenate([data[i:j] for i,j in cycle_cash_data_idx], axis=0)
        cycle_data.append(cycle_cash_data)
        
    hours_idx = search_data_idx(data.shape[0], num_of_hours, label_start_idx, num_of_pre, int(num_of_pre/hour_points_num), 0, hour_points_num)
    if not hours_idx:
        return None

    # num_pre*(num_of_hours or data for cycle), n, features
    hours_data = np.concatenate([data[i:j] for i,j in hours_idx], axis=0)
    target = data[label_start_idx:label_start_idx+num_of_pre]
    return hours_data, cycle_data, target




def search_data_idx(seq_len, batch_num, label_start_idx, num_of_pre, units, num_of_seq=0, hour_points_num=12):
    '''
    :param seq_len: input sequence length(l, N, features)
    :param batch_num: num_of_hours or num_of_days or num_of_weeks
    :param label_start_idx: int, start_idx
    :param num_of_pre: int
    :param units: int,hours:1 days: 24  weeks: 24*7
    :param hour_points_num: influenced by dataSampling
    :return: ((start_idx, end_idx),...,(start_idx, end_idx))
    '''
    if hour_points_num < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_of_pre + num_of_seq > seq_len:
        return None

    idx_list = []
    for i in range(1,batch_num+1):
        start_idx = label_start_idx - i * units * hour_points_num
        if start_idx >= 0:
            end_idx = start_idx + num_of_pre + num_of_seq
            idx_list.append((start_idx, end_idx))
        else:
            return None
    if len(idx_list) != batch_num:
        return None
    return idx_list[::-1]


# adj = get_adj_matrix_PEM("../../data/PEMS/PEMS08/PEMS08.csv", 170)
# d_matrix = get_d_matrix(adj)
# laplace = get_laplace_matrix(adj, d_matrix)
# laplace = scaled_laplacian(laplace)
# chebList = chebyshev_ploynomials(laplace, 3)
# print(chebList)
# print(adj)
# print(d_matrix)
# print(laplace)

# data = dataPre()
# train_X, train_Y, test_X, test_Y = generate_dataset(data, seq_len=12, pre_len=12)
# print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)
#
# data = dataPre(add_day_in_week=True, add_time_in_day=True)
# data = generate_dataset_AST(data, 1, 1, 1, 12, 12, feature_dim=3)