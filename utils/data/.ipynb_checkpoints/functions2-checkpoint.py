import numpy as np
import pandas as pd
from datetime import datetime
import torch
import holidays as hy

startDict = {'PEMS04': '20180101', 'PEMS07': '20170501', 'PEMS08': '20160701'}
endDict = {'PEMS04': '20180228', 'PEMS07': '20170831', 'PEMS08': '20160831'}
MinuteSize = 1440
DaySize = 7



def normalization(train, test):
    '''
    :param train: np
    :param test: np
    :return: stats, train_norm, test_norm
    '''
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    train_norm = (train-mean)/std
    test_norm = (test-mean)/std
    stats = {"mean": mean, "std": std}, train_norm, test_norm
    return stats, train_norm, test_norm


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

'''query long range mask'''
def get_long_range_mask_PEM(load_path, nodes_num, lamda):
    mask = np.zeros((nodes_num, nodes_num))
    adj_data = np.array(pd.read_csv(load_path))
    mask = np.zeros((nodes_num, nodes_num))
    for i in range(adj_data.shape[0]):
        if (lamda == None or mask[int(adj_data[i, 0]), int(adj_data[i, 1])] <= lamda):
            mask[int(adj_data[i, 0]), int(adj_data[i, 1])] = 1
            mask[int(adj_data[i, 1]), int(adj_data[i, 0])] = 1
    return mask

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


'''dataPre for PEM'''
def dataPre(load_path="../../data/PEMS/PEMS08/PEMS08.npz",
            tem_speed_path = "../../data/PEMS/PEMS08/TEMP_PEMS08",
            add_time_in_day=True,
            add_day_in_week=True,
            add_tem_windspeed=False,
            add_holiday=False,
            datasetName="PEMS08",
            dtype=np.float32):
    tem_speed_data = None
    holiday_data = None
    pems = np.load(load_path)
    pems_data = np.array(pems["data"])
    time_len, nodes_num, oringe_dim = pems_data.shape
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
        if add_tem_windspeed:
            d_index = int(i / int((MinuteSize/5)))
            res[i, :, oringe_dim] = tem_speed_data[d_index, 0]
            res[i, :, oringe_dim+1] = tem_speed_data[d_index, 1]
            oringe_dim += 2
        if add_time_in_day:
            res[i, :, oringe_dim] = 5 * i % MinuteSize
            oringe_dim += 1
        if add_day_in_week:
            d_index = int(i / int((MinuteSize/5)))
            res[i, :, oringe_dim] = datetime.strptime(dateList[d_index], "%Y-%M-%d").weekday()
            oringe_dim += 1
        if add_holiday:
            d_index = int(i / int((MinuteSize / 5)))
            res[i, :, oringe_dim] = holiday_data[d_index]
        oringe_dim = dim

    return res



'''generate dataset'''
def generate_dataset(data,
                     seq_len,
                     pre_len,
                     feature_dim=1,
                     time_len=None,
                     split_ratio=0.8,
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
    train_size = int(time_len * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:time_len]
    train_X, train_Y, test_X, test_Y = list(), list(), list(), list()
    if feature_dim == 1:
        for i in range(len(train_data) - seq_len - pre_len):
            train_X.append(np.concatenate((np.expand_dims(train_data[i + seq_len: i + seq_len + pre_len, :, :feature_dim], axis=2),
                                           train_data[i + seq_len: i + seq_len + pre_len, :, 3:]), axis=2))
            train_Y.append(np.array(train_data[i + seq_len: i + seq_len + pre_len, :, 0]))
        for i in range(len(test_data) - seq_len - pre_len):
            test_X.append(np.concatenate((np.expand_dims(test_data[i + seq_len: i + seq_len + pre_len, :, :feature_dim], axis=2),
                                          test_data[i + seq_len: i + seq_len + pre_len, :, 3:]), axis=2))
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
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)
    data_stats = None
    if normalize:
        data_stats, train_X[:, :, :, :feature_dim], test_X[:, :, :, :feature_dim] = normalization(train_X[:, :, :, :feature_dim], test_X[:, :, :, :feature_dim])
    return train_X, train_Y, test_X, test_Y, data_stats


def generate_torch_datasets(
                            data,
                            config,
                            time_len=None,
                            split_ratio=0.8,
                            normalize=True,
):
    train_dataset = None
    test_dataset = None
    stats = None
    if config['Model']['model_name'] == "EMBSFormer"\
            or config['Model']['model_name'] == "Recent_attention"\
            or config['Model']['model_name'] == "Day_attention"\
            or config['Model']['model_name'] == "Week_attention" \
            or config['Model']['model_name'] == "ASTGR_Day" \
            or config['Model']['model_name'] == "ASTGFormer_noASTG" \
            or config['Model']['model_name'] == "ASTGR_Week":
        all_data = generate_dataset_AST(
            data,
            int(config['Training']['num_of_hours']),
            int(config['Training']['num_of_days']),
            int(config['Training']['num_of_weeks']),
            int(config['Data']['pre_len']),
            int(config['Data']['seq_len']),
            int(config['Data']['feature_dim'])
        )
        train_recent = all_data['train']['recent']
        train_days = all_data['train']['day']
        train_weeks = all_data['train']['week']
        train_target = all_data['train']['target']
        test_recent = all_data['test']['recent']
        test_days = all_data['test']['day']
        test_week = all_data['test']['week']
        test_target = all_data['test']['target']
        stats = all_data['stats']
        train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_recent),
                                                       torch.FloatTensor(train_days),
                                                       torch.FloatTensor(train_weeks),
                                                       torch.FloatTensor(train_target))
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
    return train_dataset, test_dataset, stats

'''
dataset generation: recently, day, week
'''
'''
dataset generation: recently, day, week
'''
def generate_dataset_AST(data, num_of_hours, num_of_days, num_of_weeks, num_of_pre, num_of_seq=0, feature_dim=1, hour_points_num=12, merge=False, split_ratio=0.8):
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
    train_set = [np.concatenate(data,axis=0) for data in zip(*all_sample[:train_len])]
    test_set = [np.concatenate(data,axis=0) for data in zip(*all_sample[train_len:])]
    train_hours, train_days, train_weeks, train_target = train_set
    test_hours, test_days, test_weeks, test_target = test_set

    print("train: hours:{}, days:{}, weeks:{}".format(train_hours.shape, train_days.shape, train_weeks.shape))
    print("test: hours:{}, days:{}, weeks:{}".format(test_hours.shape, test_days.shape, test_weeks.shape))
    hours_stats, train_hours[:, :, :feature_dim, :], test_hours[:, :, :feature_dim, :] \
        = normalization(train_hours[:, :, :feature_dim, :], test_hours[:, :, :feature_dim, :])
    days_stats, train_days[:, :, :feature_dim, :], test_days[:, :, :feature_dim, :] \
        = normalization(train_days[:, :, :feature_dim, :], test_days[:, :, :feature_dim, :])
    weeks_stats, train_weeks[:, :, :feature_dim, :], test_weeks[:, :, :feature_dim, :] \
        = normalization(train_weeks[:, :, :feature_dim, :], test_weeks[:, :, :feature_dim, :])

    all_data = {
        'train': {
            'week': train_weeks,
            'day': train_days,
            'recent': train_hours,
            'target': train_target,
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

# data = load_features("../../data/PEMS/PEMS08/PEMS08.npz")
# data = dataPre(add_day_in_week=False, add_time_in_day=False)
# data = generate_dataset_AST(data, 1, 2, 2, 12)
#
# data = dataPre(add_day_in_week=True, add_time_in_day=True)
# data = generate_dataset_AST(data, 1, 1, 1, 12, 12, feature_dim=3)