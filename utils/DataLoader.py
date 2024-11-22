#!/usr/env/bin python
# -*- coding: utf-8 -*-
"""
@author:
@since: 2023/03/09
@DataLoader.py
@function: data loading, and data processing (generated three types of anomalies)
"""
import copy
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
import torch
import scipy.io as scio


def mv_data_loader(batch_size=32, path_to_data='./utils/DATA/', data_name='Multi-COIL-20',
                   train_mode=True, anomaly_type='view', anomaly_rate=0.1):
    """load multi-view dataset."""
    if train_mode is True:
        # load dataset
        mv_data = scio.loadmat(path_to_data + data_name + '.mat')  # mat_dict : dict
    else:
        test_data_file = path_to_data + data_name + '_' + anomaly_type + str(anomaly_rate) + '.mat'
        mv_data = scio.loadmat(test_data_file)
    # data collate: merges a list of samples to form a mini-batch of Tensor(s)
    num_view = len(mv_data) - 3 - 1  # as database-info itself has 3-dims, and label has 1-dim
    x1 = mv_data['X1']
    x2 = mv_data['X2']
    print("View 1 shape {}".format(x1.shape))
    print("View 2 shape {}".format(x2.shape))
    view1 = torch.from_numpy(x1).float()
    x1 = []  # remove space
    view2 = torch.from_numpy(x2).float()
    x2 = []  # remove space
    if num_view == 3:
        x3 = mv_data['X3']
        print("View 3 shape {}".format(x3.shape))
        view3 = torch.from_numpy(x3).float()
        x3 = []  # remove space
    y = mv_data['Y']
    num_cluster = len(np.unique(y))
    print('Num of clusters: %d.' % num_cluster)
    y = torch.from_numpy(y.squeeze()).float()  # Remove single-dimensional axes from an input array.
    if num_view == 2:
        mv_xy = [view1, view2, y]
    elif num_view == 3:
        mv_xy = [view1, view2, view3, y]
    else:
        raise RuntimeError("{} viewed data not supported. Check the number of views.".format(num_view))
    num_data = len(mv_xy[0])
    print('Num of data: %d' % num_data)
    # DataLoader. Combines a dataset (X) and a sampler (batch_size), and provides an iterable over the given dataset.
    # if not train_mode:
    #     mv_xy = label_corruption(mv_xy, anomaly_type, anomaly_rate, path_to_data)
    #     num_data = len(mv_xy[0])
    #     batch_size = num_data
    #     print('Num of test data: %d' % num_data)
    if num_view == 2:
        input_x = TensorDataset(mv_xy[0], mv_xy[1], mv_xy[2])
    elif num_view == 3:
        input_x = TensorDataset(mv_xy[0], mv_xy[1], mv_xy[2], mv_xy[3])
    if not train_mode:
        batch_size = num_data
    train_loader = DataLoader(input_x, batch_size=batch_size, shuffle=train_mode)

    return train_loader, num_view, num_cluster, num_data


def load_saver(batch_size=32, path_to_data='./utils/DATA/', data_name='Multi-COIL-10', train_mode=False,
               anomaly_type='view', anomaly_rate=0.1):
    """load multi-view dataset."""
    # load dataset
    mv_data = scio.loadmat(path_to_data + data_name + '.mat')  # mat_dict : dict
    # data collate: merges a list of samples to form a mini-batch of Tensor(s)
    num_view = len(mv_data) - 3 - 1  # as database-info itself has 3-dims, and label has 1-dim
    x1 = mv_data['X1']
    x2 = mv_data['X2']
    print("View 1 shape {}".format(x1.shape))
    print("View 2 shape {}".format(x2.shape))
    view1 = torch.from_numpy(x1).float()
    x1 = []  # remove space
    view2 = torch.from_numpy(x2).float()
    x2 = []  # remove space
    if num_view == 3:
        x3 = mv_data['X3']
        print("View 3 shape {}".format(x3.shape))
        view3 = torch.from_numpy(x3).float()
        x3 = []  # remove space
    y = mv_data['Y']
    num_cluster = len(np.unique(y))
    print('Num of clusters: %d.' % num_cluster)
    y = torch.from_numpy(y.squeeze()).float()  # Remove single-dimensional axes from an input array.
    if num_view == 2:
        mv_xy = [view1, view2, y]
    elif num_view == 3:
        mv_xy = [view1, view2, view3, y]
    else:
        raise RuntimeError("{} viewed data not supported. Check the number of views.".format(num_view))
    num_data = len(mv_xy[0])
    print('Num of data: %d' % num_data)
    # Test data
    mv_xy = label_corruption(mv_xy, anomaly_type, anomaly_rate, path_to_data)
    num_data = len(mv_xy[0])
    print('Num of test data: %d' % num_data)
    test_data_file = path_to_data + data_name + '_' + anomaly_type + str(anomaly_rate) + '.mat'
    # mat_dict = scio.loadmat(test_data_file)
    if num_view == 2:
        # input_x = [mv_xy[0], mv_xy[1], mv_xy[2]]
        scio.savemat(test_data_file, {'X1': mv_xy[0].numpy(), 'X2': mv_xy[1].numpy(), 'Y': mv_xy[2].numpy()})
    elif num_view == 3:
        # input_x = [mv_xy[0], mv_xy[1], mv_xy[2], mv_xy[3]]
        scio.savemat(test_data_file, {'X1': mv_xy[0].numpy(), 'X2': mv_xy[1].numpy(), 'X3': mv_xy[2].numpy(),
                                      'Y': mv_xy[3].numpy()})

    # -> TODO: for baselines in matlab, using the following '.reshape()' to flatten a tensor into 2D
    # test_data_file = path_to_data + data_name + '_' + anomaly_type + str(anomaly_rate) + '_2d.mat'
    # x1 = mv_xy[0].reshape(mv_xy[0].size()[0], -1)
    # x2 = mv_xy[1].reshape(mv_xy[1].size()[0], -1)
    # if num_view == 2:
    #     y = mv_xy[2].reshape(mv_xy[2].size()[0], -1)
    #     scio.savemat(test_data_file, {'X1': x1.numpy(), 'X2': x2.numpy(), 'Y': y.numpy()})
    # elif num_view == 3:
    #     x3 = mv_xy[2].reshape(mv_xy[2].size()[0], -1)
    #     y = mv_xy[3].reshape(mv_xy[3].size()[0], -1)
    #     scio.savemat(test_data_file, {'X1': x1.numpy(), 'X2': x2.numpy(), 'X3': x3.numpy(), 'Y': y.numpy()})


def ood_data_loader(path_to_data='./utils/DATA/', data_name='Multi-COIL-10'):
    """load multi-view dataset."""
    # load dataset
    mv_data = scio.loadmat(path_to_data + data_name + '.mat')  # mat_dict : dict
    # data collate: merges a list of samples to form a mini-batch of Tensor(s)
    num_view = len(mv_data) - 3 - 1  # as database info has 3-dims, and label has 1-dim
    x1 = mv_data['X1']
    x2 = mv_data['X2']
    view1 = torch.from_numpy(x1).float()
    x1 = []  # remove space
    view2 = torch.from_numpy(x2).float()
    x2 = []  # remove space
    if num_view == 3:
        x3 = mv_data['X3']
        view3 = torch.from_numpy(x3).float()
        x3 = []  # remove space
    y = mv_data['Y']
    num_data = y.shape[1]
    print('Num of data: %d' % num_data)
    num_cluster = len(np.unique(y))
    print('Num of clusters: %d.' % num_cluster)
    y = torch.from_numpy(y.squeeze())  # Remove single-dimensional axes from an input array.
    if num_view == 2:
        mv_xy = [view1, view2, y]
    elif num_view == 3:
        mv_xy = [view1, view2, view3, y]
    else:
        raise RuntimeError("{} viewed data not supported. Check the number of views.".format(num_view))

    return mv_xy, num_data


def attribute_anomaly(data, anomaly_rate):
    """feature perturbation"""
    view_num = len(data) - 1  # total num of views
    data_num = data[view_num].shape[0]  # total num of data
    anomaly_num = int(data_num * anomaly_rate)

    anomaly_idx = random.sample(range(data_num), anomaly_num)  # randomly set a list of idx for anomaly samples
    for idx in anomaly_idx:
        for v in range(view_num):
            data[v][idx] = data[v][idx].uniform_(0, 1)  # fill with random feature of uniform distribution

    # assigned labels
    data[view_num][:] = 1  # '1' as normal-labels
    data[view_num][anomaly_idx] = 0  # '0' as abnormal-labels

    return data


def class_anomaly(data, anomaly_rate):
    """view swapper"""
    view_num = len(data) - 1  # num of views in total
    data_num = data[view_num].shape[0]  # num of data in total
    anomaly_num = int(data_num * anomaly_rate)

    anomaly_idx = random.sample(range(data_num), int(anomaly_num/2))  # randomly set a list of idx for anomaly samples
    anomaly_set = set(anomaly_idx)
    for idx in anomaly_idx:
        swap_idx = random.randint(0, data_num - 1)  # randomly select swapper idx from other classes
        while data[view_num][idx] == data[view_num][swap_idx] or swap_idx in anomaly_set:
            swap_idx = random.randint(0, data_num - 1)
        anomaly_set.add(swap_idx)  # add swap idx <- anomaly
        v = random.randint(0, view_num - 1)  # randomly trigger a view, then swap their features of this view
        fea_x = copy.deepcopy(data[v][idx])
        data[v][idx] = copy.deepcopy(data[v][swap_idx])
        data[v][swap_idx] = copy.deepcopy(fea_x)

    anomaly_idx = list(anomaly_set)
    data[view_num][:] = 1  # '1' as normal-labels
    data[view_num][anomaly_idx] = 0  # '0' as abnormal-labels

    return data


def view_anomaly(data, anomaly_rate):
    """mixed corruption"""
    view_num = len(data) - 1  # total num of views
    data_num = data[view_num].shape[0]  # total num of data
    anomaly_num = int(data_num * anomaly_rate)

    anomaly_idx = random.sample(range(data_num), anomaly_num)  # randomly set a list of idx for anomaly samples
    for idx in anomaly_idx:
        swap_idx = random.randint(0, data_num - 1)  # randomly select swapper idx from other classes
        while data[view_num][idx] == data[view_num][swap_idx]:
            swap_idx = random.randint(0, data_num - 1)
        rnd_v = random.randint(0, view_num - 1)  # randomly trigger a view, then swap their features of this view
        data[rnd_v][idx] = data[rnd_v][swap_idx]  # corrupt the features of the data:idx in this view
        for v in range(view_num):
            if v == rnd_v:
                continue
            else:
                data[v][idx] = data[v][idx].uniform_(0, 1)  # fill the other views with random features

    data[view_num][:] = 1  # '1' as normal-labels
    data[view_num][anomaly_idx] = 0  # '0' as abnormal-labels

    return data


def ood_anomaly(data, anomaly_rate, path_to_data):
    """inject data with out-of-distribution"""
    view_num = len(data) - 1  # total num of views
    data_num = data[view_num].shape[0]  # total num of data
    anomaly_num = int(data_num * anomaly_rate)
    ood_dataset = 'Multi-FMNIST'
    ood_data, num_data = ood_data_loader(path_to_data, data_name=ood_dataset)

    anomaly_idx = random.sample(range(num_data), anomaly_num)
    for col in range(len(data)):
        data[col] = torch.cat((data[col], ood_data[col][anomaly_idx]), dim=0)

    data[view_num][:] = 1  # '1' as normal-labels
    data[view_num][data_num:data_num+anomaly_num] = 0  # '0' as abnormal-labels

    return data


def label_corruption(x, anomaly_type, anomaly_rate, path_to_data):
    """specified-type of anomaly generation on a dataset."""
    if anomaly_type == 'attr':  # Type-I: attribute-anomaly
        new_data = attribute_anomaly(x, anomaly_rate)
    elif anomaly_type == 'class':  # Type-II: class-anomaly
        new_data = class_anomaly(x, anomaly_rate)
    elif anomaly_type == 'view':  # Type-III: view-anomaly
        new_data = view_anomaly(x, anomaly_rate)
    elif anomaly_type == 'ood':  # out-of-distribution-anomaly
        new_data = ood_anomaly(x, anomaly_rate, path_to_data)
    else:
        raise RuntimeError("Injection error. Check the type of label corruption (anomaly_type).")

    return new_data


if __name__ == '__main__':

    datasets = ['Multi-COIL-20']
    # datasets = ['Multi-COIL-10',
    #             'Multi-COIL-20',
    #             'Multi-MNIST',
    #             'Multi-FMNIST',
    #             '2V_MNIST_USPS',
    #             'Pascal-Sentence',
    #             'FOX-News',
    #             'CNN-News']
    for d_idx in range(len(datasets)):  # Iteratively processing dataset with dataset index
        dataset = datasets[d_idx]
        print(dataset)
        # anomaly generating...
        # anomaly_type = 'attr', 'class', 'view', or 'ood'
        load_saver(batch_size=32, path_to_data='./DATA/', data_name=dataset,
                   train_mode=False, anomaly_type='view', anomaly_rate=0.1)
        print('Done!')
