#!/usr/env/bin python
# -*- coding: utf-8 -*-
"""
@author:
@since: 2023/03/09
@main.py
@function: run script
"""
import warnings
import numpy as np
import pandas as pd
import torch
import argparse
import os
import random
from modules.dPoeModel import DPoE, TCDiscriminator, weights_init
from modules.dPoeTraining import Trainer
from utils.DataLoader import mv_data_loader
from utils.ModelLoader import mv_model_loader
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
from torch import optim
from torch.autograd import Variable

warnings.filterwarnings("ignore")


def setup_seed(seed):
    torch.manual_seed(seed)  # Sets the seed for generating random numbers on CPU.
    torch.cuda.manual_seed_all(seed)  # Sets the seed for generating random numbers on all GPUs.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_gmm_baseline(dataset, args):
    run_times = 1
    for i in range(run_times):
        print('Running: %d/%d' % (i + 1, run_times))

        # --> Load training data
        train_loader, view_num, n_clusters, size = mv_data_loader(batch_size=args.batch_size,
                                                                  data_name=dataset + '.mat',
                                                                  train_mode=True)
        # data collection fn
        input_x = []
        input_y = []
        for batch_idx, batch in enumerate(train_loader):
            list_size = len(batch)
            for loc in range(0, list_size):
                batch[loc] = batch[loc].view(batch[loc].shape[0], -1)
            data_x = torch.cat(batch[:-1], dim=1)
            if batch_idx == 0:
                input_x = data_x.numpy()
                input_y = batch[list_size-1].numpy()
            else:
                input_x = np.append(input_x, data_x.numpy(), axis=0)
                input_y = np.append(input_y, batch[list_size-1].numpy(), axis=0)
        # --> Build a GMM model
        gmm = GaussianMixture(n_components=n_clusters, n_init=10)
        # Training the GMM model
        gmm.fit(input_x)
        print("====Training is done====")

        # --> Test the trained model
        test_loader, view_num, n_clusters, _ = mv_data_loader(data_name=dataset + '.mat', train_mode=False,
                                                              anomaly_type=args.type, anomaly_rate=args.rate)
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx != 0:
                raise RuntimeError("Test data load error. Test should be done in one batch.")
            list_size = len(batch)
            for loc in range(0, list_size):
                batch[loc] = batch[loc].view(batch[loc].shape[0], -1)
            data_x = torch.cat(batch[:-1], dim=1)
            input_x = data_x.numpy()
            y = batch[list_size - 1].numpy().squeeze()
            from utils.evaluation import test
            p = gmm.predict_proba(input_x)
            # print(p)
            test(y, p.max(1))
        print("====Test is done====")


def run_model(dataset, args):
    beta = args.beta
    delta = args.delta
    cap = args.cap
    AUCc = 0
    AUCcv = 0
    run_times = 1
    for i in range(run_times):
        print('Running: %d/%d' % (i + 1, run_times))
        expert = 'POE' if args.poe else 'MOE'
        params = str(beta) + '-' + str(args.tau) + '-' + str(cap) + '_tc' + str(delta) + 'dim' + str(args.comm_dim)
        model_name = expert + '_' + dataset + '_' + params + '.pt'
        use_cuda = torch.cuda.is_available()
        print("cuda is available?")
        print(use_cuda)
        latent_dims = {'spec': args.spec_dim,  # view-specific variable (representation)
                       'comm': args.comm_dim}  # view-common variable (representation)
        # img_size=(3, 64, 64)  # shape(channel, height, width)
        # img_size=(3, 32, 32)
        img_size = (1, 32, 32)
        if args.train_mode is True:
            # --> Load training data
            train_loader, view_num, n_clusters, size = mv_data_loader(batch_size=args.batch_size,
                                                                      data_name=dataset,
                                                                      train_mode=args.train_mode)
            print('Total iters in each epoch: %d' % (size / args.batch_size * args.epochs))
            # Define View-specific channels and View-common channels
            max_cap_batch = (size / args.batch_size) * args.epochs
            spec_capacity = [(args.spec_dim * cap) / 2, beta, delta, max_cap_batch]  # view-specific
            comm_capacity = [np.log(n_clusters), beta, max_cap_batch]  # view-common

            # --> Build a model
            model = DPoE(img_size=img_size, view_num=view_num, latent_dims=latent_dims, hidden_dim=args.hidden_dim,
                         tc_dim=args.tc_dim, share=args.share, tau=args.tau, use_cuda=use_cuda, poe=args.poe)
            if use_cuda:
                model.cuda()
            # print(model)
            # --> Build an optimizer
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_steps,
                                                        gamma=args.decay_rate)
            # --> Build TC Discriminator and Optimizer
            # tc_tuple = []
            # for m in range(view_num):
            tc = TCDiscriminator(s_dim=latent_dims['spec'], c_dim=latent_dims['comm'], hidden_size=args.tc_dim)
            if use_cuda:
                tc.cuda()
            tc.apply(weights_init)
            opt_tc = optim.Adam(tc.parameters(), lr=args.lr_D)
            # scd_tc = torch.optim.lr_scheduler.StepLR(opt_tc, step_size=args.decay_steps, gamma=args.decay_rate)
            # tc_tuple.append((tc, opt_tc))
            tc_tuple = (tc, opt_tc)

            # --> Build a trainer
            trainer = Trainer(model, tc_tuple, optimizer, spec_cap=spec_capacity, comm_cap=comm_capacity,
                              view_num=view_num, datax=dataset, use_cuda=use_cuda)

            # --> Train the model
            trainer.train(train_loader, scheduler, epochs=args.epochs)
            torch.save(trainer.model.state_dict(), args.ckpt_path + model_name)

        # --> Test the trained model
        test_mode = True  # TODO: always true
        pre_label = False
        if test_mode is True:
            path_to_model_folder = args.ckpt_path + model_name
            test_loader, view_num, n_clusters, _ = mv_data_loader(data_name=dataset, train_mode=False,
                                                                  anomaly_type=args.anomaly_type, anomaly_rate=args.rate)
            model = mv_model_loader(path=path_to_model_folder,
                                    img_size=img_size, view_num=view_num,
                                    latent_dims=latent_dims, hid_dim=args.hidden_dim, tc_dim=args.tc_dim,
                                    share=args.share, tau=args.tau, poe=args.poe)
            # print(model)  # Print model architecture
            model.eval()  # Sets the module in evaluation mode.
            with torch.no_grad():  # mainly for reducing memory consumption for computations
                for batch_idx, batch in enumerate(test_loader):
                    if batch_idx != 0:
                        raise RuntimeError("Test data load error. Test should be done in one batch.")
                    data = batch[0:-1]  # [0:-1] are feature dimensions
                    labels = batch[-1]  # [-1] is label dimension
                    inputs = []
                    for idx in range(view_num):
                        inputs.append(Variable(data[idx]))
                        if use_cuda:
                            inputs[idx] = inputs[idx].cuda()
                    latent_reps = model.encode(inputs)

                    z_comm = latent_reps['comm'].cpu().detach().data.numpy()
                    y = labels.cpu().detach().data.numpy()

                    from utils import evaluation
                    print('dPoE-C: y = z_comm.max(1)')
                    p = z_comm.argmax(1) if pre_label else z_comm.max(1)
                    evaluation.test(y, p)
                    AUCc += evaluation.auc(y, p)

    print('dPoE-C: %.4f ' % (AUCc / run_times))
    print("%s is done!" % dataset)
    return AUCc / run_times


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='size of batch')
    parser.add_argument('--hidden_dim', type=int, default=256, help='dimension of hidden embeddings in model')
    parser.add_argument('--spec_dim', type=int, default=10, help='dimension of view-specific representation')
    parser.add_argument('--comm_dim', type=int, default=10, help='dimension of view-common representation')
    parser.add_argument('--tc_dim', type=int, default=1000, help="dimension of TC discriminator hidden layer")
    parser.add_argument('--share', type=bool, default=False, help='share autoencoder')
    parser.add_argument('--poe', type=str, default=True, help='POE or MOE')
    parser.add_argument('--beta', type=int, default=50, help='trade-off coefficient')
    parser.add_argument('--delta', type=int, default=50, help='TC discriminator coefficient')
    parser.add_argument('--cap', type=int, default=1, help='controlled capacity')
    parser.add_argument('--tau', type=float, default=0.67, help='temperature param')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_D', type=float, default=1e-4, help='TC discriminator learning rate')
    parser.add_argument('--decay_steps', type=int, default=1, help='period of learning rate decay')
    parser.add_argument('--decay_rate', type=float, default=0.99, help='scale factor of learning rate decay')
    parser.add_argument('--data_path', type=str, default='./utils/DATA/', help='data path')
    parser.add_argument('--ckpt_path', type=str, default='./ckpts/', help='trained model store path')
    parser.add_argument('--train_mode', default='False', action='store_true', help='training or not')
    parser.add_argument('--anomaly_type', type=str, default='class', help='anomaly type: attr, class, view, mix')
    parser.add_argument('--rate', type=float, default=0.1, help='anomaly rate')
    parser.add_argument('--gpu', type=str, default='0', help='specified gpus')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    setup_seed(5)
    datasets = ['Multi-COIL-20']
    # view_num: 3, 3, 2, 3, 2, 2, 2, 2
    # datasets = ['Multi-COIL-10',
    #             'Multi-COIL-20',
    #             'Multi-MNIST',
    #             'Multi-FMNIST',
    #             '2V_MNIST_USPS',
    #             'Pascal-Sentence',
    #             'FOX-News',
    #             'CNN-News']

    # beta = [1, 10, 20, 30, 50, 70, 100, 150, 200, 500]
    # beta = [50]
    # delta = [1, 10, 20, 30, 50, 70, 100, 150, 200, 500]
    delta = [50]
    # dims = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 80, 100]
    # dims = [10]
    result = []
    # for args.beta in beta:
    # for args.comm_dim in dims:
        # args.spec_dim = args.comm_dim
    for args.delta in delta:
        aucs = []
        for d_idx in range(len(datasets)):  # select the processing dataset with dataset index
            print(args)
            dataset = datasets[d_idx]
            print(dataset)
            # model running...
            auc = run_model(dataset, args)
            aucs.append(auc)
            # run_gmm_baseline(dataset, args)
    #     result.append(aucs)
    # df = pd.DataFrame(result)
    # if args.poe is True:
    #     df.to_excel(r'./results/POE_TC_delt_' + args.type + str(args.rate) + '.xls')
    # else:
    #     df.to_excel(r'./results/MOE_TC_delta_' + args.type + str(args.rate) + '.xls')
