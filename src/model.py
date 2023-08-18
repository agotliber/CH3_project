import torch
import torch.nn as nn
import numpy as np
import random
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from scipy import stats
from torchmetrics import SpearmanCorrCoef

import dataset as Dataset
from helperMethods import *
from conf import *

def eval(test_seq, test_exp, test_dist, test_labels, multi_model):
    if torch.cuda.is_available():
        test_seq = test_seq.cuda()
        test_exp = test_exp.cuda()
        test_dist = test_dist.cuda()
        test_labels = test_labels.cuda()
    if torch.cuda.is_available():
        multi_model = multi_model.cuda()
    else:
        multi_model = multi_model.float()
#     multi_model = multi_model.to('cpu')
    test_predictions = multi_model(test_seq, test_dist, test_exp)
    test_loss = nn.L1Loss()(test_predictions, test_labels).item()
    test_corr = SpearmanCorrCoef()(test_predictions.squeeze(), test_labels.squeeze()).item()
#     print("Test      | Loss: {:.3f} Corr: {:.3f}".format(test_loss, test_corr))
    return test_loss, test_corr


class MotifDetector(nn.Module):
    def __init__(self):
      super(MotifDetector, self).__init__()
      self.conv_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(11,5))
      self.norm_1 = nn.BatchNorm2d(num_features=64) #momentum=0.99, eps=0.001?
      self.pooling_1 = nn.AvgPool2d(kernel_size=(10, 1), stride=(10, 1)) #conv_p , conv_s
      
      self.linear_1 = nn.Linear(in_features=5120, out_features=50)
      self.norm_2 = nn.BatchNorm1d(num_features=50) 
      self.linear_2 = nn.Linear(in_features=50, out_features=45)
      self.norm_3 = nn.BatchNorm1d(num_features=45) 
      self.linear_3 = nn.Linear(in_features=45, out_features=40) 
      self.norm_4 = nn.BatchNorm1d(num_features=40) 
      
    def forward(self, x):
      x = self.conv_1(x)
      x = self.norm_1(x)

      activation = nn.ELU(alpha=1.0, inplace=False)
      x = activation(x)
      x = self.pooling_1(x)
      x = x.reshape((-1, x.shape[1] * x.shape[2] * x.shape[3]))
      x = self.linear_1(x)
      x = self.norm_2(x)
      activation = nn.ELU(alpha=1.0, inplace=False)
      x = activation(x)
      x = self.linear_2(x)
      x = self.norm_3(x)
      x = activation(x)
      x = self.linear_3(x)
      x = self.norm_4(x)

      x = activation(x)
      return x


class DistAttenaionNet(nn.Module):
    def __init__(self):
        super(DistAttenaionNet, self).__init__()
        self.linear_1 = nn.Linear(in_features=Conf.dist_n_genes, out_features=int(Conf.dist_n_genes / 6)).half()
        self.norm_1 = nn.BatchNorm1d(num_features=int(Conf.dist_n_genes / 6)).half()
        self.linear_2 = nn.Linear(in_features=int(Conf.dist_n_genes / 6), out_features=Conf.exp_n_genes).half()
        self.norm_2 = nn.BatchNorm1d(num_features=Conf.exp_n_genes).half()
        self.softmax = nn.Softmax(dim=-1).half()


    def forward(self, x):
      x = self.linear_1(x)
      x = self.norm_1(x)

      activation = nn.ELU(alpha=1.0, inplace=False)
      x = activation(x)
      x = self.linear_2(x)
      x = self.norm_2(x)
      x = activation(x)
      x = self.softmax(x)
      return x


class ExpAttenaionNet(nn.Module):
    def __init__(self):
      super(ExpAttenaionNet, self).__init__()
      self.linear_1 = nn.Linear(in_features=Conf.exp_n_genes, out_features=int(Conf.exp_n_genes / 6)).half()
      self.norm_1 = nn.BatchNorm1d(num_features=int(Conf.exp_n_genes / 6)).half()
      self.linear_2 = nn.Linear(in_features=int(Conf.exp_n_genes / 6), out_features=Conf.exp_n_genes).half()
      self.norm_2 = nn.BatchNorm1d(num_features=Conf.exp_n_genes).half()
      self.softmax = nn.Softmax(dim=-1).half()


    def forward(self, x):
      x = self.linear_1(x)
      x = self.norm_1(x)
      activation = nn.ELU(alpha=1.0, inplace=False)
      x = activation(x)
      x = self.linear_2(x)
      x = self.norm_2(x)
      x = activation(x)
      x = self.softmax(x)
      return x


class SeqAttenaionNet(nn.Module):
    def __init__(self):
      super(SeqAttenaionNet, self).__init__()
      self.conv_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(11,5)).half()
      self.norm_1 = nn.BatchNorm2d(num_features=64).half() 
      self.pooling_1 = nn.AvgPool2d(kernel_size=(10, 1), stride=(10, 1)).half() 
      self.linear_1 = nn.Linear(in_features=5120, out_features=20).half()
      self.norm_2 = nn.BatchNorm1d(num_features=20).half() 
      self.linear_2 = nn.Linear(in_features=20, out_features=Conf.exp_n_genes).half()
      self.norm_3 = nn.BatchNorm1d(num_features=Conf.exp_n_genes).half()
      self.softmax = nn.Softmax(dim=-1).half()


    def forward(self, x):
      x = self.conv_1(x)
      x = self.norm_1(x)
      activation = nn.ELU(alpha=1.0, inplace=False)
      x = activation(x)
      x = self.pooling_1(x)
      x = x.reshape((-1, x.shape[1] * x.shape[2] * x.shape[3]))
      x = self.linear_1(x)
      x = self.norm_2(x)
      x = activation(x)
      x = self.linear_2(x)
      x = self.norm_3(x)
      x = activation(x)
      x = self.softmax(x)
      return x


connected_features_size = [50,45,40]

class MultiModel(nn.Module):
    def __init__(self):
      super(MultiModel, self).__init__()
      self.motifDetector = MotifDetector().float()
      self.distAttenaionNet = DistAttenaionNet().float()
      self.expAttenaionNet = ExpAttenaionNet().float()
      self.seqAttenaionNet = SeqAttenaionNet().float()
      #in_features=54028
      self.linear_1 = nn.Linear(in_features=3*Conf.exp_n_genes + 40, out_features=connected_features_size[0]).half()
      self.norm_1 = nn.BatchNorm1d(num_features=connected_features_size[0]).half()
      self.linear_2 = nn.Linear(connected_features_size[0],out_features=connected_features_size[1]).half()
      self.norm_2 = nn.BatchNorm1d(num_features=connected_features_size[1]).half()
      self.linear_3 = nn.Linear(in_features=connected_features_size[1], out_features=connected_features_size[2]).half()
      self.norm_3 = nn.BatchNorm1d(num_features=connected_features_size[2]).half()
      self.linear_4 = nn.Linear(in_features=connected_features_size[2], out_features=1).half()
      
    def forward(self, seq, dist, exp):
      motifDetector_out = self.motifDetector(seq.float())

      distAttenaionNet_out = self.distAttenaionNet(dist.float())
      distAttenaionNet_out = torch.mul(exp, distAttenaionNet_out)

      expAttenaionNet_out = self.expAttenaionNet(exp.float())
      expAttenaionNet_out = torch.mul(exp, expAttenaionNet_out)

      seqAttenaionNet_out = self.seqAttenaionNet(seq.float())
      seqAttenaionNet_out = torch.mul(seqAttenaionNet_out, exp)

      x = torch.cat((motifDetector_out, distAttenaionNet_out, expAttenaionNet_out, seqAttenaionNet_out),1)
      x = x.float()
      x = self.linear_1(x)
      x = self.norm_1(x)
      activation = nn.ELU(alpha=1.0, inplace=False)
      x = activation(x)
      x = self.linear_2(x)
      x = self.norm_2(x)
      x = activation(x)
      x = self.linear_3(x)
      x = self.norm_3(x)
      x = activation(x)
      x = self.linear_4(x)
      return x      


def get_next_batch(data, size, testing=False, use_cuda=True):
    np_seq_batch, np_exp_batch, np_dist_batch, np_labels_batch, _, _, _ = data.next_batch( batch_size=size, shuffle=True, testing=testing)

    seq_batch = torchvision.transforms.functional.to_tensor(np.array(np_seq_batch, dtype=np.float64))
    seq_batch =  seq_batch.reshape((-1, 1, Conf.numSurrounding * 2, Conf.numBases))
    padding = (0, 0, 5, 5, 0, 0) 
    seq_batch = F.pad(seq_batch, padding, "constant", 0)

    np_labels_batch = np.array(np_labels_batch).reshape((-1, 1))

    exp_batch = torchvision.transforms.functional.to_tensor(np.array(np_exp_batch, dtype=np.float64))
    exp_batch = exp_batch[0]
    # exp_batch = exp_batch.reshape((Conf.batch_size, 1, -1))

    dist_batch = torchvision.transforms.functional.to_tensor(np.array(np_dist_batch, dtype=np.float64))
    dist_batch = dist_batch[0]
    # dist_batch = dist_batch.reshape((Conf.batch_size, 1, -1))
    labels_batch = torch.from_numpy(np.array(np_labels_batch, dtype=np.float32))

    if torch.cuda.is_available():
        if use_cuda:
            seq_batch = seq_batch.cuda()
            exp_batch = exp_batch.cuda()
            dist_batch = dist_batch.cuda()
            labels_batch = labels_batch.cuda()
    else:
        print(f"Cuda is not available")

    return seq_batch, exp_batch, dist_batch, labels_batch

