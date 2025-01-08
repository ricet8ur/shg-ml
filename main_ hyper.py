import  time
import os.path as osp
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.utils import remove_self_loops
from torch_geometric.datasets import MoleculeNet
import random
import argparse
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score,  precision_recall_curve
from sklearn.model_selection import train_test_split
import torch.optim as optim
from sklearn.model_selection import KFold
from hyperopt import fmin, tpe, hp, Trials
import os
from sklearn import metrics
import torch
import pandas as pd

import random
import argparse
import time

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score,  precision_recall_curve

import torch.optim as optim
from sklearn.model_selection import KFold
import os
import torch
import sys
from gnn_utils import EarlyStopping
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from model_gnn import GNN_Graph
from gnn.data import CIFData, CIF_Lister
class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if __name__ == '__main__':

    log_path = './Logs/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
    sys.stdout = Logger(log_file_name)

start_time = time.time()

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
setup_seed(1)
def flatten(a):
    return [item for sublist in a for item in sublist]

class Normalizer(object):

        def __init__(self, tensor):
            self.mean = torch.mean(tensor)
            self.std = torch.std(tensor)

        def norm(self, tensor):
            return (tensor - self.mean) / self.std

        def denorm(self, normed_tensor):
            return normed_tensor * self.std + self.mean

        def state_dict(self):
            return {'mean': self.mean,
                    'std': self.std}

        def load_state_dict(self, state_dict):
            self.mean = state_dict['mean']
            self.std = state_dict['std']

def train(model, data_loader,  criterion, optimizer, device):
        model.train()

        loss_accum = 0
        for step, batch_data in enumerate(data_loader):
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            pred = model(batch_data)
            true = batch_data.y.view(pred.shape)
            true_normed = normalizer.norm(true)
            loss = criterion(pred, true_normed)
            loss.backward()
            optimizer.step()
            loss_accum += loss.item()
        return loss_accum / (step + 1) 

def eval(model, data_loader, criterion, device):
        model.eval()
        loss_accum = 0
        with torch.no_grad():
            for step, batch_data in enumerate(data_loader):
                batch_data = batch_data.to(device)
                pred = model(batch_data)
                true = batch_data.y.view(pred.shape)
                true_normed = normalizer.norm(true)
                loss = criterion(pred, true_normed)
                loss_accum += loss.item()
            return loss_accum / (step + 1)

def test(model, data_loader, device):
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_id, batch_data in enumerate(data_loader):
                batch_data = batch_data.to(device)
                pred = model(batch_data)
                true = batch_data.y.view(pred.shape)
                y_true.append(true.view(pred.shape).detach().cpu())
                y_pred.append(normalizer.denorm(pred.detach().cpu()))

        return y_true, y_pred
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(description='CrystalGNN')
parser.add_argument('--property', default='nlo',
                        choices=['matbench_dielectric', 'matbench_log_gvrh', 'matbench_log_kvrh',
                                 'matbench_mp_e_form', 'matbench_mp_gap', 'matbench_jdft2d'],
                        help='crystal property to train ')
parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs to train (default: 500)')
parser.add_argument('--patience', type=float, default=100,
                        help='patiece (default:50)')
args = parser.parse_args()


loss_func= torch.nn.L1Loss()
num_classes = 1
stopper = EarlyStopping(mode='lower', patience=args.patience)


data_path = './data/nlo'
CRYSTAL_DATA = CIFData(data_path)
id_prop_file = os.path.join(data_path, 'id_prop.csv')
dataset = pd.read_csv(id_prop_file, names=['cif_id','label','gap_',' volume_','gap0','gap1',
                                           ' gap2','gap3','gap4','gap5','gap6','gap7','gap8',
                                           'gap9','volume0','volume1',' volume2','volume3',
                                           'volume4','volume5','volume6','volume7','volume8','volume9'])
k_folds = 10
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=1)
data_induce = np.arange(0, len(dataset))
    
print('gine_angle')
print('--------------------------------')
fold = 0
val_res = []
test_res = []
for train_val_idx, test_idx in kfold.split(data_induce):

    print(f'FOLD {fold}')
    print('--------------------------------')
    train_idx, val_idx = train_test_split(train_val_idx, train_size=0.9, random_state=1)
    target = dataset['label'].tolist()
    target_train = [target[i]for i in train_idx]
    target_train = torch.tensor(target_train)
    normalizer = Normalizer(target_train)
    train_dataset = CIF_Lister(train_idx, CRYSTAL_DATA, df=dataset)

    val_dataset = CIF_Lister(val_idx, CRYSTAL_DATA, df=dataset)
    test_dataset = CIF_Lister(test_idx, CRYSTAL_DATA, df=dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    hyper_space = {'lr': hp.choice('lr', [10 ** -2, 10 ** -3]),
                   'hidden_dim': hp.choice('hidden_dim', [64, 128, 256]),
                   'num_layer': hp.choice('num_layer_gin', [3, 4, 5, 6]),
                   'drop_ratio': hp.choice('drop_ratio', [0, 0.2,0.5])}
    def hyper_opt(hyper_paras):
                # get the model instance

        my_model = GNN_Graph(num_layer=hyper_paras['num_layer'],
                                         num_classes=1,
                                         emb_dim=hyper_paras['hidden_dim'],
                                         drop_ratio=hyper_paras['drop_ratio']
                                        )
        model_file_name = './saved_model/%s_%.6f_%s_%s_%s.pth' % (args.property,
                                                            hyper_paras['lr'],
                                                               hyper_paras['num_layer'],
                                                                hyper_paras['hidden_dim'],
                                                                  hyper_paras['drop_ratio'])



        stopper = EarlyStopping(mode='lower', patience=args.patience, filename=model_file_name)

        my_model.to(device)
        optimizer = torch.optim.AdamW(my_model.parameters(), lr=hyper_paras['lr'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        for epoch in range(1, args.epochs + 1):
            train_loss = train(my_model, train_loader, loss_func, optimizer, device)
            # early stopping
            val_loss = eval(my_model, val_loader, loss_func, device)
            scheduler.step()
            early_stop = stopper.step(val_loss, my_model)
            if early_stop:
                break

        stopper.load_checkpoint(my_model)

        return val_loss
    print('******hyper-parameter optimization is starting now******')
    trials = Trials()
    opt_res = fmin(hyper_opt, hyper_space, algo=tpe.suggest, max_evals=100, trials=trials)

    print('******hyper-parameter optimization is over******')

    print('the best hyper-parameters settings are:  ', opt_res)
    if fold == 0:
        break



        

