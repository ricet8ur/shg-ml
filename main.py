
import numpy as np
import pandas as pd

import random
import argparse
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.optim as optim
from sklearn.model_selection import KFold
import os
import torch
from gnn_utils import EarlyStopping
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from model_gnn import GNN_Graph
from gnn.data import CIFData, CIF_Lister
import sys


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
                    help='number of epochs to train')
parser.add_argument('--patience', type=float, default=100,
                        help='patiece (default:50)')
parser.add_argument('--pretrain_model_file', type=str, default='./pre-trained/mp_gap.pth',
                        help='filename to read the model (if there is any)')
args = parser.parse_args()


loss_func = torch.nn.L1Loss()

data_path = './data/nlo'
CRYSTAL_DATA = CIFData(data_path)
id_prop_file = os.path.join(data_path, 'id_prop.csv')
dataset = pd.read_csv(id_prop_file, names=['cif_id','label','gap_','gap0','gap1',
                                           ' gap2','gap3','gap4','gap5','gap6','gap7','gap8',
                                           'gap9'])
k_folds =10
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=0)
data_induce = np.arange(0, len(dataset))
    
print('--------------------------------')
fold = 0
val_res = []
test_res = []
for train_val_idx, test_idx in kfold.split(data_induce):

    print(f'FOLD {fold}')
    print('--------------------------------')
    train_idx, val_idx = train_test_split(train_val_idx, train_size=0.9, random_state=0)
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

    model = GNN_Graph(num_layer=4,
                           num_classes=1,
                           emb_dim=64,
                           drop_ratio=0.2)
    if not args.pretrain_model_file == "":
        # model.load_state_dict(torch.load(args.model_file))
        pretrained_model = torch.load(args.pretrain_model_file)
    model_dict = model.state_dict()
    pretrained_dict = pretrained_model['model_state_dict']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model_file = './saved_model/%s_bst_%s.pth' % (args.property, fold)

    loss_func = torch.nn.L1Loss()
    stopper = EarlyStopping(mode='lower', patience=args.patience, filename=model_file)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=10 ** -3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, loss_func, optimizer, device)
        val_loss = eval(model, val_loader, loss_func, device)
        val_res.append(val_loss)
        if epoch % 20 == 0:
            print(epoch)
            print(train_loss)
            print(val_loss)
        scheduler.step()
        early_stop = stopper.step(val_loss, model)
        if early_stop:
            break

    stopper.load_checkpoint(model)
    val_mae = eval(model, val_loader, loss_func, device)
    val_res.append(val_mae)
    print('best_val_mae', val_mae)
    print('---------Evaluate Model on Test Set---------------')
    y_true, y_pred = test(model, test_loader, device)
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    test_scores = {'rmse': rmse, 'mae': mae, 'r2': r2}
    print("test score", test_scores)
    test_res.append(test_scores)
    test_true = flatten(y_true)
    test_pred = flatten(y_pred)
    df_test = pd.DataFrame(test_true, columns=['true'])
    df_test = pd.concat([df_test, pd.DataFrame(test_pred, columns=['pred'])], axis=1)
    save_path = 'predictions'
    os.makedirs(save_path, exist_ok=True)
    save_name = f'output_cv{fold}.csv'
    df_test.to_csv(f'{save_path}/{save_name}', index_label='Index')
    fold = fold + 1
cols = ['rmse', 'mae', 'r2']

te = [list(item.values()) for item in test_res]
te_pd = pd.DataFrame(te, columns=cols)
te_pd.to_csv(
    './stat_res/{}_statistical_results.csv'.format(args.property),
    index=False)

print('val mean:', np.mean(val_res), 'val std:', np.std(val_res))
print('testing mean:', np.mean(te, axis=0), 'test std:', np.std(te, axis=0))
end_time = time.time()
print('the total elapsed time is', end_time - start_time, 'S')



        

