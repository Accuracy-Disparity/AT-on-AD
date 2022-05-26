import argparse
import logging
import numpy as np
import os
import os.path as osp
import torch
import torch.nn as nn


class MyLR(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MyLR, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        out = self.linear(x)
        return out

class MnistDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.X[index], 0 if self.y[index]==1 else 1

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--R', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--n-epochs', type=int, default=20)
    parser.add_argument('--adv', action='store_true', default=False)
    parser.add_argument('--norm-type', type=str, default='linf', choices=['linf', 'l2'])
    parser.add_argument('--norm-scale', type=float)
    return parser.parse_args()


args = parse_args()

folder_main = 'record_mnist'

if args.adv:
    folder_sub = osp.join(folder_main, f'train-adv_R-{args.R}_lr-{args.lr}_normtype-{args.norm_type}_normscale-{args.norm_scale}_epoch-{args.n_epochs}_bs-{args.bs}_seed-{args.seed}')
else:
    folder_sub = osp.join(folder_main, f'train-std_R-{args.R}_lr-{args.lr}_epoch-{args.n_epochs}_bs-{args.bs}_seed-{args.seed}')

if not osp.exists(folder_sub):
    os.makedirs(folder_sub)

task_name = 'test'
setup_logger(task_name, os.path.join(folder_sub, f'test.log'))
logger = logging.getLogger(task_name)

logger.info(f'args = {args}')


torch.manual_seed(args.seed)
np.random.seed(args.seed)

# 2. data loading

data = torch.load(osp.join('data_mnist', f'data_R_1.pth'))
X_test, y_test = data['X_test'], data['y_test']

test_set = MnistDataset(X_test, y_test)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.bs, shuffle=False)


batch_size = 64
input_size = 784
num_classes = 2


model = MyLR(input_size, num_classes)
model.load_state_dict(torch.load(osp.join(folder_sub, 'model.pth')))
model.cuda()
model.eval()

criterion = nn.CrossEntropyLoss()  


def test(data_loader):
    eval_losses = []
    correct = 0
    correct_class = [0, 0]
    total = 0    
    for X, y in data_loader:
        X = X.view(-1, 784).cuda()
        y = y.cuda()

        output = model(X)
        loss = criterion(output, y)
        _, predicted = torch.max(output.data, 1)

        eval_losses.append(loss.item())

        total += y.size(0)
        correct += (predicted == y).sum().item()
        
        correct_class[0] += torch.logical_and(predicted==y, y==0).sum().item()
        correct_class[1] += torch.logical_and(predicted==y, y==1).sum().item()

    return correct, correct_class, total, np.mean(eval_losses)


test_correct, test_correct_class, test_total, test_loss = test(test_loader)
logger.info(f'test_correct = {test_correct}, test_correct_class = {test_correct_class}, test_total = {test_total}, test_loss = {test_loss}')
