import argparse
import logging
import numpy as np
import os
import os.path as osp
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class CifarNet(nn.Module):
    def __init__(self, vgg_name):
        super(CifarNet, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 2)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class CifarDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.X[index], 0 if self.y[index]==3 else 1

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
    parser.add_argument('--hidden', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--n-epochs', type=int, default=20)
    parser.add_argument('--adv', action='store_true', default=False)
    parser.add_argument('--norm-type', type=str, default='linf', choices=['linf', 'l2'])
    parser.add_argument('--norm-scale', type=float)
    return parser.parse_args()


args = parse_args()

folder_main = 'record_cifar'

if args.adv:
    folder_sub = osp.join(folder_main, f'train-adv_R-{args.R}_lr-{args.lr}_normtype-{args.norm_type}_normscale-{args.norm_scale}_epoch-{args.n_epochs}_bs-{args.bs}_seed-{args.seed}_vgg')
else:
    folder_sub = osp.join(folder_main, f'train-std_R-{args.R}_lr-{args.lr}_epoch-{args.n_epochs}_bs-{args.bs}_seed-{args.seed}_vgg')

if not osp.exists(folder_sub):
    os.makedirs(folder_sub)

task_name = 'test'
setup_logger(task_name, os.path.join(folder_sub, f'test.log'))
logger = logging.getLogger(task_name)

logger.info(f'args = {args}')


torch.manual_seed(args.seed)
np.random.seed(args.seed)

# 2. data loading

data = torch.load(osp.join('data_cifar', f'data_R_1.pth'))
X_test, y_test = data['X_test'], data['y_test']

test_set = CifarDataset(X_test, y_test)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.bs, shuffle=False)


batch_size = 64
input_size = 3072
num_classes = 2


model = CifarNet(vgg_name='VGG11')
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
        X = X.cuda()
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

