import argparse
import copy
import logging
import numpy as np
import os
import os.path as osp
import time
from tqdm import tqdm
import torch
import torch.nn as nn

from advertorch.attacks.one_step_gradient import GradientSignAttack, GradientAttack
from advertorch.attacks.iterative_projected_gradient import LinfPGDAttack, L2PGDAttack

class MyLR(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MyLR, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        out = self.linear(x)
        return out

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

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
    parser.add_argument('--data-type', type=str, default='syn', 
                        choices=['syn', 'cauchy', 'cauchy_2', 'levy_1.5'])
    parser.add_argument('--norm-scale', type=float)
    return parser.parse_args()


# TODO:
# 5. may add support for lr scheduler



# 1. set up logging



args = parse_args()

if args.data_type == 'syn':
    folder_main = 'record'
else:
    folder_main = f'record_{args.data_type}'

if args.adv:
    folder_sub = osp.join(folder_main, f'train-adv_R-{args.R}_lr-{args.lr}_normtype-{args.norm_type}_normscale-{args.norm_scale}_epoch-{args.n_epochs}_bs-{args.bs}_seed-{args.seed}')
else:
    folder_sub = osp.join(folder_main, f'train-std_R-{args.R}_lr-{args.lr}_epoch-{args.n_epochs}_bs-{args.bs}_seed-{args.seed}')

if not osp.exists(folder_sub):
    os.makedirs(folder_sub)

task_name = 'train'
setup_logger(task_name, os.path.join(folder_sub, f'train.log'))
logger = logging.getLogger(task_name)

logger.info(f'args = {args}')


torch.manual_seed(args.seed)
np.random.seed(args.seed)

# 2. data loading

data = torch.load(osp.join(f'data_{args.data_type}', f'data_R_{args.R}.pth'))
X_train, X_val, X_test, y_train, y_val, y_test = data['X_train'], data['X_valid'], data['X_test'], data['y_train'], data['y_valid'], data['y_test']

train_set = Dataset(X_train, y_train)
val_set = Dataset(X_val, y_val)
test_set = Dataset(X_test, y_test)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.bs, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.bs, shuffle=False)


batch_size = 64
input_size = 100
num_classes = 2


model = MyLR(input_size, num_classes).cuda()

criterion = nn.CrossEntropyLoss()  
# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

losses = []


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


if args.adv:
    if args.norm_type == 'linf':
        adversary = GradientSignAttack(model, loss_fn=criterion, eps=args.norm_scale, clip_min=None, clip_max=None)
    elif args.norm_type == 'l2':
        adversary = GradientAttack(model, loss_fn=criterion, eps=args.norm_scale, clip_min=None, clip_max=None)


best_val_loss = 1e10
best_model = None

t = time.time()
for epoch in (pbar := tqdm(range(args.n_epochs))):
    # 1. training
    model.train()
    for i, (X, y) in enumerate(train_loader):
        X = X.cuda()
        y = y.cuda()

        optimizer.zero_grad()

        if args.adv:
            X_adv = adversary.perturb(X, y)
            output = model(X_adv)
        else:
            output = model(X)

        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # if (i+1) % 50 == 0:
        #     print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
        #     % (epoch+1, args.n_epochs, i+1, len(train_set)//batch_size, loss.item()))

    # 2. validation
    model.eval()
    val_correct, val_correct_class, val_total, val_loss = test(val_loader)
    pbar.set_description(f"val_loss = {val_loss : .4f}, val_acc = {val_correct / val_total : .4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        best_epoch = epoch

    if epoch - best_epoch == 50:
        break



logger.info(f'training finishes using {time.time() - t} seconds!')

model = copy.deepcopy(best_model)

val_correct, val_correct_class, val_total, val_loss = test(val_loader)
logger.info(f'val_correct = {val_correct}, val_correct_class = {val_correct_class}, val_total = {val_total}, val_loss = {val_loss}')

test_correct, test_correct_class, test_total, test_loss = test(test_loader)
logger.info(f'test_correct = {test_correct}, test_correct_class = {test_correct_class}, test_total = {test_total}, test_loss = {test_loss}')

model.cpu()
model_path = osp.join(folder_sub, 'model.pth')
torch.save(model.state_dict(), model_path)
torch.save(losses, osp.join(folder_sub, 'train_losses.pth'))

logger.info(f'model at epoch {best_epoch} saved to {model_path}')