import argparse
import numpy as np
import os
import os.path as osp
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='syn', 
            choices=['syn', 'mnist', 'cifar', 'fmnist', 'cauchy', 'cauchy_2', 'levy_1.5'])
    return parser.parse_args()


def parse(s, mode='val'):
#     print(s)
    
    s1 = f'{mode}_correct = '
    s2 = f'{mode}_correct_class = ['
    s3 = f'{mode}_total = '
    s4 = f'{mode}_loss = '
    p1 = s.find(s1)
    p2 = s.find(s2)
    p3 = s.find(s3)
    p4 = s.find(s4)
    
#     print(p1, p2, p3, p4)
    
    val1 = int(s[p1+len(s1): p2-2])
    s_sub = s[p2+len(s2): p3-3].split(',')
    val2 = int(s_sub[0])
    val3 = int(s_sub[1])
    
    val4 = int(s[p3+len(s3): p4-2])
    val5 = float(s[p4+len(s4):])

    return (val1, val2, val3, val4), val5


epoch = 500
bs = 64

R_list = [1,2,5,10]
seed_list = [2,22,42,62,82]
lr_std_list = [0.1, 0.05, 0.01, 0.005, 0.001,0.0005,0.0002,0.0001]
lr_adv_list = [0.1, 0.05, 0.01, 0.005, 0.001,0.0005,0.0002,0.0001]
total_imb_class = np.asarray([[1000,1000],[1000,500],[1000,200],[1000,100]])
total_b_class = np.asarray([[1000,1000],[1000,1000],[1000,1000],[1000,1000]])
total_all = np.asarray([2000, 1500, 1200, 1100])

normtype_list = ['l2', 'linf']

args = parse_args()

if args.dataset == 'syn':
    suffix = ''
    norm_scale_list = {'l2': [5.0, 3.75, 2.5], 'linf': [1.0, 0.75, 0.5]} # synthetic dataset
elif args.dataset == 'cauchy':
    suffix = '_cauchy'
    norm_scale_list = {'l2': [23.425, 17.5688, 11.7125], 'linf': [14.403, 10.8023, 7.2015]} # synthetic dataset
elif args.dataset == 'cauchy_2':
    suffix = '_cauchy_2'
    norm_scale_list = {'l2': [70.0, 52.5, 35.0], 'linf': [53.0, 39.75, 26.5]}
elif args.dataset == 'levy_1.5':
    suffix = '_levy_1.5'
    norm_scale_list = {'l2': [2.94, 2.205, 1.47], 'linf': [0.555, 0.4163, 0.2775]}
elif args.dataset == 'mnist':
    suffix = '_mnist'
    norm_scale_list = {'l2': [2.6956,2.0217,1.3478], 'linf': [0.4341, 0.3256, 0.2171]} # 
elif args.dataset == 'cifar':
    suffix = '_cifar'
    # norm_scale_list = {'l2': [0.7822, 0.5867, 0.3911, ], 'linf': [ 0.0305, 0.0228, 0.0152, ]}
    # norm_scale_list = {'l2': [2.3468, 1.7601, 1.1734, ], 'linf': [ 0.0305, 0.0228, 0.0152, ]}
    # norm_scale_list = {'l2': [1.7601, 1.1734, 0.7822], 'linf': [ 0.0305, 0.0228, 0.0152, ]}
    # norm_scale_list = {'l2': [1.1734, 0.9778, 0.7822], 'linf': [ 0.0305, 0.0228, 0.0152, ]}
    norm_scale_list = {'l2': [0.9778, 0.8800, 0.7822], 'linf': [ 0.0305, 0.0228, 0.0152, ]}
    lr_adv_list = [0.005, 0.001,0.0005,0.0002,0.0001]
elif args.dataset == 'fmnist':
    suffix = '_fmnist'
    norm_scale_list = {'l2': [3.2306,2.4230,1.6153], 'linf': [0.2889,0.2166,0.1444]}
    total_imb_class = np.asarray([[1000,1000],[500,1000],[200,1000],[100,1000]])
else:
    raise NotImplementedError(f'dataset = {args.dataset} not implemented!')

# 1. collect raw data for std training

len_lr_std = len(lr_std_list)
len_lr_adv = len(lr_adv_list)

val_result = np.zeros((4,5,len_lr_std,4), dtype=np.int64)
test_imb_result = np.zeros((4,5,len_lr_std,4), dtype=np.int64)
test_b_result = np.zeros((4,5,len_lr_std,4), dtype=np.int64)

val_loss = np.zeros((4,5,len_lr_std))
test_imb_loss = np.zeros((4,5,len_lr_std))
test_b_loss = np.zeros((4,5,len_lr_std))

for i, R in enumerate(R_list):
    for k, seed in enumerate(seed_list):
        for j, lr in enumerate(lr_std_list):
            file_name = osp.join(osp.join(f'record{suffix}', f'train-std_R-{R}_lr-{lr}_epoch-{epoch}_bs-{bs}_seed-{seed}'), 'train.log')
            print(i,j,k, file_name)
            with open(file_name) as f:
                lines = f.readlines()

            r1, r2 = parse(lines[-3], 'val')
            val_result[i][k][j] = r1
            val_loss[i][k][j] = r2

            r1, r2 = parse(lines[-2], 'test')
            test_imb_result[i][k][j] = r1
            test_imb_loss[i][k][j] = r2

            file_name = osp.join(osp.join(f'record{suffix}', f'train-std_R-{R}_lr-{lr}_epoch-{epoch}_bs-{bs}_seed-{seed}'), 'test.log')
            with open(file_name) as f:
                lines = f.readlines()

            r1, r2 = parse(lines[-1], 'test')
            test_b_result[i][k][j] = r1
            test_b_loss[i][k][j] = r2


indmin = np.argmin(val_loss, axis=-1)


def get_std_accs(indmin, test_result, total_class):
    acc_class = np.zeros((4,5,2))
    correct_class = np.zeros((4,5,2), dtype=np.int64)
    acc = np.zeros((4, 5))

    for k in range(4):
        for j in range(5):
            v = indmin[k][j]

            print(test_result[k][j][v])
            print(test_result[k][j][v][1:-1] / total_class[k])
            print(test_result[k][j][v][0] / total_all[k])

            correct_class[k][j] = test_result[k][j][v][1:-1]
            acc_class[k][j] = test_result[k][j][v][1:-1] / total_class[k]
            acc[k][j] = test_result[k][j][v][0] / total_all[k]
    return acc_class, correct_class, acc

acc_imb_class, correct_imb_class, acc_imb = get_std_accs(indmin, test_imb_result, total_imb_class)
acc_b_class, correct_b_class, acc_b = get_std_accs(indmin, test_b_result, total_b_class)


# 2. adversarial training


adv_val_result = np.zeros((2,3,4,5,len_lr_adv,4), dtype=np.int64)
adv_test_imb_result = np.zeros((2,3,4,5,len_lr_adv,4), dtype=np.int64)
adv_test_b_result = np.zeros((2,3,4,5,len_lr_adv,4), dtype=np.int64)

adv_val_loss = np.zeros((2,3,4,5,len_lr_adv))
adv_test_imb_loss = np.zeros((2,3,4,5,len_lr_adv))
adv_test_b_loss = np.zeros((2,3,4,5,len_lr_adv))

for k, normtype in enumerate(normtype_list):
    for l, normscale in enumerate(norm_scale_list[normtype]):
        for i, R in enumerate(R_list):
            for p, seed in enumerate(seed_list):
                for j, lr in enumerate(lr_adv_list):
                    print(k,l,i,p,j)
                    file_name = osp.join(osp.join(f'record{suffix}', f'train-adv_R-{R}_lr-{lr}_normtype-{normtype}_normscale-{normscale}_epoch-{epoch}_bs-{bs}_seed-{seed}'), 'train.log')
                    print(file_name)
                    with open(file_name) as f:
                        lines = f.readlines()

                    r1, r2 = parse(lines[-3], 'val')
                    adv_val_result[k][l][i][p][j] = r1
                    adv_val_loss[k][l][i][p][j] = r2

                    r1, r2 = parse(lines[-2], 'test')
                    adv_test_imb_result[k][l][i][p][j] = r1
                    adv_test_imb_loss[k][l][i][p][j] = r2

                    file_name = osp.join(osp.join(f'record{suffix}', f'train-adv_R-{R}_lr-{lr}_normtype-{normtype}_normscale-{normscale}_epoch-{epoch}_bs-{bs}_seed-{seed}'), 'test.log')
                    print(file_name)
                    with open(file_name) as f:
                        lines = f.readlines()

                    r1, r2 = parse(lines[-1], 'test')
                    adv_test_b_result[k][l][i][p][j] = r1
                    adv_test_b_loss[k][l][i][p][j] = r2


indmin_adv = np.argmin(adv_val_loss, axis=-1)

def get_adv_accs(indmin_adv, adv_test_result, total_class):
    adv_acc_class = np.zeros((2,3,4,5,2))
    adv_correct_class = np.zeros((2,3,4,5,2), dtype=np.int64)
    adv_acc = np.zeros((2,3,4,5))

    for i in range(2):
        for j in range(3):
            for k in range(4):
                for l in range(5):
                    v = indmin_adv[i][j][k][l]

                    print(adv_test_result[i][j][k][l][v])
                    print(adv_test_result[i][j][k][l][v][1:-1] / total_class[k])
                    print(adv_test_result[i][j][k][l][v][0] / total_all[k])


                    adv_correct_class[i][j][k][l] = adv_test_result[i][j][k][l][v][1:-1]
                    adv_acc_class[i][j][k][l] = adv_test_result[i][j][k][l][v][1:-1] / total_class[k]
                    adv_acc[i][j][k][l] = adv_test_result[i][j][k][l][v][0] / total_all[k]
    return adv_acc_class, adv_correct_class, adv_acc

adv_acc_imb_class, adv_correct_imb_class, adv_acc_imb = get_adv_accs(indmin_adv, adv_test_imb_result, total_imb_class)
adv_acc_b_class, adv_correct_b_class, adv_acc_b = get_adv_accs(indmin_adv, adv_test_b_result, total_b_class)


# 3. compute AD

ad_class = acc_b_class[:,:,0] - acc_b_class[:,:,1]
adv_ad_class = adv_acc_b_class[:,:,:,:,0] - adv_acc_b_class[:,:,:,:,1]

if args.dataset == 'fmnist':
    ad_class = -ad_class
    adv_ad_class = -adv_ad_class
# diff = np.abs(ad_class - adv_ad_class)
diff = adv_ad_class - ad_class
diff_ave = np.mean(diff, axis=-1)
diff_std = np.std(diff, axis=-1) / np.sqrt(5)


print('diff_ave', diff_ave)
print('diff_std', diff_std)



# 4. compute ACC diff


diff_acc = acc_imb - adv_acc_imb
diff_acc_ave = np.mean(diff_acc, axis=-1)
diff_acc_std = np.std(diff_acc, axis=-1) / np.sqrt(5)



# 4. write to disk

write_dir = f'result{suffix}'
if not osp.exists(write_dir):
    os.makedirs(write_dir)
write_path = osp.join(write_dir, 'all_result.pth')

torch.save({
    'val_result': val_result,
    'val_loss': val_loss,
    'test_imb_result': test_imb_result,
    'test_b_result': test_b_result,
    'test_imb_loss': test_imb_loss,
    'test_b_loss': test_b_loss,

    'adv_val_result': adv_val_result,
    'adv_val_loss': adv_val_loss,
    'adv_test_imb_result': adv_test_imb_result,
    'adv_test_b_result': adv_test_b_result,
    'adv_test_imb_loss': adv_test_imb_loss,
    'adv_test_b_loss': adv_test_b_loss,

    'correct_imb_class': correct_imb_class,
    'correct_b_class': correct_b_class,
    'acc_imb_class': acc_imb_class,
    'acc_b_class': acc_b_class,
    'acc_imb': acc_imb,
    'acc_b': acc_b,
    'ad_class': ad_class,
    
    'adv_correct_imb_class': adv_correct_imb_class,
    'adv_correct_b_class': adv_correct_b_class,
    'adv_acc_imb_class': adv_acc_imb_class,
    'adv_acc_b_class': adv_acc_b_class,
    'adv_acc_imb': adv_acc_imb,
    'adv_acc_b': adv_acc_b,
    'adv_ad_class': adv_ad_class,

    'indmin': indmin,
    'indmin_adv': indmin_adv,
    
    'diff': diff,
    'diff_ave': diff_ave,
    'diff_std': diff_std,

    'diff_acc': diff_acc,
    'diff_acc_ave': diff_acc_ave,
    'diff_acc_std': diff_acc_std,
}, write_path)

