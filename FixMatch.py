from __future__ import print_function

import argparse
import datetime
import os
import shutil
import time
import random
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

from Ada.utils.console_logger import ConsoleLogger
from Ada.models.backbone import ResNet_18
import Ada.dataset.raf as dataset
from Ada.losses import SupConLoss
from Ada.utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from utils.log_to_email import send_log
from utils.set_data import set_datasets

parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')

epochs = 20
num_workers = 1
n_labels = 100
train_iter = 200

# 设置训练数据
train_root, test_root, train_label_root, test_label_root, start_model_path = set_datasets("R")

# epochs = 10
# num_workers = 2
# n_labels = 100
# train_iter = 2



# Optimization options 优化器参数
# epochs: 训练的轮数
parser.add_argument('--epochs', default=epochs, type=int, metavar='N',
                    help='number of total epochs to run')
# start-epoch: 从第几轮开始训练
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
# batch-size: 每个batch的大小
parser.add_argument('--batch-size', default=16, type=int, metavar='N',
                    help='train batchsize')
# learning-rate: 学习率
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                    metavar='LR', help='initial learning rate')
# num_workers: 使用多少个线程来加载数据
parser.add_argument('--num_workers', type=int, default=num_workers,
                    help='num of workers to use')
# Checkpoints 保存模型的参数
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs： manualSeed: 随机种子
parser.add_argument('--manualSeed', type=int, default=5, help='manual seed')
# Device options 设备参数 gpu: 使用哪个gpu:0
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

# Method options 模型参数
# n-labeled: 有标签的数据的数量
parser.add_argument('--n-labeled', type=int, default=n_labels,
                    help='Number of labeled data')
# train-iteration: 每个epoch训练多少次
parser.add_argument('--train-iteration', type=int, default=train_iter,
                    help='Number of iteration per epoch')
# out: 输出结果的文件夹
parser.add_argument('--out', default='result',
                    help='Directory to output the result')
# ema-decay: 指数移动平均的参数
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--lambda-u', default=1, type=float)
# Data
# train-root: 训练数据的路径
parser.add_argument('--train-root', type=str, default=train_root,
                    help="root path to train data directory")
# test-root: 测试数据的路径
parser.add_argument('--test-root', type=str, default=test_root,
                    help="root path to test data directory")
# old_label-train: 训练数据的标签
parser.add_argument('--label-train', default=train_label_root, type=str, help='')
# old_label-test: 测试数据的标签
parser.add_argument('--label-test', default=test_label_root, type=str, help='')

args = parser.parse_args(args=[])
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA 使用gpu cuda加速
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# check if GPU is available, if False choose CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Random seed 设置随机种子：如果参数中没有设置随机种子，则随机生成一个
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy 最高的准确率
last_time = datetime.datetime.now()
console_log_path = os.path.join(args.out, 'console_log_fixmatch.txt')
log_path = os.path.join(args.out, 'log_fixmatch.txt')

def main():
    global best_acc, last_time

    # create checkpoint dir 创建保存模型的文件夹
    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    use_cuda_console = '==> Use CUDA' if use_cuda else '==> Use CPU'
    print(use_cuda_console)

    console_logger = ConsoleLogger(console_log_path,title='console_log_fixmatch',console_content='')
    console_logger.append('\n==========================================')
    console_logger.append(last_time.strftime('%Y-%m-%d %H:%M:%S'))
    console_logger.append(use_cuda_console)

    params_console = '==> Parameters: \n' \
                     'epochs: {}, batch_size: {}, lr: {}, num_workers: {}, n_labeled: {}, train_iteration: {}, \n' \
                        'out: {}, ema_decay: {}, train_root: {}, test_root: {}, label_train: {}, label_test: {}'.format(
        args.epochs, args.batch_size, args.lr, args.num_workers, args.n_labeled, args.train_iteration, args.out,
        args.ema_decay, args.train_root, args.test_root, args.label_train, args.label_test)
    console_logger.append(params_console)

    # Data loading 加载数据
    loading_console = '==> Preparing RAF-DB'
    print(loading_console)
    console_logger.append(loading_console)
    # pytorch上给的通用的统计值
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # transform_train: 训练数据的预处理
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomApply([
            transforms.RandomCrop(224, padding=8)
        ], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_labeled_set, train_unlabeled_set, test_set = dataset.get_raf(args.train_root, args.label_train,
                                                                       args.test_root, args.label_test, args.n_labeled,
                                                                       transform_train=transform_train,
                                                                       transform_val=transform_val)

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.num_workers, drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.num_workers, drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=args.num_workers)

    # Model
    create_model_console = '==> Creating model'
    print(create_model_console)
    console_logger.append(create_model_console)

    def create_model(ema=False):
        model = ResNet_18(num_classes=7, checkpoint_path=start_model_path)

        # 多卡GPU训练
        model = torch.nn.DataParallel(model).to(device)

        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    cudnn.benchmark = True

    # 模型参数量
    total_params_console = '    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0)
    print(total_params_console)
    console_logger.append(total_params_console)

    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion_simclr = SupConLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ema_optimizer = WeightEMA(model, ema_model, alpha=args.ema_decay)

    # log文件位置
    logger = Logger(log_path, title='RAF')
    # log文件表头
    logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss U',  'Test Loss', 'Test Acc.'])

    test_accs = []
    threshold = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]

    # Train and val
    for epoch in range(args.start_epoch, args.epochs + 1):
        time_console = '\n当前系统时间：{}-{}-{} {}:{}:{}'\
            .format(last_time.year, last_time.month, last_time.day, last_time.hour, last_time.minute,
                last_time.second)
        print(time_console)
        console_logger.append(time_console)

        epoch_console = '\nEpoch: [%d | %d] LR: %f Threshold=[%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]' % (
            epoch, args.epochs, state['lr'], threshold[0], threshold[1], threshold[2], threshold[3], threshold[4],
            threshold[5], threshold[6])
        print(epoch_console)
        console_logger.append(epoch_console)

        train_loss, train_loss_x, train_loss_u = train(labeled_trainloader, unlabeled_trainloader,
                                                                       model, optimizer, ema_optimizer, criterion, threshold, use_cuda)
        _, train_acc, _, _ = validate(labeled_trainloader, ema_model, criterion, epoch, use_cuda,
                                                          mode='Train Stats')
        # threshold = adaptive_threshold_generate(outputs_new, targets_new, threshold, epoch)

        test_loss, test_acc, _, _ = validate(test_loader, ema_model, criterion, epoch, use_cuda, mode='Test Stats')

        # append logger file
        logger.append([train_loss, train_loss_x, train_loss_u, test_loss, test_acc])
        test_acc_log = 'Current Test Acc: {}' .format(test_acc)
        print(test_acc_log)
        console_logger.append(test_acc_log)


        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        test_accs.append(test_acc)

        epoch_end_time = datetime.datetime.now()
        time_console = '当前系统时间：{}-{}-{} {}:{}:{}'\
            .format(epoch_end_time.year, epoch_end_time.month, epoch_end_time.day, epoch_end_time.hour,
                epoch_end_time.minute, epoch_end_time.second)
        print(time_console)
        console_logger.append(time_console)

        cost_time_console = '本次耗时：{}'.format(epoch_end_time-last_time)
        print(cost_time_console)
        console_logger.append(cost_time_console)

        last_time = epoch_end_time

    logger.close()

    best_acc_console = 'Best acc:\t {}'.format(best_acc)
    print(best_acc_console)
    console_logger.append(best_acc_console)

    # 发送邮件
    send_log(log_path, console_log_path)


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion_ce, threshold, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.train_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(args.train_iteration):
        try:
            inputs_x, targets_x = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = next(labeled_train_iter)

        try:
            (inputs_u, inputs_u2, inputs_strong), _ = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_strong), _ = next(unlabeled_train_iter)

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_strong = inputs_strong.cuda()

        # compute guessed labels of unlabeled samples 计算无标签样本的猜测标签
        outputs_u, feature_u = model(inputs_u)

        p = torch.softmax(outputs_u, dim=1)
        max_probs, max_idx = torch.max(p, dim=1)
        max_idx = max_idx.detach()

        output_x, _ = model(inputs_x)

        mask = mask_generate(max_probs, max_idx, batch_size, threshold)

        Lx = criterion_ce(output_x, targets_x.long()).mean()

        output_strong, _ = model(inputs_strong)
        Lu = criterion_ce(output_strong, max_idx) * mask
        Lu = Lu.mean()
        loss = Lx + Lu * args.lambda_u

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))

        # compute gradient and do SGD step 计算随机梯度下降SGD步骤
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time 测量时间
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Total: {total:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f}'.format(
            batch=batch_idx + 1,
            size=args.train_iteration,
            total=bar.elapsed_td,
            loss=losses.avg,
            loss_x=losses_x.avg,
            loss_u=losses_u.avg,
        )
        bar.next()
    bar.finish()

    return (losses.avg, losses_x.avg, losses_u.avg)


def validate(valloader, model, criterion, epoch, use_cuda, mode):
    '''
    Run evaluation 验证
    '''
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))

    outputs_new = torch.ones(1, 7).to(device)
    targets_new = torch.ones(1).long().to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets.long()).mean()

            ##
            outputs_new = torch.cat((outputs_new, outputs), dim=0)
            targets_new = torch.cat((targets_new, targets), dim=0)
            ##

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Total: {total:} | Loss: {loss:.4f} | Accuracy: {top1: .4f}'.format(
                batch=batch_idx + 1,
                size=len(valloader),
                total=bar.elapsed_td,
                loss=losses.avg,
                top1=top1.avg,
            )
            bar.next()
        bar.finish()
    return (losses.avg, top1.avg, outputs_new, targets_new)


def save_checkpoint(states, is_best, checkpoint=args.out, filename='checkpoint_fixmatch.pth.tar'):
    '''Saves checkpoint to disk 将checkpoint保存到磁盘'''
    filepath = os.path.join(checkpoint, filename)
    torch.save(states, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best_fixmatch.pth.tar'))


def linear_rampup(current, rampup_length=args.epochs):
    """Linear rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class WeightEMA(object):
    '''
    Exponential moving average of model weights
    '''
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)


def mask_generate(max_probs, max_idx, batch, threshold):
    mask_ori = torch.zeros(batch)
    for i in range(7):
        idx = np.where(max_idx.cpu() == i)[0]
        m = max_probs[idx].ge(threshold[i]).float().to('cpu')
        for k in range(len(idx)):
            mask_ori[idx[k]] += m[k]
    return mask_ori.to(device)


def adaptive_threshold_generate(outputs, targets, threshold, epoch):
    '''
    自适应阈值生成
    '''
    outputs_l = outputs[1:, :]
    targets_l = targets[1:]
    probs = torch.softmax(outputs_l, dim=1)
    max_probs, max_idx = torch.max(probs, dim=1)
    eq_idx = np.where(targets_l.eq(max_idx).cpu() == 1)[0]

    probs_new = max_probs[eq_idx]
    targets_new = targets_l[eq_idx]
    for i in range(7):
        idx = np.where(targets_new.cpu() == i)[0]
        if idx.shape[0] != 0:
            threshold[i] = probs_new[idx].mean().cpu() * 0.97 / (1 + math.exp(-1 * epoch)) if probs_new[
                                                                                                  idx].mean().cpu() * 0.97 / (
                                                                                                          1 + math.exp(
                                                                                                      -1 * epoch)) >= 0.8 else 0.8
        else:
            threshold[i] = 0.8
    return threshold


if __name__ == '__main__':
    main()
