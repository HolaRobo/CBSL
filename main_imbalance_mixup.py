from __future__ import print_function
from __future__ import division
import argparse
import os
import time
import datetime
import math
import numpy as np
import torch
import torch.utils.data
import torch.optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

from resnet import *
from MyDataset import *
from sampler_utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Training settings
parser = argparse.ArgumentParser(description='PyTorch imbalance')

# DATA
parser.add_argument('--root_path', type=str, default='',
                    help='path to root path of images')
parser.add_argument('--database', type=str, default='cifar10',
                    help='Which Database for train. ')
parser.add_argument('--train_list', type=str, default=None,
                    help='path to training list')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
# Network
parser.add_argument('--network', type=str, default='resnet34',
                    help='Which network for train. ')
# LR policy
parser.add_argument('--epochs', type=int, default=240,
                    help='number of epochs to train (default: 300)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate (default: 0.1)')
parser.add_argument('--step_size', type=list, default=None,
                    help='lr decay step')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    metavar='W', help='weight decay (default: 0.0005)')
parser.add_argument('--sample_ratio', type=float, default=0.6,
                    metavar='S', help='sample ratio (default: 0.6)')
parser.add_argument('--drop_ratio', type=float, default=0,
                    help='drop_ratio (default: 0)')
parser.add_argument('--right', type=float, default=0.6,
                    help='sample ratio (default: 0.6)')
parser.add_argument('--easy_ratio', type=float, default=0.7,
                    help='random ratio to select easy samples (default: 0.3)')
parser.add_argument('--dynamic', type=bool, default=True,
                    help='dynamic resample (default: True)')
parser.add_argument('--resample_epoch', type=int, default=60,
                    help='where to start dynamic sampler (default: 40)')
parser.add_argument('--weight_limit', type=bool, default=True,
                    help='set the weight limit (default: True)')
parser.add_argument('--cb', type=bool, default=True,
                    help='if use cb loss (default: True)')
parser.add_argument('--ml', type=bool, default=True,
                    help='if use ml (default: True)')
parser.add_argument('--num_class', type=int, default=100,
                    help='num class (default: 10)')
# Common settings
parser.add_argument('--save_path', type=str, default='checkpoint/',
                    help='path to save checkpoint')
parser.add_argument('--no_cuda', type=bool, default=False,
                    help='disables CUDA training')
parser.add_argument('--workers', type=int, default=2,
                    help='how many workers to load data')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda")



best_acc = 0

def main():
    # --------------------------------------model----------------------------------------
    if args.network is 'resnet34':
        model = resnet18()
        model_eval = resnet18()
    elif args.network is 'resnet50':
        model = resnet50()
        model_eval = resnet50()
    else:
        raise ValueError("NOT SUPPORT NETWORK! ")


    model = torch.nn.DataParallel(model).to(device)
    model_eval = model_eval.to(device)
    print(model)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)



    # ------------------------------------load image---------------------------------------
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    filelist = '../data/cifar100_train_imbalance.txt'
    trainset = MyDataset(root='../data/', datatxt='cifar100_train_imbalance.txt', transform=transform_train)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    testset = MyDataset(root='../data/', datatxt='cifar-100-python/cifar100_test.txt', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.workers)

    print('length of train Database: ' + str(len(trainloader.dataset)))
    print('Number of Identities: ' + str(args.num_class))

    # --------------------------------loss function and optimizer-----------------------------
    criterion_test = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(),\
                                lr=args.lr,\
                                momentum=args.momentum,\
                                weight_decay=args.weight_decay)\

    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    total_time = 0
    total_time_ = 0

    num_samples = len(trainloader.dataset)
    # ----------------------------------------train-------------------------------------------
    weights = [0.6] * num_samples
    stime = datetime.datetime.now()
    cls = get_class_num(filelist, args.num_class)
    if not args.cb:
        cls = np.ones(args.num_class)
    mlweight = np.ones(args.num_class)
    beta = 0.9999
    cbweight = [((1.0 - beta) / (1.0 - math.pow(beta, i))) for i in cls]
    #criterion_train = torch.nn.CrossEntropyLoss(weight=cbweight).to(device)
    for epoch in range(1, args.epochs + 1):
        stime = datetime.datetime.now()
        if epoch > args.resample_epoch + 1:
            weightsampler = CustomRandomSampler(weights, num_samples=num_samples, easy_ratio=args.easy_ratio, drop_ratio=args.drop_ratio, replacement=False)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,  num_workers=args.workers, sampler=weightsampler)
        sstime = datetime.datetime.now()
        loss_weight = cbweight * mlweight
        if not args.cb:
            loss_weight = mlweight

        if not args.ml:
            loss_weight = cbweight

        criterion_train = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(loss_weight)).to(device)
        weights = train(trainloader, model, criterion_train, optimizer, epoch, num_samples, weights)
        etime = datetime.datetime.now()
        total_time += (etime - stime).seconds
        total_time_ += (etime - sstime).seconds
        mlweight = test(testloader, model, criterion_test, optimizer, epoch, args.num_class)
        if not args.ml:
            mlweight = np.ones(args.num_class)

        print ('Total Train Time1: %ds ,   Total Train Time2: %ds' % (total_time_, total_time))
    print('Best Acc: %.2f' % best_acc)
    print('Finished Training')

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train(trainloader, model, criterion, optimizer, epoch, num_samples, weights):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    #weight_vec = torch.ones(num_samples)
    s_t = datetime.datetime.now();
    for batch_idx, (inputs, targets, imgpath, weight, idx) in enumerate(trainloader):
        adjust_learning_rate(optimizer, epoch, args.step_size)
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.alpha)
        optimizer.zero_grad()
        inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
        outputs = model(inputs)

        loss_func = mixup_criterion(targets_a, targets_b, lam)
        loss = loss_func(criterion, outputs)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        probs, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += lam * predicted.eq(targets_a.data).cpu().sum().item() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().item()

        if epoch > args.resample_epoch:
            pre = predicted.tolist()
            tara = targets_a.tolist()
            tarb = targets_b.tolist()
            ids = idx.tolist()
            prob = probs.tolist()
            decay = 0.3 * (epoch - args.resample_epoch) / (args.epochs - args.resample_epoch)
            if args.weight_limit:
                for i in range(len(pre)):
                    if weights[ids[i]] > 0 and weights[ids[i]] < 1.0:
                        if pre[i] == tara[i]:
                            weights[ids[i]] -= (decay * (1 - (math.fabs(prob[i] - lam))))
                        elif pre[i] == tarb[i]:
                            weights[ids[i]] -= (decay * (1 - (math.fabs(prob[i] + lam - 1))))
                        else:
                            if weights[ids[i]] < 0.5:
                                weights[ids[i]] = 0.55
                            else:
                                weights[ids[i]] += 0.1


            else:
                for i in range(len(pre)):
                    if pre[i] == tara[i]:
                        weights[ids[i]] -= (decay * (1 - (math.fabs(prob[i] - lam))))
                    elif pre[i] == tarb[i]:
                        weights[ids[i]] -= (decay * (1 - (math.fabs(prob[i] + lam - 1))))
                    else:
                        if weights[ids[i]] < 0.5:
                            weights[ids[i]] = 0.55
                        else:
                            weights[ids[i]] += 0.1
        if (batch_idx + 1) % 50 == 0:
            print ('Batch[%3d/%d] | Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
                % (batch_idx + 1, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        if (batch_idx + 1 == len(trainloader)):
            e_t = datetime.datetime.now()
            print ('Batch[%3d/%d] | Train Loss: %.3f | Train Acc: %.3f%% (%d/%d) | Train Time: %ds%dms '
                % (batch_idx + 1, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total, (e_t - s_t).seconds, (e_t - s_t).microseconds / 1000))

        #progress_bar(batch_idx, len(trainloader), 'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
        #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    #return weight_vec
    return weights

def test(testloader, model, criterion, optimizer, epoch, num_class):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    starttime = datetime.datetime.now()
    correct_cls = np.zeros(num_class)
    total_cls = np.zeros(num_class)
    with torch.no_grad():
        for batch_idx, (inputs, targets, imgpath, weight, idx) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            #inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pre = predicted.tolist()
            tar = targets.tolist()
            for i in range(len(pre)):
                total_cls[int(tar[i])] += 1
                if (pre[i] == tar[i]):
                    correct_cls[int(tar[i])] += 1

            if batch_idx + 1 == len(testloader):
                endtime = datetime.datetime.now()
                print ('Batch[%3d/%d] | Test Loss: %.3f | Test Acc: %.3f%% (%d/%d) | Test Time: %ds%dms'
                    % (batch_idx + 1, len(testloader), test_loss/(batch_idx+1), 100.*correct/total, correct, total, (endtime-starttime).seconds, (endtime-starttime).microseconds / 1000))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_' + str(args.drop_ratio) + '_' + str(args.easy_ratio) +  '.pth')
        best_acc = acc

    mlweight = np.sqrt((2.0 - np.true_divide(correct_cls, total_cls)))
    return mlweight

def adjust_learning_rate(optimizer, epoch, step_size):

    lr = args.lr
    if epoch <= 140:
        lr = lr * 1;
    elif epoch > 140 and epoch < 200:
       lr = lr * 0.1;
    else:
      lr = lr * 0.01
    #lr = args.lr * (0.1 ** (epoch // 120))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_class_num(input_file, num_class):
    cls = np.zeros(num_class)
    with open(input_file, 'r') as inf:
        lines = inf.readlines()
        for line in lines:
            line = line.rstrip()
            line = line.split()
            label = int(line[1])
            cls[label] += 1
            #print (label, cls[label])
    return cls

if __name__ == '__main__':
    print(args)
    main()
