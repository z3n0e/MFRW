import argparse
import os
import random

import matplotlib
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from advisor import *

matplotlib.use('agg')

from loader_clothing1M import Clothing1M
import model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description='PyTorch Clothing1M Resnet50 Training')
parser.add_argument('--epochs', default=20, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs_change_1', default=10, type=int,
                    help='number of total epochs to run')
parser.add_argument('--epochs_change_2', default=17, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch_size', '--batch-size', default=32, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-3)')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--prefetch', type=int, default=5, help='Pre-fetching threads')
parser.add_argument('--gpu', default='0', type=str, help='select gpu')

parser.set_defaults(augment=True)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

use_cuda = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")


def build_dataset():
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214))
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214))
    ])

    train_data = Clothing1M(set_split='train', transform=train_transform)
    val_data = Clothing1M(set_split='val', transform=train_transform)
    test_data = Clothing1M(set_split='test', transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.prefetch, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)

    return train_loader, val_loader, test_loader


def build_model():
    model_1, model_2 = model.resnet50_divided(pretrained=True, num_classes=14)

    if torch.cuda.is_available():
        model_1.cuda()
        model_2.cuda()
        torch.backends.cudnn.benchmark = True

    return model_1, model_2


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(optimizer_1, optimizer_2, epoch):
    lr = optimizer_1.param_groups[0]['lr']

    if epoch == args.epochs_change_1 + 1:
        lr = lr * 0.1
    if epoch == args.epochs_change_2 + 1:
        lr = lr * 0.1

    for param_group in optimizer_1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer_2.param_groups:
        param_group['lr'] = lr


def test(model_1, model_2, test_loader):
    global step_global

    model_1.eval()
    model_2.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            feature_out = model_1(inputs)
            outputs = model_2(feature_out)
            test_loss += F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    return accuracy


step_global = 0


def train(train_loader, train_meta_loader, model_1, model_2, vnet, optimizer_model_1, optimizer_model_2, optimizer_vnet,
          epoch):
    print('\nEpoch: %d' % epoch)

    global step_global
    train_loss = 0
    meta_loss = 0

    train_meta_loader_iter = iter(train_meta_loader)

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        model_1.train()
        model_2.train()

        inputs, targets = inputs.to(device), targets.to(device)

        # meta training (virtual train step + vnet optimization step)
        meta_model_1, meta_model_2 = build_model()
        meta_model_1.cuda()
        meta_model_2.cuda()

        meta_model_1.load_state_dict(model_1.state_dict())
        meta_model_2.load_state_dict(model_2.state_dict())

        with torch.no_grad():
            meta_feature = meta_model_1(inputs)
            meta_score = meta_model_2(meta_feature)
            meta_cost = F.cross_entropy(meta_score, targets, reduction='none')
            cost_v = torch.reshape(meta_cost, (len(meta_cost), 1))

        meta_feature = meta_model_1(inputs)

        meta_feature_w = vnet(meta_feature.data, cost_v.data)

        meta_feature_new = meta_feature * meta_feature_w

        meta_score = meta_model_2(meta_feature_new)

        meta_cost = F.cross_entropy(meta_score, targets)

        meta_model_1.zero_grad()
        meta_model_2.zero_grad()

        meta_lr = optimizer_model_1.param_groups[0]['lr']

        grads = torch.autograd.grad(meta_cost, (meta_model_1.params()), create_graph=True)
        meta_model_1.update_params(lr_inner=meta_lr, source_params=grads)
        del grads

        grads = torch.autograd.grad(meta_cost, (meta_model_2.params()), create_graph=True)
        meta_model_2.update_params(lr_inner=meta_lr, source_params=grads)
        del grads

        try:
            inputs_val, targets_val = next(train_meta_loader_iter)
        except StopIteration:
            train_meta_loader_iter = iter(train_meta_loader)
            inputs_val, targets_val = next(train_meta_loader_iter)

        inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)

        meta_feature_val = meta_model_1(inputs_val)
        meta_score_val = meta_model_2(meta_feature_val)
        l_g_meta = F.cross_entropy(meta_score_val, targets_val)
        prec_meta = accuracy(meta_score_val.data, targets_val.data, topk=(1,))[0]

        optimizer_vnet.zero_grad()
        l_g_meta.backward()
        optimizer_vnet.step()

        # Main training with vnet updated

        feature = model_1(inputs)
        with torch.no_grad():
            feature_w = vnet(feature.data, cost_v.data)

        feature_new = feature * feature_w

        score = model_2(feature_new)

        cost = F.cross_entropy(score, targets)

        prec_train = accuracy(score.data, targets.data, topk=(1,))[0]

        optimizer_model_1.zero_grad()
        optimizer_model_2.zero_grad()
        cost.backward()
        optimizer_model_1.step()
        optimizer_model_2.step()

        train_loss += cost.item()
        meta_loss += l_g_meta.item()

        if (batch_idx) % 50 == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'MetaLoss:%.4f\t'
                  'Prec@1 %.2f\t'
                  'Prec_meta@1 %.2f' % (
                      (epoch), args.epochs, batch_idx + 1, len(train_loader.dataset) / args.batch_size,
                      (train_loss / (batch_idx + 1)),
                      (meta_loss / (batch_idx + 1)), prec_train, prec_meta))

        step_global += 1


num_classes = 14

train_loader, train_meta_loader, test_loader = build_dataset()

model_1, model_2 = build_model()

vnet = Advisor(2048, 100, 2048).cuda()

optimizer_model_1 = torch.optim.SGD(model_1.params(), args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
optimizer_model_2 = torch.optim.SGD(model_2.params(), args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)

optimizer_vnet = torch.optim.Adam(vnet.params(), 1e-4)

def main():
    global step_global

    best_acc = 0
    for epoch in range(args.epochs):

        adjust_learning_rate(optimizer_model_1, optimizer_model_2, epoch + 1)

        train(train_loader, train_meta_loader, model_1, model_2, vnet, optimizer_model_1, optimizer_model_2,
              optimizer_vnet, epoch + 1)
        test_acc = test(model_1, model_2, test_loader=test_loader)

        if test_acc >= best_acc:
            best_acc = test_acc

    print('best accuracy:', best_acc)


if __name__ == '__main__':
    main()
