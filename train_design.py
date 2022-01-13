import argparse
import math
import os
import time
import numpy as np
import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import model as m
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets.dataset import FashionAIDataset
from dataset import FashionAI
from sklearn.metrics import accuracy_score
from ASL_Loss import AsymmetricLossOptimized
from timm.loss.asymmetric_loss import AsymmetricLossMultiLabel
from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler


class CFG:
    debug = False
    apex = False
    resume = False
    checkpoint = '/workspace/fashionai/save/coat_length_labels/tf_efficientnet_b6_68.pth'
    data_folder = '/workspace/fashionai/datasets/fashionAI/train'
    log_dir = '/workspace/fashionai/save/'
    print_freq = 200
    num_workers = 1
    model_name = 'tf_efficientnet_b6'
    # tresnet_l_448 tresnet_xl_448 effientnet_b3a tf_efficientnet_b6 tf_efficientnet_b8
    # vit_base_patch16_224_in21k need std to be [0.5, 0.5, 0,5] modify dataset config
    img_tarin_size = [528, 528]
    img_test_size = [528, 528]
    epochs = 100
    lr = 0.01
    min_lr = 1e-6
    batch_size = 4
    weight_decay = 1e-6
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 42
    target_size = 5
    target_col = 'label'
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]
    smoothing = 0.05
    attribute = 'collar_design_labels'
    attribute_classes = FashionAI.AttrKey[attribute]

    # 'coat_length_labels': 8,
    # 'collar_design_labels': 5,
    # 'lapel_design_labels': 5,
    # 'neck_design_labels': 5,
    # 'neckline_design_labels': 10,
    # 'pant_length_labels': 6,
    # 'skirt_length_labels': 6,
    # 'sleeve_length_labels': 9,


def get_transforms(*, data):
    if data == 'train':
        return A.Compose([
            # A.Resize(CFG.size, CFG.size),
            A.RandomResizedCrop(CFG.img_tarin_size[0], CFG.img_tarin_size[1]),
            # A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            # A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.CoarseDropout(p=0.5),
            A.Normalize(
                mean=[0.642, 0.602, 0.588],
                std=[0.084, 0.088, 0.088],
            ),
            ToTensorV2()
        ])
    elif data == 'test':
        return A.Compose([
            A.Resize(CFG.img_test_size[0], CFG.img_test_size[1]),
            # A.CenterCrop(CFG.img_test_size[0], CFG.img_test_size[1]),
            # A.CenterCrop(CFG.size, CFG.size),
            A.Normalize(
                mean=[0.642, 0.602, 0.588],
                std=[0.084, 0.088, 0.088],
            ),
            ToTensorV2(),
        ])


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    minutes = math.floor(s / 60)
    s -= minutes * 60
    return '%dm %ds' % (minutes, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s' % (asMinutes(rs))
    # return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def get_scores(gt_labels, outputs, threshold=0.5):
    outputs[outputs > threshold] = 1
    outputs[outputs < threshold] = 0
    print(outputs)
    s = accuracy_score(gt_labels, outputs)
    return s


def train(train_loader, model, criterion, optimizer, epoch, device, attribute_classes, LOGGER):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    model.train()
    correct = 0
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        end = time.time()
        # data_time.update(time.time() - end)

        batch_size = target.size(0)
        # one_hot_target = F.one_hot(target, attribute_classes)  # todo
        # batch_size = 32
        # data.shape [32, 3, 288, 288]
        # target shape [32]
        # output shape [32, 8]

        if device:
            data, target = data.to(device), target.to(device)

        output = model(data)

        # loss = F.cross_entropy(output, target)
        loss = criterion(output, target)
        losses.update(loss.item(), batch_size)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        pred = output.cpu().data.max(1, keepdim=True)[1]
        normal_target = torch.argmax(target, -1)
        correct += np.equal(pred.squeeze(), normal_target.cpu().numpy().squeeze()).sum()

        # multi-label acc
        # print(target)
        # print(target.shape)
        # print(output)
        # print(output.shape)
        # scores.update(get_scores(normal_target.cpu().detach().numpy(), output.cpu().detach().numpy()))

        # pred = output.cpu().data.numpy().argmax()
        # scores.update(accuracy_score(normal_target.cpu(), pred.cpu()))

        batch_time.update(time.time() - end)

        if batch_idx % CFG.print_freq == 0:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            LOGGER.info('Epoch: {} [{}/{}] \t'
                        'Loss: {loss:.3f} \t'
                        'Lr:{Lr:.5f} \t'
                        'BatchTime:{batch_time.val:.1f}s \t'
                        'Finish:{to_finish:s} \t'
                        .format(
                            epoch, batch_idx * len(data), len(train_loader.dataset),
                            loss=losses.avg,
                            Lr=lr,
                            # data_time=data_time,
                            batch_time=batch_time,
                            to_finish=timeSince(start, float(batch_idx + 1) / len(train_loader)),
                        ))
        acc = 100 * correct / len(train_loader.dataset)

    return acc


def validation(valid_loader, model, criterion, device, attribute_classes, LOGGER):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    model.eval()
    preds = []
    correct = 0
    start = end = time.time()

    for batch_index, (data, target) in enumerate(valid_loader):
        data_time.update(time.time() - end)
        batch_size = target.size(0)
        # one_hot_target = F.one_hot(target, attribute_classes)

        if device:
            data, target = data.cuda(), target.cuda()

        with torch.no_grad():
            output = model(data)
        loss = criterion(output, target)
        losses.update(loss.item(), batch_size)

        # preds.append(output.softmax(1).to('cpu').numpy())
        # pred = output.softmax(1).to('cpu').numpy()

        pred = output.cpu().data.max(1, keepdim=True)[1]
        normal_target = torch.argmax(target, -1)
        correct += np.equal(pred.squeeze(), normal_target.cpu().numpy().squeeze()).sum()

        # pred = output.cpu().data.max(1, keepdim=True)[1]
        # normal_target = torch.argmax(target, -1)
        # scores.update(accuracy_score(normal_target.cpu(), pred.cpu()))

    LOGGER.info('Test set: Average loss: {:.4f}, Accuracy: {} / {} ({:.0f}%)'.format(
    # LOGGER.info('Test set: Average loss: {:.3f}, Accuracy:{score:.1f}%'.format(
        losses.avg,
        # score=scores.val,
        correct,
        len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)),
        )
    score = correct / len(valid_loader.dataset)
    return score


def init_logger(log_dir, task):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = CFG.log_dir + f'{task}_train1.log'
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='FashionAI')
    parser.add_argument('--model', type=str, default='timm', metavar='M',
                        help='model name')
    parser.add_argument('--attribute', type=str, default='collar_design_labels', metavar='A',
                        help='fashion attribute (default: coat_length_labels / collar_design_labels)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='input batch size for testing (default: 10)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.09, metavar='M',
                        help='SGD momentum (default: 0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--ci', action='store_true', default=False,
                        help='running CI')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # trainset = FashionAI('/workspace/fashionai/', attribute=args.attribute,
    #                      img_width=CFG.img_tarin_size[0], img_height=CFG.img_tarin_size[1],
    #                      split=0.8, ci=args.ci, data_type='train', reset=False)
    # testset = FashionAI('/workspace/fashionai/', attribute=args.attribute,
    #                     img_width=CFG.img_test_size[0], img_height=CFG.img_test_size[1],
    #                     split=0.8, ci=args.ci, data_type='test', reset=trainset.reset)

    LOGGER = init_logger(CFG.log_dir, CFG.attribute)

    train_dataset = FashionAIDataset(CFG.data_folder, CFG.attribute, mode='train',
                                     transform=get_transforms(data='train'))
    test_dataset = FashionAIDataset(CFG.data_folder, CFG.attribute, mode='test', transform=get_transforms(data='test'))

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=1, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=CFG.batch_size,
                             shuffle=False,
                             num_workers=1, pin_memory=True, drop_last=False)

    if args.ci:
        CFG.model_name = 'ci'
    LOGGER.info("============== Task:{} ================".format(CFG.attribute))
    LOGGER.info("Using model {}".format(CFG.model_name))
    # model = m.create_model(args.model, FashionAI.AttrKey[args.attribute])
    model = timm.create_model(CFG.model_name, pretrained=True, num_classes=CFG.attribute_classes)
    # model = timm.create_model(CFG.model_name, pretrained=True, features_only=True)

    save_folder = os.path.join(os.path.expanduser('.'), 'save', CFG.attribute, CFG.model_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if CFG.resume:
        # chakan wenjian shifou cunzai
        LOGGER.info("============ loading checkpoint {} ================".format(CFG.checkpoint))
        # start_epoch = torch.load(CFG.checkpoint) # todo
        model.load_state_dict(torch.load(CFG.checkpoint))
        start_epoch = 0
    else:
        start_epoch = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=CFG.lr, momentum=0.9, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1, last_epoch=-1)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG.epochs, T_mult=1, eta_min=1e-6,
                                                               last_epoch=-1)

    # loss_fn = nn.CrossEntropyLoss()
    # new loss fun
    # loss_fn = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=1, clip=0.05)
    loss_fn = AsymmetricLossMultiLabel()

    best_score = 0
    for epoch in range(start_epoch + 1, CFG.epochs + 1):
        train_acc, multi_score = train(train_loader, model, loss_fn, optimizer, epoch, device,
                          attribute_classes=CFG.attribute_classes, LOGGER=LOGGER)
        LOGGER.info('train set acc:{}%, multi_score: {}'.format(train_acc, multi_score))

        score = validation(test_loader, model, loss_fn, device, attribute_classes=CFG.attribute_classes, LOGGER=LOGGER)

        scheduler.step()

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), os.path.join(save_folder, CFG.model_name + '_' + str(epoch) + '.pth'))


if __name__ == "__main__":
    main()
