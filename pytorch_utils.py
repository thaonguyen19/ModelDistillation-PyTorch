import itertools
import tqdm
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import f1_score, precision_recall_fscore_support

# For NO REASON PyTorch v0.2 doesn't actually come with this???
from torch.optim import Optimizer
from bisect import bisect_right
from torch.nn import Softmax

TRAIN_PATH = "/mnt/disks/imagenet/ILSVRC2012_img_train"
#TRAIN_PATH = "/lfs/raiders3/1/ddkang/imagenet/ilsvrc2012/ILSVRC2012_img_train"
VAL_PATH = "/mnt/disks/imagenet/ILSVRC2012_img_val/"
#VAL_PATH = "/lfs/raiders3/1/ddkang/imagenet/ilsvrc2012/ILSVRC2012_img_val"
class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class MultiStepLR(_LRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma once the number of epoch reaches one of the milestones. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: -0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer uses lr = 0.5 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>     validate(...)
    """

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gamma = gamma
        super(MultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs]

class ReduceLROnPlateau(object):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. Default: 10.
        verbose (bool): If True, prints a message to stdout for
            each update. Default: False.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.eps = eps
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + mode + ' is unknown!')
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            self.is_better = lambda a, best: a < best * rel_epsilon
            self.mode_worse = float('Inf')
        elif mode == 'min' and threshold_mode == 'abs':
            self.is_better = lambda a, best: a < best - threshold
            self.mode_worse = float('Inf')
        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            self.is_better = lambda a, best: a > best * rel_epsilon
            self.mode_worse = -float('Inf')
        else:  # mode == 'max' and epsilon_mode == 'abs':
            self.is_better = lambda a, best: a > best + threshold
            self.mode_worse = -float('Inf')




import torch.utils.data as data
import torchvision
# THIS CLASS DEPENDS ON THE INTERNAL IMPLEMENTATION OF IMAGEFOLDER
def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
class ImageList(torchvision.datasets.ImageFolder):
    # Images take the form (path, class)
    def __init__(self, classes, imgs, transform=None, target_transform=None,
                 loader=pil_loader):
        self.classes = classes
        self.class_to_idx = dict(zip(classes, range(len(classes))))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

class RandomRotate(object):
    def __init__(self, rot_range):
        self.rot_range = rot_range
    def __call__(self, img):
        angle = np.random.uniform(-self.rot_range, self.rot_range)
        return img.rotate(angle)

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

def pytorch_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)#, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train_epoch(train_loader, big_model, small_model, T, criterion, optimizer, epoch, loss_weight=0.2):
    big_model.eval()
    big_model.cuda()
    small_model.train()
    small_model.cuda()
    losses = AverageMeter()
    top1_acc = AverageMeter()
    #top1_f1 = AverageMeter()

    pbar = tqdm.tqdm(train_loader)
    for inp, class_target in pbar:
        pbar.set_description('loss: %2.4f, acc: %2.1f' % (losses.avg, top1_acc.avg))
        inp = inp.cuda(async=True)
        input_var = torch.autograd.Variable(inp)

        logits_small = small_model(input_var) #type Var
        logits_big = big_model(input_var) #type Var
        #logits_small_var = torch.autograd.Variable(torch.div(logits_small.data, T).cuda())
        #logits_big_var = torch.autograd.Variable(torch.div(logits_big.data, T).cuda())

        soft_logits_small = Softmax()(logits_small * 1.0/T) 
        soft_logits_big = Softmax()(logits_big *1.0/T)

        loss_soft = torch.nn.BCELoss().cuda()(soft_logits_small, soft_logits_big)

        #output = logits_small
        class_target = class_target.cuda(async=True)
        class_target_var = torch.autograd.Variable(class_target)
        loss_hard = criterion(logits_small, class_target_var)

        loss = loss_weight * loss_hard + loss_soft
        prec1 = pytorch_accuracy(logits_small.data, class_target)
        #f1score1 = pytorch_f1(output.data, target)

        losses.update(loss.data[0], inp.size(0))
        top1_acc.update(prec1[0][0], inp.size(0))
        #top1_f1.update(f1score1[0], inp.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def val_epoch(val_loader, model, criterion): #temperature 1
    model.eval()
    model.cuda()
    losses = AverageMeter()
    top1_acc = AverageMeter()

    targets = np.array([])
    preds = np.array([])
    for i, (inp, target) in enumerate(val_loader):
        inp = inp.cuda(async=True)
        target_cuda = target.cuda(async=True)
        input_var = torch.autograd.Variable(inp, volatile=True)
        target_var = torch.autograd.Variable(target_cuda, volatile=True)

        output = model(input_var)
        loss = criterion(output, target_var)
        prec1 = pytorch_accuracy(output.data, target_cuda)
        targets = np.append(targets, target.cpu().numpy())
        preds = np.append(preds, np.argmax(output.data.cpu().numpy(), axis=1))

        losses.update(loss.data[0], inp.size(0))
        top1_acc.update(prec1[0][0], inp.size(0))
    #top1_f1 = f1_score(targets, preds, average='binary')*100.0
    return losses.avg, top1_acc.avg #top1_f1

# TODO: should possibly make this into a class, a la torchsample
def trainer(big_model, small_model, T, criterion, optimizer, scheduler,
            loaders, 
            nb_epochs=50,
            patience=5, save_every=5,
            model_ckpt_name='model-epoch{epoch:02d}.t7', model_best_name='model.best.t7',
            scheduler_arg='loss'): # 'loss' or 'epoch'
    train_loader, val_loader = loaders

    best_loss = (float('Inf'), -1)
    best_acc = (0, -1)

    last_update = -1
    pbar = tqdm.tqdm(range(nb_epochs))
    for epoch in pbar:
        if scheduler_arg == 'epoch':
            scheduler.step(epoch)
        train_epoch(train_loader, big_model, small_model, T, criterion, optimizer, epoch)

        val_loss, val_acc = val_epoch(val_loader, small_model, criterion)
        pbar.set_description('val loss: %2.4f, val acc: %2.1f' % (val_loss, val_acc))

        if val_loss < best_loss[0]:
            best_loss = (val_loss, epoch)
            last_update = epoch
        if val_acc > best_acc[0]:
            best_acc = (val_acc, epoch)
            last_update = epoch
            torch.save({'state_dict': small_model.state_dict(), 'acc': val_acc}, model_best_name)
            
        if epoch % save_every == 0:
            fname = model_ckpt_name.format(epoch=epoch)
            torch.save({'state_dict': small_model.state_dict()}, fname)

        if epoch - last_update > patience:
            break

        if scheduler_arg == 'loss':
            scheduler.step(val_loss)

    print 'Best loss: ' + str(best_loss)
    print 'Best acc: ' + str(best_acc)
    return best_acc

def get_datasets(CLASS_NAMES=None,
                 normalize=None, RESOL=224,
                 batch_size=32, num_workers=16,
                 use_rotate=False):
    #NB_CLASSES = len(train_fnames)
    #assert NB_CLASSES == len(val_fnames)
    '''
    if normalize is None:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    if CLASS_NAMES is None:
        CLASS_NAMES = map(str, range(NB_CLASSES))

    train_imgs = sum(map(lambda x: zip(x[0], itertools.repeat(x[1])),
                         zip(train_fnames, range(NB_CLASSES))), [])
    val_imgs = sum(map(lambda x: zip(x[0], itertools.repeat(x[1])),
                       zip(val_fnames, range(NB_CLASSES))), [])

    if use_rotate:
        rotation = [RandomRotate(20)]
    else:
        rotation = []

    train_dataset = ImageList(
            CLASS_NAMES, train_imgs,
            transforms.Compose(rotation + [
                    transforms.RandomSizedCrop(RESOL),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
            ]))
    val_dataset = ImageList(
            CLASS_NAMES, val_imgs,
            transforms.Compose([
                    transforms.Scale(int(256.0 / 224.0 * RESOL)),
                    transforms.CenterCrop(RESOL),
                    transforms.ToTensor(),
                    normalize,
            ]))

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, num_workers=num_workers,
            shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size, num_workers=num_workers,
            shuffle=False, pin_memory=True)
    '''
    train_transform = transforms.Compose([
        transforms.RandomSizedCrop(RESOL),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
        ])
    val_transform = transforms.Compose([
        transforms.Scale(int(256.0 / 224.0 * RESOL)),
        transforms.CenterCrop(RESOL),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
        ])

    train_dataset = torchvision.datasets.ImageFolder(TRAIN_PATH, train_transform)
    val_dataset = torchvision.datasets.ImageFolder(VAL_PATH, val_transform)
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, num_workers=num_workers,
            shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size, num_workers=num_workers,
            shuffle=False, pin_memory=True)
    return train_loader, val_loader
