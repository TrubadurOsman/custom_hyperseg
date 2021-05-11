import os
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm
from utils.utils import set_device, set_seed, str2int, save_checkpoint, get_arch
from utils.img_utils import make_grid
from utils.obj_factory import obj_factory
from utils.seg_utils import blend_seg
from datasets.seg_transforms import Compose
from utils.tensorboard_logger import TensorBoardLogger
import time
from dataset import get_dataloader


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
general = parser.add_argument_group('general')
general.add_argument('exp_dir', metavar='DIR',
                     help='path to experiment directory')
general.add_argument('-r', '--resume', metavar='PATH',
                     help='path to resume directory or checkpoint')
general.add_argument('-se', '--start-epoch', metavar='N',
                     help='manual epoch number (useful on restarts)')
general.add_argument('-e', '--epochs', default=90, type=int, metavar='N',
                     help='number of total epochs to run')
general.add_argument('-ti', '--train_iterations', metavar='N',
                     help='number of train iterations per resolution to run')
general.add_argument('-vi', '--val_iterations', metavar='N',
                     help='number of validation iterations per resolution to run')
general.add_argument('--gpus', nargs='+', type=int, metavar='N',
                     help='list of gpu ids to use (default: all)')
general.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                     help='number of data loading workers (default: 4)')
general.add_argument('-b', '--batch-size', default=(4,), type=int, metavar='N',
                     help='mini-batch size (default: 64)')
general.add_argument('--seed', type=int, metavar='N',
                     help='random seed')
general.add_argument('-lf', '--log_freq', default=20, type=int, metavar='N',
                     help='number of steps between each loss plot')
general.add_argument('-lmr', '--log_max_res', default=512, type=int, metavar='N',
                     help='maximum resolution of logged images (larger axis)')

data = parser.add_argument_group('data')
data.add_argument('-td', '--train_dataset', default='torchvision.datasets.voc.VOCSegmentation',
                  help='train dataset object')
data.add_argument('-vd', '--val_dataset',
                  help='val dataset object')
data.add_argument('-tit', '--train_img_transforms', nargs='+',
                  help='train image transforms')
data.add_argument('-vit', '--val_img_transforms', nargs='+',
                  help='validation image transforms')
data.add_argument('-tt', '--tensor_transforms', nargs='+', help='tensor transforms',
                  default=('seg_transforms.ToTensor', 'seg_transforms.Normalize'))

training = parser.add_argument_group('training')
training.add_argument('-o', '--optimizer', default='optim.Adam(betas=(0.5,0.999))',
                      help='network\'s optimizer object')
training.add_argument('-s', '--scheduler', default='lr_scheduler.StepLR(step_size=10,gamma=0.6)',
                      help='scheduler object')
training.add_argument('-c', '--criterion', default='nn.CrossEntropyLoss(ignore_index=255)',
                      help='criterion object')
training.add_argument('-m', '--model', default='fsgan.models.simple_unet.UNet(n_classes=3,feature_scale=1)',
                      help='model object')
training.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',
                      help='use pre-trained model')
training.add_argument('-be', '--benchmark', default='hyperseg.utils.seg_utils.IOUBenchmark',
                      help='benchmark object')
training.add_argument('-bs', '--batch_scheduler', action='store_true',
                      help='if True, the learning rate will be scheduled after each batch iteration')
d = parser.get_default


def main(
    # General arguments
    exp_dir, resume=d('resume'), start_epoch=d('start_epoch'), epochs=d('epochs'),
    train_iterations=d('train_iterations'), val_iterations=d('val_iterations'), gpus=d('gpus'), workers=d('workers'),
    batch_size=d('batch_size'), seed=d('seed'), log_freq=d('log_freq'), log_max_res=d('log_max_res'),

    # Data arguments
    train_dataset=d('train_dataset'), val_dataset=d('val_dataset'), train_img_transforms=d('train_img_transforms'),
    val_img_transforms=d('val_img_transforms'), tensor_transforms=d('tensor_transforms'),

    # Training arguments
    optimizer=d('optimizer'), scheduler=d('scheduler'), criterion=d('criterion'), model=d('model'),
    pretrained=d('pretrained'), benchmark=d('benchmark'), hard_negative_mining=d('hard_negative_mining'),
    batch_scheduler=d('batch_scheduler'),model_path=None,load = False
):
    def train_model(model, dataloaders, criterion, optimizer,scheduler, num_epochs=25):
        since = time.time()
        val_mean_iou_history = []
        best_mean_iou = 0.0
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_mean_iou = 0

                batch_counter =0
                for sample in tqdm(iter(dataloaders[phase])):

                    inputs = sample['image'].to(device)
                    labels = sample['mask'].to(device)

                    label = torch.argmax(labels, dim=1)

                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):

                        outputs = model(inputs)
                        # outputs['out'] = outputs['out'].to(device)
                        loss = criterion(outputs, label)
                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()


                    running_loss += loss.item() * inputs.size(0)
                    iou =  mean_iou(torch.argmax(outputs, 1), label).item()
                    running_mean_iou += iou
                    batch_counter +=1
                    if(batch_counter%10 ==0 and phase == 'train'):
                        print('{}  Loss: {:.4f}  mIoU: {:.4f} lr: '.format(phase, loss, iou,scheduler.get_last_lr()[0]))

                if not os.path.exists(exp_dir+'/models'):
                    os.makedirs(exp_dir+'/models')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'scheduler':scheduler,
                },exp_dir+'/models/'+str(epoch)+'.pt')

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_mean_iou = running_mean_iou / len(dataloaders[phase])
                if phase == 'valid':
                    scheduler.step(epoch_loss)
                print('{} Loss: {:.4f} mIoU: {:.4f}'.format(phase, epoch_loss, epoch_mean_iou))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val mean IoU: {:4f}'.format(best_mean_iou))

    # Setup device
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Create model
    arch = get_arch(model, num_classes=8)
    model = obj_factory(model, num_classes=8).to(device)

    # Optimizer and scheduler
    optimizer = obj_factory(optimizer, model.parameters())
    scheduler = obj_factory(scheduler, optimizer)
    criterion = obj_factory(criterion).to(device)

    # Resume
    start_epoch = 0
    best_iou = 0.

    if load and os.path.isfile(model_path):
        print("=> loading checkpoint from '{}'".format(model_path))
        # model
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else start_epoch
        best_iou = checkpoint['best_iou'] if 'best_iou' in checkpoint else best_iou
        model.apply(init_weights)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        print("=> no checkpoint found at '{}'".format(model_path))
        if not pretrained:
            print("=> randomly initializing networks...")
            model.apply(init_weights)

    DATA_PATH = 'data/yamaha_v0/'
    MODEL_PATH = 'models/model_v0.1.pt'
    IMG_HEIGHT = 512
    IMG_WIDTH = 1024
    dataloaders = get_dataloader(DATA_PATH,
                                 batch_size=batch_size,
                                 resize_shape=(IMG_HEIGHT, IMG_WIDTH))

    train_model(model,dataloaders,criterion,optimizer,scheduler,num_epochs= epochs)

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)


# def limit_resolution(img, max_res=512, mode='bilinear'):
#     img_res = max(img.shape[-2:])
#     if img_res <= max_res:
#         return img
#
#     scale = max_res / img_res
#     if mode == 'nearest':
#         return F.interpolate(img.float(), scale_factor=scale, mode=mode).long()
#     else:
#         return F.interpolate(img, scale_factor=scale, mode=mode)

def mean_iou(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """ Calculates the mean IoU (Jaccard index) between
        two tensors. Shape expected to be same.
    Args:
        outputs: (torch.Tensor) the output of a model
        labels: (torch.Tensor) the ground truth labels
    Returns:
        float
    """
    outputs = outputs.byte()
    labels = labels.byte()
    intersection = torch.logical_and(labels, outputs)
    union = torch.logical_or(labels, outputs)
    iou_score = torch.sum(intersection) / torch.sum(union)
    return iou_score.mean()

if __name__ == "__main__":
    main(**vars(parser.parse_args()))
