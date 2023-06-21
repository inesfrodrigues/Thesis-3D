# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
import numpy as np

import matplotlib.pyplot as plt

torch.cuda.empty_cache()


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,
                    fp32=False):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        #targets = torch.unsqueeze(targets, 2)


        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # with torch.cuda.amp.autocast():
        #     outputs = model(samples)
        #     loss = criterion(samples, outputs, targets)
        with torch.cuda.amp.autocast(enabled=not fp32):
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.MSELoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        #target = torch.unsqueeze(target, 2)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target.long())

        # root mean square error
        #error = torch.sqrt(torch.sum(torch.square(output.reshape(N, D, H, W) - target), dim = (1,2,3)))/torch.sqrt(torch.sum(torch.square(target), dim = (1,2,3)))*100
        error = torch.mean(torch.sqrt(torch.sum(torch.square(output - target), dim = (1,2)))/torch.sqrt(torch.sum(torch.square(target), dim = (1,2))+0.000001))*100
        #acc1, acc5 = accuracy(output, target.long(), topk=(1, 5)) 

        #print(error)
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['error'].update(error.item(), n=batch_size)
        #print(metric_logger.error)
        #metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    #dict_channel = {0: 'temperature', 1: 'salinity', 2: 'oxygen', 3: 'chla', 4: 'ppn'}
    directory = '/home/gpietrop/fig/images/0/'

    print('New image')
    #number_fig = len(target[0, :, 0, 0])  # number of levels of depth

    #for i in range(number_fig):
    cmap = plt.get_cmap('Greens')
    target2 = torch.squeeze(target, 1)
    target2 = target2.cpu().detach().numpy()
    plt.imshow(target2[1, :, :], cmap=cmap)
    #plt.title(dict_channel[3])
    plt.title('temp')
    plt.colorbar()
    plt.savefig(directory + "profondity_level_" + str(2) + ".png")
    plt.close()
            
    print('Plot done')

    directory = '/home/gpietrop/fig/output/0/'

    print('New output')
    #number_fig = len(output[0, 0, :, 0, 0])  # number of levels of depth

    #for i in range(number_fig):
    cmap = plt.get_cmap('Greens')
    output2 = output.cpu().detach().numpy()
    plt.imshow(output2[1, 0, :, :], cmap=cmap)
    #plt.title(dict_channel[3])
    plt.title('temp')
    plt.colorbar()
    plt.savefig(directory + "profondity_level_" + str(2) + ".png")
    plt.close()
            
    print('Plot done')
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    #print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #      .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    print('* error {top1.global_avg:.3f}, loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.error, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
