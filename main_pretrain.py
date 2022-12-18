"""
@author: Yanzuo Lu
@email:  luyz5@mail2.sysu.edu.cn
"""

import argparse
import datetime
import io
import os
import random
import time
from contextlib import contextmanager

import numpy as np
import timm.optim.optim_factory as optim_factory
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.distributed.elastic.utils.data
import torch.nn.parallel
import torch.optim
import torch.utils.data
from timm.utils import AverageMeter
from torch.utils.tensorboard import SummaryWriter

import datasets
import models
import utils.transforms
from conf_pretrain import _C as cfg
from utils.logger import setup_logger
from utils.lr_scheduler import adjust_learning_rate
from utils.pos_embed import interpolate_pos_embed
from utils.scaler import NativeScalerWithGradNormCount


def main(cfg):
    seed = cfg.SEED + RANK
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device_id = LOCAL_RANK
    torch.cuda.set_device(device_id)
    cudnn.benchmark = True
    logger.info(f'set cuda device = {device_id}')

    dist.init_process_group(backend='nccl')

    model, optimizer, scaler = initialize_model(cfg, device_id)
    train_loader = initialize_data_loader(cfg)
    summary_writer = SummaryWriter(cfg.OUTPUT_DIR) if RANK == 0 else None

    model_without_ddp = model.module
    state = load_checkpoint(cfg, model_without_ddp, optimizer, scaler)

    start_epoch = state.epoch + 1
    logger.info(f'start_epoch: {start_epoch}')

    accum_iter = cfg.ENGINE.ACCUM_ITER
    epochs = cfg.ENGINE.EPOCHS
    warmup_epochs = cfg.ENGINE.WARMUP_EPOCHS
    lr = cfg.OPTIMIZER.BASE_LR * cfg.ENGINE.BATCH_SIZE * WORLD_SIZE / 256
    mask_ratio = cfg.MODEL.MASK_RATIO
    print_freq = cfg.PRINT_FREQ

    start_time = time.time()
    for epoch in range(start_epoch, cfg.ENGINE.EPOCHS):
        state.epoch = epoch
        train_loader.batch_sampler.sampler.set_epoch(epoch)

        train_one_epoch(train_loader, model, optimizer, scaler, epoch, device_id, accum_iter,
                        epochs, warmup_epochs, lr, mask_ratio, print_freq, summary_writer)

        if device_id == 0:
            save_checkpoint(state, cfg.OUTPUT_DIR, epoch, cfg.CHECKPOINT_OVERWRITE)

    total_time = time.time() - start_time
    logger.info(f'training time {datetime.timedelta(seconds=int(total_time))}')
    if summary_writer:
        summary_writer.close()


def train_one_epoch(train_loader, model, optimizer, scaler, epoch, device_id, accum_iter,
                    epochs, warmup_epochs, lr, mask_ratio, print_freq, summary_writer):
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    model_without_ddp = model.module
    num_steps = len(train_loader)

    start = time.time()
    end = time.time()
    for i, (img,) in enumerate(train_loader):
        if i % accum_iter == 0:
            lr_decayed = adjust_learning_rate(optimizer, i / num_steps + epoch, epochs, warmup_epochs, lr)

        img = img.cuda(device_id, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, loss_value, _, _ = model(img, mask_ratio=mask_ratio)
        loss_item = loss.item()

        loss = loss / accum_iter
        scaler(loss, optimizer, parameters=model.parameters(), update_grad=(i + 1) % accum_iter == 0)
        if (i + 1) % accum_iter == 0:
            optimizer.zero_grad()
            if hasattr(model_without_ddp, 'ema_update'):
                model_without_ddp.ema_update()

        torch.cuda.synchronize()

        losses.update(loss_item)
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            etas = batch_time.avg * (num_steps - i)
            logger.info(
                f'Train [{epoch}/{epochs}]({i}/{num_steps})  '
                f'Time {batch_time.val:.4f}({batch_time.avg:.4f})  '
                f'Loss {losses.val:.4f}({losses.avg:.4f})  '
                f'Lr {lr_decayed:.4e}  '
                f'Eta {datetime.timedelta(seconds=int(etas))}'
            )

        loss_value_reduce = torch.tensor(loss_value).cuda()
        dist.all_reduce(loss_value_reduce)
        loss_value_reduce_mean = loss_value_reduce / WORLD_SIZE
        loss_value = loss_value_reduce_mean.tolist()
        if summary_writer:
            summary_writer.add_scalar('Loss', loss_value[0], epoch * num_steps + i)
            summary_writer.add_scalar('Lr', lr_decayed, epoch * num_steps + i)
            for index, v in enumerate(loss_value[1:]):
                summary_writer.add_scalar(f'Loss{index+1}', v, epoch * num_steps + i)

    epoch_time = time.time() - start
    logger.info(f'EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}')


def initialize_model(cfg, device_id):
    logger.info(f'creating model: {cfg.MODEL.NAME}')
    model = models.__dict__[cfg.MODEL.NAME](cfg)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(device_id)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id])

    model_without_ddp = model.module
    param_groups = optim_factory.add_weight_decay(model_without_ddp, cfg.OPTIMIZER.WEIGHT_DECAY)
    lr = cfg.OPTIMIZER.BASE_LR * cfg.ENGINE.BATCH_SIZE * WORLD_SIZE / 256
    optimizer = torch.optim.AdamW(param_groups, lr, betas=cfg.OPTIMIZER.BETAS)
    scaler = NativeScalerWithGradNormCount()
    return model, optimizer, scaler


def initialize_data_loader(cfg):
    transform = utils.transforms.__dict__[cfg.INPUT.TRANSFORM](cfg)
    train_dataset = datasets.__dict__[cfg.DATASET.NAME](cfg, transform)
    train_sampler = torch.distributed.elastic.utils.data.ElasticDistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.ENGINE.BATCH_SIZE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        sampler=train_sampler
    )
    return train_loader


class State:
    def __init__(self, arch, model_without_ddp, optimizer, scaler):
        self.epoch = -1
        self.arch = arch
        self.model_without_ddp = model_without_ddp
        self.optimizer = optimizer
        self.scaler = scaler

    def capture_snapshot(self):
        return {
            'epoch': self.epoch,
            'arch': self.arch,
            'model': self.model_without_ddp.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict()
        }

    def apply_snapshot(self, obj):
        msg = self.model_without_ddp.load_state_dict(obj['model'], strict=False)
        if 'arch' in obj.keys() and self.arch == obj['arch']:
            self.epoch = obj['epoch']
            self.optimizer.load_state_dict(obj['optimizer'])
            self.scaler.load_state_dict(obj['scaler'])
        return msg

    def save(self, f):
        torch.save(self.capture_snapshot(), f)

    def load(self, f, pos_embed_size):
        snapshot = torch.load(f, map_location='cpu')
        interpolate_pos_embed(self.model_without_ddp, snapshot['model'], pos_embed_size, 'pos_embed')
        interpolate_pos_embed(self.model_without_ddp, snapshot['model'], pos_embed_size, 'decoder_pos_embed')
        msg = self.apply_snapshot(snapshot)
        logger.info(msg)


def load_checkpoint(cfg, model_without_ddp, optimizer, scaler):
    state = State(cfg.MODEL.NAME, model_without_ddp, optimizer, scaler)

    if os.path.isfile(cfg.MODEL.CHECKPOINT_PATH):
        logger.info(f'loading checkpoint file: {cfg.MODEL.CHECKPOINT_PATH}')
        state.load(cfg.MODEL.CHECKPOINT_PATH, cfg.MODEL.CHECKPOINT_POS_EMBED_SIZE)
        logger.info(f'loaded checkpoint file: {cfg.MODEL.CHECKPOINT_PATH}')

    with tmp_process_group(backend='gloo') as pg:
        rank = dist.get_rank(group=pg)

        epochs = torch.zeros(dist.get_world_size(group=pg), dtype=torch.int32)
        epochs[rank] = state.epoch
        dist.all_reduce(epochs, op=dist.ReduceOp.SUM, group=pg)
        t_max_epoch, t_max_rank = torch.max(epochs, dim=0)
        max_epoch = t_max_epoch.item()
        max_rank = t_max_rank.item()

        if max_epoch == -1:
            logger.info('no workers have checkpoints, starting from epoch 0')
            return state

        logger.info(f'using checkpoint from rank: {max_rank}, max_epoch: {max_epoch}')

        with io.BytesIO() as f:
            torch.save(state.capture_snapshot(), f)
            raw_blob = np.frombuffer(f.getvalue(), dtype=np.uint8)

        blob_len = torch.tensor(len(raw_blob))
        dist.broadcast(blob_len, src=max_rank, group=pg)
        logger.info(f'checkpoint broadcast size is: {blob_len}')

        if rank != max_rank:
            blob = torch.zeros(blob_len.item(), dtype=torch.uint8)
        else:
            blob = torch.as_tensor(raw_blob, dtype=torch.uint8)

        dist.broadcast(blob, src=max_rank, group=pg)
        logger.info('done broadcasting checkpoint')

        if rank != max_rank:
            with io.BytesIO(blob.numpy()) as f:
                snapshot = torch.load(f, map_location='cpu')
            _ = state.apply_snapshot(snapshot)

        dist.barrier(group=pg)

    logger.info('done restoring from previous checkpoint')
    return state


@contextmanager
def tmp_process_group(backend):
    cpu_pg = dist.new_group(backend=backend)
    try:
        yield cpu_pg
    finally:
        dist.destroy_process_group(cpu_pg)


def save_checkpoint(state, output_dir, epoch, checkpoint_overwrite):
    if checkpoint_overwrite:
        filename = os.path.join(output_dir, 'checkpoint.pth')
    else:
        filename = os.path.join(output_dir, 'checkpoint_%04d.pth' % epoch)
    checkpoint_dir = os.path.dirname(filename)
    os.makedirs(checkpoint_dir, exist_ok=True)

    tmp_filename = filename + '.tmp'
    torch.save(state.capture_snapshot(), tmp_filename)
    os.rename(tmp_filename, filename)
    logger.info(f'saved checkpoint for epoch {state.epoch} as {filename}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Antelope pre-training')
    parser.add_argument('--config_file', default='', help='path to config file', type=str)
    parser.add_argument('opts', help='modify config options using the command-line', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    RANK = int(os.environ['RANK'])
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logger = setup_logger(cfg.OUTPUT_DIR, LOCAL_RANK, cfg.MODEL.NAME)
    logger.info(f'running with config:\n{str(cfg)}')

    main(cfg)
