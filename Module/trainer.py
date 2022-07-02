#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import wandb

import torch
from datetime import datetime, timedelta
from collections import defaultdict
from torch.optim import lr_scheduler

from Config.configs import LDIF_CONFIG

from Method.paths import getModelPath
from Method.dataloaders import LDIF_dataloader
from Method.models import LDIF
from Method.optimizers import load_optimizer, load_scheduler

class AverageMeter(object):
    def __init__(self):
        self.reset()
        return

    def reset(self):
        self.val = []
        self.avg = 0
        self.sum = 0
        self.count = 0
        return True

    def update(self, val, n=1):
        if not isinstance(val, list):
            self.sum += val * n
            self.count += n
            self.val += [val] * n
        else:
            self.sum += sum(val)
            self.count += len(val)
            self.val += val
        self.avg = self.sum / self.count
        return True

class LossRecorder(object):
    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self.loss_recorder = {}
        return

    def update_loss(self, loss_dict):
        for key, item in loss_dict.items():
            if key not in self.loss_recorder:
                self.loss_recorder[key] = AverageMeter()
            self.loss_recorder[key].update(item, self.batch_size)
        return True

class Trainer(object):
    def __init__(self):
        self.config = {}
        self.device = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        return

    def loadConfig(self, config):
        self.config = config

        log_dict = self.config['log']
        if 'resume_path' in log_dict.keys():
            resume_path = log_dict['resume_path']
            if resume_path[-1] != "/":
                self.config['log']['resume_path'] += "/"

        if 'path' in log_dict.keys():
            path = log_dict['path']
            if path[-1] != "/":
                self.config['log']['path'] += "/"
            os.makedirs(path, exist_ok=True)
            name = log_dict['name']
            log_save_path = path + name + "/"
            os.makedirs(log_save_path, exist_ok=True)
        return True

    def initWandb(self):
        resume = True
        log_dict = self.config['log']

        id = log_dict['path'].split('/')[-2]

        wandb.init(project=log_dict['project'],
                   config=self.config,
                   dir=log_dict['path'] + log_dict['name'] + "/",
                   name=log_dict['name'],
                   id=id, resume=resume)
        wandb.summary['pid'] = os.getpid()
        wandb.summary['ppid'] = os.getppid()
        return True

    def loadDevice(self):
        device = self.config['device']['device']
        self.device = torch.device(device)
        return True

    def loadDataset(self, dataloader):
        self.train_dataloader = dataloader(self.config, 'train')
        self.test_dataloader = dataloader(self.config, 'val')
        return True

    def loadModel(self, model):
        self.model = model(self.config, 'train')

        model_path = getModelPath(self.config)
        if model_path is None:
            print("[INFO][Trainer::loadModel]")
            print("\t trained model not found, start training from 0 epoch...")
        else:
            state_dict = torch.load(model_path)
            #  self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        wandb.watch(self.model, log=None)
        return True

    def loadOptimizer(self):
        self.optimizer = load_optimizer(self.config, self.model)
        self.scheduler = load_scheduler(self.config, self.optimizer)
        return True

    def initEnv(self, config, dataloader, model):
        if not self.loadConfig(config):
            print("[ERROR][Trainer::initEnv]")
            print("\t loadConfig failed!")
            return False
        if not self.initWandb():
            print("[ERROR][Trainer::initEnv]")
            print("\t initWandb failed!")
            return False
        if not self.loadDevice():
            print("[ERROR][Trainer::initEnv]")
            print("\t loadDevice failed!")
            return False
        if not self.loadDataset(dataloader):
            print("[ERROR][Trainer::initEnv]")
            print("\t loadDevice failed!")
            return False
        if not self.loadModel(model):
            print("[ERROR][Trainer::initEnv]")
            print("\t loadModel failed!")
            return False
        if not self.loadOptimizer():
            print("[ERROR][Trainer::initEnv]")
            print("\t loadOptimizer failed!")
            return False
        return True

    def to_device(self, data):
        device = self.device
        ndata = {}
        for k, v in data.items():
            if type(v) is torch.Tensor and v.dtype is torch.float32:
                ndata[k] = v.to(device)
            else:
                ndata[k] = v
        return ndata

    def compute_loss(self, data):
        data = self.to_device(data)

        est_data = self.model(data)

        loss = self.model.loss(est_data, data)
        return loss

    def train_step(self, data):
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        if loss['total'].requires_grad:
            loss['total'].backward()
            self.optimizer.step()

        loss['total'] = loss['total'].item()
        return loss

    def eval_step(self, data):
        loss = self.compute_loss(data)
        loss['total'] = loss['total'].item()
        return loss

    def show_lr(self):
        lrs = [self.optimizer.param_groups[i]['lr'] for i in range(len(self.optimizer.param_groups))]
        print('Current learning rates are: ' + str(lrs) + '.')
        return True

    def save(self, suffix=None):
        print("start save")
        outdict = {}
        for k, v in self.config.items():
            if hasattr(v, 'state_dict'):
                print("====")
                print("====")
                print(k)
                print("====")
                print(v)
                print("====")
                print("====")
                outdict[k] = v.state_dict()
            else:
                outdict[k] = v
        print("end save")

        if not suffix:
            filename = 'model_last.pth'
        else:
            filename = 'model_last.pth'.replace('last', suffix)
        save_path = self.config['log']['path'] + filename
        torch.save(outdict, save_path)
        return True

    def train_epoch(self, epoch, dataloaders, step):
        for phase in ['train', 'val']:
            dataloader = dataloaders[phase]
            batch_size = self.config[phase]['batch_size']
            loss_recorder = LossRecorder(batch_size)
            self.model.train(phase == 'train')
            self.model.set_mode()
            print('-' * 10)
            print('Switch Phase to %s.' % (phase))
            print('-'*10)
            for iter, data in enumerate(dataloader):
                if phase == 'train':
                    loss = self.train_step(data)
                else:
                    loss = self.eval_step(data)

                loss_recorder.update_loss(loss)

                if ((iter + 1) % self.config['log']['print_step']) == 0:
                    pretty_loss = [f'{k}: {v:.3f}' for k, v in loss.items()]
                    print('Process: Phase: ' + phase + \
                          '. Epoch ' + str(epoch) + ': ' + str(iter + 1) + '/' + str(len(dataloader)) + \
                          '. Current loss: {' + ', '.join(pretty_loss) + '}.')
                    if phase == 'train':
                        loss = {f'train_{k}': v for k, v in loss.items()}
                        wandb.log(loss, step=step)
                        wandb.log({'epoch': epoch}, step=step)

                if phase == 'train':
                    step += 1

            print('=' * 10)
            for loss_name, loss_value in loss_recorder.loss_recorder.items():
                print('Currently the last %s loss (%s) is: %f' % (phase, loss_name, loss_value.avg))
            print('=' * 10)
        return loss_recorder.loss_recorder, step

    def start_train(self):
        min_eval_loss = 1e8
        epoch = 0
        step = 0

        start_epoch = self.scheduler.last_epoch
        if isinstance(self.scheduler, (lr_scheduler.StepLR, lr_scheduler.MultiStepLR)):
            start_epoch -= 1
        total_epochs = self.config['train']['epochs']

        dataloaders = {'train': self.train_dataloader, 'val': self.test_dataloader}

        for epoch in range(start_epoch, total_epochs):
            print('-' * 10)
            print('Epoch (%d/%s):' % (epoch + 1, total_epochs))
            self.show_lr()

            eval_loss_recorder, step = self.train_epoch(epoch + 1, dataloaders, step)

            eval_loss = eval_loss_recorder['total'].avg
            if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(eval_loss)
            elif isinstance(self.scheduler, (lr_scheduler.StepLR, lr_scheduler.MultiStepLR)):
                self.scheduler.step()
            else:
                raise NotImplementedError
            loss = {f'test_{k}': v.avg for k, v in eval_loss_recorder.items()}
            wandb.log(loss, step=step)
            wandb.log({f'lr{i}': g['lr'] for i, g in enumerate(self.optimizer.param_groups)}, step=step)
            wandb.log({'epoch': epoch + 1}, step=step)

            print('Epoch (' + str(epoch + 1) + '/' + str(total_epochs) + ').')

            # save checkpoint
            if self.config['log'].get('save_checkpoint', True):
                self.save('last')
            print('Saved the latest checkpoint.')
            if epoch==-1 or eval_loss<min_eval_loss:
                if self.config['log'].get('save_checkpoint', True):
                    self.save('best')
                min_eval_loss = eval_loss
                print('Saved the best checkpoint.')
                print('=' * 10)
                for loss_name, loss_value in eval_loss_recorder.items():
                    wandb.summary[f'best_test_{loss_name}'] = loss_value.avg
                    print('Currently the best val loss (%s) is: %f' % (loss_name, loss_value.avg))
                print('=' * 10)
        return True

    def train(self):
        num_params = sum(p.numel() for p in self.model.parameters())
        print("num_params =")
        print(num_params)
        wandb.summary['num_params'] = num_params

        self.start_train()
        return True

def demo():
    config = LDIF_CONFIG
    dataloader = LDIF_dataloader
    model = LDIF

    trainer = Trainer()
    trainer.initEnv(config, dataloader, model)
    trainer.train()
    return True

if __name__ == "__main__":
    demo()

