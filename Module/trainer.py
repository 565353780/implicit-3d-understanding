#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import wandb

import argparse
from configs.config_utils import CONFIG

from Config.configs import LDIF_CONFIG

from Method.paths import getModelPath
from Method.dataloaders import LDIF_dataloader
from Method.models import LDIF

class Trainer(object):
    def __init__(self):
        self.config = {}
        self.device = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.model = None
        return

    def loadConfig(self, config):
        self.config = config

        parser = argparse.ArgumentParser('test1.')
        parser.add_argument('--config', type=str, default='configs/ldif.yaml')
        parser.add_argument('--mode', type=str, default='train')
        self.cfg = CONFIG(parser)
        return True

    def initWandb(self):
        resume = True
        name = "test_Trainer"
        id = self.config['log']['path'].split('/')[-2]

        wandb.init(project="LDIF_Train",
                   config=self.config,
                   dir=self.config['log']['path'],
                   name=name, id=id, resume=resume)
        wandb.summary['pid'] = os.getpid()
        wandb.summary['ppid'] = os.getppid()
        return True

    def loadDevice(self):
        device = self.config['device']['device']
        self.device = torch.device(device)
        return True

    def loadDataset(self, dataloader):
        self.train_dataloader = dataloader(self.cfg.config, 'train')
        self.test_dataloader = dataloader(self.cfg.config, 'val')
        return True

    def loadModel(self, model):
        self.model = model(self.cfg).to(self.device)

        model_path = getModelPath(self.config)
        if model_path is None:
            print("[INFO][Trainer::loadModel]")
            print("\t trained model not found, start training from 0 epoch...")
            return True

        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
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
        return True

def demo():
    config = LDIF_CONFIG
    dataloader = LDIF_dataloader
    model = LDIF

    trainer = Trainer()
    trainer.initEnv(config, dataloader, model)
    return True

if __name__ == "__main__":
    demo()

