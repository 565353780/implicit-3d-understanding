#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from Config.configs import LDIF_CONFIG

from Method.paths import getModelPath
from Method.models import LDIF

from Module.base_loader import BaseLoader

class Detector(BaseLoader):
    def __init__(self):
        super(Detector, self).__init__()

        self.model = None
        return

    def loadModel(self, model):
        self.model = model(self.config, 'test')

        model_path = getModelPath(self.config)
        if model_path is None:
            print("[INFO][Detector::loadModel]")
            print("\t trained model not found!")
            return False

        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict['model'])
        self.model.to(self.device)
        return True

    def initEnv(self, config, model):
        if not self.loadConfig(config):
            print("[ERROR][Detector::initEnv]")
            print("\t loadConfig failed!")
            return False
        if not self.loadDevice():
            print("[ERROR][Detector::initEnv]")
            print("\t loadDevice failed!")
            return False
        if not self.loadModel(model):
            print("[ERROR][Detector::initEnv]")
            print("\t loadModel failed!")
            return False
        return True

    def detect(self, data):
        data = self.to_device(data)
        est_data = self.model(data)
        return est_data

def demo():
    config = LDIF_CONFIG
    model = LDIF

    detector = Detector()
    detector.initEnv(config, model)
    result = detector.detect(None)
    print(result)
    return True

if __name__ == "__main__":
    demo()

