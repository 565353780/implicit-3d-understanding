#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

def getModelPath(config):
    model_dict = config['model']
    if 'model_save_path' not in model_dict.keys():
        return None

    model_save_path = model_dict['model_save_path']
    if model_save_path[-1] != "/":
        model_save_path += "/"
    last_model_path = model_save_path + "model_last.pth"
    if not os.path.exists(last_model_path):
        return None
    return last_model_path

