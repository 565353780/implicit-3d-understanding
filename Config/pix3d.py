#!/usr/bin/env python
# -*- coding: utf-8 -*-

class PIX3DConfig(object):
    def __init__(self):
        self.root_path = './data/pix3d/'
        self.train_split = self.root_path + 'splits/train.json'
        self.test_split = self.root_path + 'splits/test.json'
        self.metadata_path = self.root_path + 'metadata/'
        self.metadata_file = self.metadata_path + 'pix3d.json'
        self.classnames = ['misc',
                           'bed', 'bookcase', 'chair', 'desk', 'sofa',
                           'table', 'tool', 'wardrobe']
        return

