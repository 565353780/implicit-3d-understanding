#!/usr/bin/env python
# -*- coding: utf-8 -*-

import horovod.torch as hvd

hvd.init()

print("Hello! I'm ", hvd.rank())

