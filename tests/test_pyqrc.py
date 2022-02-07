#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pytest
import math
from pyqrc import pyQRC as QRC
from .conftest import datapath

@pytest.mark.parametrize("path, nproc, mem", [
    ('Gaussian/acetaldehyde.log', 4,  '8GB'),
])
def test_QRC(path, nproc, mem):
    path = datapath(path)
    amplitude = 0.2
    route = None
    verbose = True
    suffix = 'QRC'
    freq, freq_num = None, None
    qrc = QRC.gen_qrc(path, amplitude, nproc, mem, route, verbose, suffix, freq, freq_num)
