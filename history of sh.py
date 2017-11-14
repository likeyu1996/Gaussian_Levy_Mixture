# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 20:05:59 2017

@author: dell
"""

import tushare as ts
h_sh=ts.get_hist_data('sh')
h_sh.to_csv('his of sh.csv')