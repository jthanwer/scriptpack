#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 16:12:19 2018

@author: jthanwer
"""
import numpy as np
import xarray as xr


def get_aire_byphi(year=2017, month='01'):
    phy_path = '/home/satellites1/fcheval/LMDZ5/phystoke.an{}.m{}.nc'.format(year, month)
    aire = xr.open_dataset(phy_path)['aire']
    aire = np.sum(aire.values)
    return aire

# M_1H = 1.007825  # g/mol
# M_2H = 2.014102  # g/mol
# M_12C = 12.0000  # g/mol
# M_13C = 13.0034  # g/mol


R_PDB = 0.0112372  # Pee Dee Belemnite Standard
R_PDB_2 = 0.01117960  # Pee Dee Belemnite Standard 2nd value
R_VSMOW = 155.76e-6  # Vienna Standard Mean Ocean Water

M_AIR = 28.966  # g/mol
M_CH4 = 16.0425  # g/mol
M_CH3D = 17.0376  # g/mol
M_C13 = 17.0347  # g/mol
M_C12 = 16.0312  # g/mol
M_OH = 17.0080  # g/mol
M_Cl = 35.453  # g/mol

GRAV = 9.80665  # m/s2
R_GAZ = 8.314  # J.K-1.mol-1
KB = 1.3806e-23  # m2.kg.s-2.K-1
N_AVO = 6.022e23  # mol-1

R_TERRE = 6371000  # m
PI = np.pi
# AIRE = get_aire_byphi()

