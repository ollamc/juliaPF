# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 18:41:24 2020

@author: Olivier Lampron
"""

import numpy as np
import matplotlib.pyplot as plt

data1 = np.loadtxt('NCPU\RF.txt',)
data2 = np.loadtxt('SpectralSplit\RF.txt',)
#data3 = np.loadtxt('RF_QM_500.txt',)
#data4 = np.loadtxt('RF_QM_5000.txt',)
data4 = np.loadtxt('Miehe_Fig7a_WebPlotDigitizer.txt',)


data1 = np.concatenate((np.zeros((1,3)), data1),axis = 0)
data2 = np.concatenate((np.zeros((1,3)), data2),axis = 0)
#data3 = np.concatenate((np.zeros((1,3)), data3),axis = 0)
#data4 = np.concatenate((np.zeros((1,3)), data4),axis = 0)

#### staggered vs quasi-mono
plt.figure(1)
plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'font.size': 15})
plt.rcParams["legend.loc"] = 'upper left'
plt.plot(data1[:,0]*0.01,-data1[:,2], color='#f67e4b', linestyle = 'dashed', linewidth=2.5, label='No Split', alpha = 1)
#plt.plot(data3[:,0]*0.01,-data3[:,2], '--', color='blue',linewidth=2, label='Quasi-monolithic, n=500' )
plt.plot(data2[:,0]*0.01,-data2[:,2], color='0.3', linestyle = (0, (1, 1)), linewidth=2.5, label='Spectral Split', alpha = 1)
plt.plot(data4[:,0],data4[:,1]*1000, color='#364b9a', linestyle = 'solid', linewidth=2.5, label='Results from Miehe', alpha = 1)
plt.legend(frameon=False)
plt.xlabel('Displacement [mm]')
plt.ylabel('Force [N]')
plt.grid(True,'major')
plt.ylim((0.0, 800.0))
plt.xlim((0.0, 0.008))
plt.tight_layout()
plt.show()
plt.savefig('ComparisonWithMiehe.pdf')

