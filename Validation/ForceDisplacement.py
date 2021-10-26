# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 18:41:24 2020

@author: Olivier Lampron
"""

import numpy as np
import matplotlib.pyplot as plt

data1 = np.loadtxt('gerasimov2019_VD_AT1.txt',)
data2 = np.loadtxt('RF.txt',)


data2 = np.concatenate((np.zeros((1,3)), data2),axis = 0)


plt.figure(1)
plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'font.size': 15})
plt.rcParams["legend.loc"] = 'upper left'
plt.plot(data1[:,0],data1[:,1], color='#364b9a', linewidth=2.5, label=r'AT1 V-D - Gerasimov \& De Lorenzis 2019', alpha = 1)
plt.plot(data2[:,0]*0.012,-data2[:,1], color='#364b9a', linestyle = (0, (1, 1)), linewidth=2.5, label=r'AT1 V-D - juliaPF', alpha = 1)
plt.legend(frameon=False)
plt.xlabel('Displacement [mm]')
plt.ylabel('Force [N]')
plt.grid(True,'major')
plt.ylim((0.0, 500.0))
plt.xlim((0.0, 0.012))
plt.tight_layout()
plt.show()
plt.savefig('ForceDeplacement.pdf')
