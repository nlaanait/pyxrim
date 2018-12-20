# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 20:42:02 2016

@author: nouamanelaanait
"""

import xrayutilities as xu
import numpy as np


#%%
def Pnma(a, b, c):
    #create orthorhombic unit cell
    l = xu.materials.Lattice([a, 0, 0], [0, b, 0], [0, 0, c])
    return l


latticeConstants=[3.905, 3.905, 3.905]
STO = xu.materials.Material("SrTiO3", Pnma(*latticeConstants))

#%%
# S2-D2 geometry
# start with outermost circle:
# S2: phi, theta
# D2: nu, del
# with beam direction along +z and surface normal along +y
S2 = ('y+', 'x-')
D2 = ('y-', 'x-')
beamDir = (0,0,1)
qconv=xu.QConversion(S2, D2, beamDir)
#%%
# Specify in-plane direction [1,0,0] and normal direction [0,0,1] surface normal
hxrd = xu.HXRD(STO.Q(0, 1, 0), STO.Q(0, 0, 1), qconv=qconv)
hxrd.energy = 10000
#%%
hkl=(0, 0, 1)
q_material = STO.Q(hkl)
q_laboratory = hxrd.Transform(q_material) # transform

print('SrTiO3: hkl ', hkl, ' qvec ', np.round(q_material, 5))
print('Lattice plane distance: %.4f' % STO.planeDistance(hkl))

#%%
#### determine the goniometer angles with the correct geometry restrictions
# tell bounds of angles / (min,max) pair or fixed value for all motors
# maximum of three free motors! 
# phi, eta, nu, del
bounds = (0, (-1, 90), (-1,90), (-1, 90))
ang,qerror,errcode = xu.Q2AngFit(q_laboratory, hxrd, bounds)
print('err %d (%.3g) \n angles %s' % (errcode, qerror, str(np.round(ang, 5))))
#%%
# check that qerror is small!!
print('sanity check with back-transformation (hkl): ',
      np.round(hxrd.Ang2HKL(*ang,mat=STO),5))
#%% to calculate off-specular:
hxrd = xu.HXRD(STO.Q(0, 1, 0), STO.Q(1, 0, 1), qconv=qconv)
hkl=(1, 0, 3)
q_material = STO.Q(hkl)
q_laboratory = hxrd.Transform(q_material) # transform

print('SrTiO3: hkl ', hkl, ' qvec ', np.round(q_material, 5))
print('Lattice plane distance: %.4f' % STO.planeDistance(hkl))
#%%
bounds = ((-180, 180), (0,40), (-1, 90), (-1, 90))
ang,qerror,errcode = xu.Q2AngFit(q_laboratory, hxrd, bounds)
print('err %d (%.3g) \n angles %s' % (errcode, qerror, str(np.round(ang, 5))))

#%%
# check that qerror is small!!
print('sanity check with back-transformation (hkl): ',
      np.round(hxrd.Ang2HKL(*ang,mat=STO),5))








      