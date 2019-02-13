# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:41:38 2015

@author: nouamanelaanait
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May  2 17:46:54 2015

@author: nouamanelaanait
"""

# 
# The basics
import os
import sys
import numpy as np

# PyQT stuff
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui


def stackViewFunc(data, **kwargs):    
    ''' This makes a QWindow, fills it up with an ImageView object. Then returns both.
    Input: 
        data: 3d,2d np.ndarray.
        xvals: 1d np.ndarray, Optional
                x-axis Values of Roi slice through the scan, default slice numbers.
        winTitle: string, Optional 
                Title of QWindow, default = 'Window'.
        plotTitle: string, Optional
                Title of Roi plot window, default = 'Plot'.
        normalize: Boolean, Optional
                Normalize each frame of the stack, default False.
    Output:
        QtGui.QMainWindow and PyQtGraph.ImageView.
    '''
    xvals = kwargs.get('xvals',np.arange(0,data.shape[0]))    
    winTitle = kwargs.get('winTitle', 'Window')
    plotTitle = kwargs.get('plotTitle', 'Plot') 
    normalize = kwargs.get('normalize', False)
    
    win = QtGui.QMainWindow()
    win.resize(800,800)
    plt = pg.PlotItem()
    imv = pg.ImageView(view=plt)
    win.setCentralWidget(imv)
    win.setWindowTitle(winTitle)
    
    # Normalize the data
    if normalize:
        data = np.array([ (imp-imp.mean())/imp.mean() for imp in data ])
        data[data < 0] = 0
        data.astype('float32')
        
    # Display the data and assign each frame a value from scan
    imv.setImage(data, xvals=xvals, autoHistogramRange = True)


    # Customize roi plot windows
    pltWdgt = imv.getRoiPlot()
    plt = pltWdgt.getPlotItem()
    plt.setLabels(title = plotTitle)
    plt.showGrid(x = True, y = True, alpha = 1)
    pltdataitem = plt.listDataItems()[0]
    pltdataitem.setPen(color = 'r', width =1.)
    pltdataitem.setSymbol('o')
    
    return win, imv


def pixelReader(imv):
    # Make text field that reads pixel positions of the image.
    view = imv.getView()
    label = pg.TextItem()
    label.setPos(0,-20)
    view.addItem(label)
    
    def mouseMoved(evt):
        pos = evt[0]  ## using signal proxy turns original arguments into a tuple
#        if view.sceneBoundingRect().contains(pos):
        mousePoint = view.mapSceneToView(pos)
        indexX = int(mousePoint.x())
        indexY = int(mousePoint.y())
#        if indexX > 0 and indexY > 0 and indexX < data.shape[1] and indexY < data.shape[2]:
        label.setText("x=%0.1f,  y=%0.1f" % (indexX, indexY))
    proxy = pg.SignalProxy(view.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)



#stackViewFunc(data, xvals, winTitle, title).show()

    
#if __name__ == '__main__':
#    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#        QtGui.QApplication.instance().exec_()

