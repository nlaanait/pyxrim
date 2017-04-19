"""
Created on 4/19/17

@author: Numan Laanait -- nlaanait@gmail.com
"""


#MIT License

#Copyright (c) 2017 Numan Laanait

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import pyxrim as px
import os

# define data paths
datadir = os.path.join(os.getcwd(),'data')
specfile = os.path.join(datadir,'BFO_STO_1_1.spec')
imagedir = os.path.join(datadir,'images')

# load ioHDF5
ss = px.SpecScanStack(imagedir,specfile)
ss.setScan(1)
scan, stack = ss.getData()
# print(int(ss.specData.getStats().split(' ')[3]))
# io.scans_export(specfile,imagedir,parentGroup='Main')