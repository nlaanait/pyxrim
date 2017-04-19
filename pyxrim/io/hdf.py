"""
Created on 4/18/17

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

"""
Module where HDF5 operations are defined. The main class ioHDF5 is taking from the pycroscopy package.
(by N. Laanait, S. Somnath, and C. Smith).
"""

from time import time, sleep
import h5py
import sys
import os
import numpy as np
from warnings import warn
import subprocess

from spec import SpecScanStack
#
class ioHDF5(object):

    def __init__(self, file_handle):
        """
        Handles:
            + I/O operation from HDF5 file.
            + Utilities to get data and associated auxiliary.

        Parameters
        ----------
        file_handle : Object - String or Unicode or open hdf5 file
            Absolute path to the h5 file or an open hdf5 file
        """
        if type(file_handle) in [str, unicode]:
            # file handle is actually a file path
            propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
            try:
                fid = h5py.h5f.open(file_handle,fapl=propfaid)
                self.file = h5py.File(fid, mode = 'r+')
            except IOError:
                #print('Unable to open file %s. \n Making a new one! \n' %(filename))
                fid = h5py.h5f.create(file_handle,fapl=propfaid)
                self.file = h5py.File(fid, mode = 'w')
            except:
                raise
            self.path = file_handle
        elif type(file_handle) == h5py.File:
            # file handle is actually an open hdf file
            if file_handle.mode == 'r':
                warn('ioHDF5 cannot work with open HDF5 files in read mode. Change to r+ or w')
                return
            self.file = file_handle.file
            self.path = file_handle.filename

    def clear(self):
        """
        Clear h5.file of all contents

        file.clear() only removes the contents, it does not free up previously allocated space.
        To do so, it's necessary to use the h5repack command after clearing.
        Because the file must be closed and reopened, it is best to call this
        function immediately after the creation of the ioHDF5 object.
        """
        self.file.clear()
        self.repack()

    def repack(self):
        """
        Uses the h5repack command to recover cleared space in an hdf5 file.
        h5repack can also be used to change chunking and compression, but these options have
        not yet been implemented here.
        """
        self.close()
        tmpfile = self.path+'.tmp'

        '''
        Repack the opened hdf5 file into a temporary file
        '''
        try:
            repack_line = ' '.join(['h5repack', '"'+self.path+'"', '"'+tmpfile+'"'])
            subprocess.check_output(repack_line,
                                    stderr=subprocess.STDOUT,
                                    shell=True)
            # Check that the file is done being modified
            sleep(0.5)
            while time()-os.stat(tmpfile).st_mtime <= 1:
                sleep(0.5)
        except subprocess.CalledProcessError as err:
            print('Could not repack hdf5 file')
            raise Exception(err.output)
        except:
            raise

        '''
        Delete the original file and move the temporary file to the originals path
        '''
        # TODO Find way to get the real OS error that works in and out of Spyder
        try:
            os.remove(self.path)
            os.rename(tmpfile, self.path)
        except:
            print('Could not copy repacked file to original path.')
            print('The original file is located {}'.format(self.path))
            print('The repacked file is located {}'.format(tmpfile))
            raise

        '''
        Open the repacked file
        '''
        self.file = h5py.File(self.path, mode = 'r+')

    def close(self):
        """
        Close h5.file
        """
        self.file.close()

    def delete(self):
        """
        Delete h5.file
        """
        self.close()
        os.remove(self.path)

    def flush(self):
        """
        Flush data from memory and commit to file.
        Use this after manually inserting data into the hdf dataset
        """
        self.file.flush()

    def scans_export(self, specfilepath, imagedir, parentGroup=''):
        """
        Writes scans into HDF5 file. 3 groups are created: Raw: raw data i.e. scan images, Process: processed data,
        Meta: metadata e.g. DARK, READ images, etc...
        :param specfilepath: String, Absolute Path of spec file from which scans will be exported.
        :param imagedir: String, Absolute Path of directory that contain scan image folders.
        :param parentGroup: String, optional, default specfilename. Name of parent group that hosts the scans.
        """
        if parentGroup is '':
            specfilename = specfilepath.split(os.sep)[-1]
            parentName = specfilename.split('.')[0]
        else:
            parentName = parentGroup

        # create parent group and subgroups
        try:
            pGroup = self.file.create_group(parentName)
        except ValueError:
            print('Group %s already Exists in %s.\nOverwriting is not allowed. Please provide a different parentGroup name or '
                  'a different HDF5 file.'
                  %(self.file["/"+parentName].name, self.file))
            sys.exit(1)
        self.raw = pGroup.create_group('Raw')
        self.process = pGroup.create_group('Process')
        self.meta = pGroup.create_group('Meta')

        # load a spec file
        scanstack = SpecScanStack(imagedir,specfilepath)
        scanno = int(scanstack.specData.getStats().split(' ')[3])

        # loop over scans
        for scanNumber in np.arange(1, scanno+1):
            try:
                dsetname = '{:s}{:03d}'.format('S', scanNumber)
                # Get stack of images and scan data for each scan
                scanstack.setScan(int(scanNumber))
                scan, stack = scanstack.getData()

                # Create dataset and load images data into it
                dset = self.raw.create_dataset(dsetname, data=stack, dtype=stack.dtype,
                                               compression='lzf')

                # Create dataset attributes and load their values from scan data
                scandic = scan.__getstate__()
                keys = scandic.keys()

                # List of keys to skip
                # TODO: This should be one of the input arguments
                invL = ['datafile', 'data', 'scanplotCCD', 'scandata', 'scandate', 'scanplot',
                        'values', 'ccdDarkFilenames', 'header', 'ccdFilenames', 'ccdAcquirePeriod', 'ccdAcquireTime',
                        'ccdNumAcquisitions']
                for key in keys:
                    try:
                        invL.index(key)
                    except ValueError:
                        attrs = dset.attrs
                        data = scandic[key]
                        attrs.create(key, data)
                    else:
                        pass
                print('Finished Writing dataset %s\n' %dsetname)
            except IndexError:
                print('Skipped Writing dataset %s\n'%dsetname)

    def direct_export(self, group='Meta'):
        """
        Inputs any data in the form of a numpy array into any of the subgroups (Raw, Process, Meta)
        :param:group: String
        """
        # TODO: Implement this method.
        pass