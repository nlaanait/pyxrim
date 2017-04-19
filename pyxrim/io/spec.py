"""
Created on 4/17/17

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
The module contains classes and methods to parse a spec file (adapted from pyspec (by S. B. Wilkins))
and classes that wrap spec information with spec scan in the form of an image stack as helper classes
to export and pack scans into an hdf5 file.
"""

import time
import sys
import os
import numpy as np
from warnings import warn
from copy import deepcopy, copy
from skimage.io import ImageCollection
from skimage.data import imread



#########################
# Begin: pyspec classes #
#########################

"""
Utility Functions for pyspec-based classes,
"""
def removeIllegals(key):
    """Remove illegal character from string"""
    illegal = [('/', ''), ('-', 'm'), ('+', 'p'), (' ', '')]
    for j,i in illegal:
        key = key.replace(j,i)
    if key[0].isalpha() == False:
        key = "X" + key
    return key

def splitSpecString(ips):
    """Split a spec string which is formated with two spaces"""
    ops = []
    for o in ips.split('  '):
        if o != '':
            ops.append(o.strip())

    return ops

class SpecDataFile(object):
    """ DataFile class for handling spec data files. Adapted from pyspec (by S. B. Wilkins)"""

    def __init__(self, fn, **kwargs):
        """Initialize SpecDataFile

        Parameters
        ----------
        fn : string
             Filename of spec file to open

        Returns a SpecDataFile object

        """
        self.filename = fn
        self.fileLastAccess = -1
        self._loadSpecFile()

    def _loadSpecFile(self):
        print("**** Opening specfile %s." % self.filename)

        self.index()
        self.readHeader()
        print(self.getStats())

        self.scandata = {}

        self.fileStats = os.stat(self.filename)
        print(self.fileStats)
        self.fileLastAccess = self.fileStats.st_mtime


    def reload(self):
        """Reload the data file
        This routine reloads (reindexes) the datafile"""

        print("**** Reloading SpecFile")
        self._loadSpecFile()

    def setMode(self, mode='concatenate'):
        """Set the modee to deal with multiple scans

        mode : string
           If mode is 'concatenate' then concatenate scans together.
           If mode is 'bin' then bin (numerically the scans together.
        """

        if mode == 'concatenate':
            self.mode = 'concat'
            print("**** Multiple scans will be concatenated.")
            return
        elif mode == 'bin':
            self.mode = 'bin'
            print("**** Multiple scans will be binned.")
            return
        else:
            raise Exception("Unknown mode %s" % mode)

        return

    def readHeader(self):
        """Read the spec header from the datafile.

        Currently supported header items:
                '#O'    (Motor positions)
        """

        self.file = open(self.filename, 'rb')

        print("---- Reading Header.")

        self.motors = []
        self.file.seek(0, 0)
        line = self.file.readline()
        while line[0:2] != "#S":
            if line[0:2] == "#O":
                self.motors = self.motors + splitSpecString(line[4:])
            line = self.file.readline()
        self.file.close()
        return

    def index(self):
        """Index the datafile

        This routine indexes and sorts the byte-offests for
        all the scans (Lines beginning with '#S')

        """
        self.file = open(self.filename, 'rb')


        print("---- Indexing scan :       ")
        sys.stdout.flush()
        sys.stderr.flush()

        self.file.seek(0, 0)
        self.findex = {}

        pos = self.file.tell()
        line = self.file.readline()
        while line != "":
            if line[0:2] == "#S":
                a = line.split()
                s = int(a[1])
                if (s % 5) is 0:
                    print("\b\b\b\b\b\b\b%5d " % s)
                    sys.stdout.flush()
                self.findex[s] = pos
            pos = self.file.tell()
            line = self.file.readline()
        # print("\b\b\b\b\b\b\bDONE  ")
        print("DONE!")
        self.file.close()
        return

    def getStats(self, head="---- "):
        """ Returns string with statistics on specfile.

        Parameters
        ----------

        head : string
           append string head to status text

        """
        string = ""
        string = string + head + "Specfile contains %d scans\n" % len(self.findex)
        string = string + head + "Start scan = %d\n" % min(self.findex.keys())
        string = string + head + "End   scan = %d\n" % max(self.findex.keys())

        return string

    def _moveto(self, item):
        """Move to a location in the datafile for scan"""

        if self.findex.has_key(item):
            self.file.seek(self.findex[item])
        else:
            # Try re-indexing the file here.
            print("**** Re-indexing scan file\n")
            self.index()
            if self.findex.has_key(item):
                self.file.seek(self.findex[item])
            else:
                raise Exception("Scan %s is not in datafile ....." % item)

    def __getitem__(self, item):
        """Convinience routine to use [] to get scan"""
        return self.getScan(item, setkeys=True)

    def getAll(self, *args, **kwargs):
        """Read all scans into the object"""
        for s in self.findex.keys():
            self.getScan(s, *args, **kwargs)

    def getScan(self, item, mask=None, setkeys=True, persistent=True,
                reread=False, **kwargs):
        """Get a scan from the data file

        This routine gets a scan from the data file and loads it into the
        list of SpecData instances.

        setting 'setkeys = True' will set attributes to the
        specdata item of all the motors and counters

        The mask can be used to detete bad points. This should be a list of datum
        (data point numbers) which should be excluded. If the "item" is a list of
        scans (that will be concatenated or binned together) this should be a list
        containing either None for no removal of data points or a list of points to
        remove

        Returns the ScanData object corresponding to the scan requested.

        """

        if type(item) == int:
            items = (item,)
        elif type(item) == float:
            items = (int(item),)
        elif type(item) == list:
            items = tuple(item)
        elif type(item) == tuple:
            items = item
        elif type(item) == np.ndarray:
            items = item.tolist()
        else:
            raise Exception("item can only be <int>, <float>, <list>, <array> or <tuple>")

        if mask is None:
            mask = [None for i in items]

        if len(mask) != len(items):
            raise Exception("The mask list should be the same size as the items list")

        self.fileStats = os.stat(self.filename)

        # Check here if file needs to be re-read

        if self.fileLastAccess < self.fileStats.st_mtime:
            # Re-read file.

            print("**** File has changed on disk. Re-reading SPEC file.")
            self._loadSpecFile()

        self.file = open(self.filename, 'rb')

        rval = []
        n = 0
        for i, m in zip(items, mask):
            if i < 0:
                _items = self.findex.keys()
                _items.sort()
                i = _items[i]
            print("**** Reading scan/item %s" % i)

            # Check if scan is in datafile
            if (self.scandata.has_key(i) is False) or (reread is True):
                self._moveto(i)
                self.scandata[i] = SpecScan(self, i, setkeys, mask=m, **kwargs)

            rval.append(self.scandata[i])

        self.file.close()

        if len(rval) > 1:
            newscan = deepcopy(rval[0])
            for i in range(len(rval) - 1):
                if self.mode == 'concat':
                    newscan.concatenate(rval[i + 1])
                elif self.mode == 'bin':
                    newscan.bin(rval[i + 1], binbreak='Seconds')
                else:
                    raise Exception("Unknown mode to deal with multiple scans.")
            rval = [newscan]

        return rval[0]

    def _getLine(self):
        """Read line from datafile"""

        line = self.file.readline()
        # if 0x10:
        #     # print("xxxx %s" % line.strip())
        return line

class SpecScan(object):
    """Class defining a SPEC scan

    This class defines a single spec scan, or a collection of scans either binned or concatenated.
    The class has members for the data and variables contained in the scan. If the optional 'setkeys'
    is defined (see __init__) then the motor positions and variables will be set as class members. For
    example, in a typical scan the motor position TTH can be accessed as a class member by SpecScan.TTH

    This object has some standard members:

    header     : [string]  Header of spec scan
    values     : [dict]    All scan data variables (Motors and Counters)
    data       : [array]   All scan data (cols = variables, rows = datum)
    comments   : [string]  Comments inserted in scan
    scandate   : [time]    Date of start of scan

    There are a number of special members. Wherever possible the following members are added to
    the class:

    Qvec       : [array]   Q Vector
    alphabeta  : [array]   array([alpha, beta])
    wavelength : [float]   incident wavelength
    omega      : [float]   omega ((TTH / 2) - TH)
    azimuth    : [float]   azimuthal angle

    """

    def __init__(self, specfile, item, setkeys=True, mask=None, **kwargs):
        """Read scan data from SpecFile

        Initialize the SpecScan class from a SpecData instance.

        Parameters
        ----------
        specfile       : [SpecFile] Instance of a spec data file.
        item           : [item]     Scan item.
        setkeys        : [bool]     If true set items of the class
                                    from all variables.
        """

        # Keep track of the datafile

        self.datafile = specfile
        self.scandata = SpecData()
        self.scanplot = None
        self.scanplotCCD = None
        self.setkeys = setkeys

        line = specfile._getLine()
        print("---- %s" % line.strip())
        sline = line.strip().split()

        self.scan = int(sline[1])
        self.scan_type = sline[2]
        self.scan_command = ' '.join(sline[2:])

        self.header = line
        self.comments = ""

        x = 0
        self.values = {}
        self.data = np.array([])

        self.UB = np.eye(3)

        line = specfile._getLine()
        self.header = self.header + line

        # Finally overide any assigments with values passed as keyword arguments
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])

        #
        # Read the spec header and place the data into this class
        #

        while (line[0:2] != "#L") & (line != ""):
            if line[0:2] == "#P":
                # Motor positions
                pos = line.strip().split()
                for i in range(1, len(pos)):
                    self.scandata.setValue(removeIllegals(specfile.motors[x]), np.array([float(pos[i])]))
                    x += 1

            elif line[0:2] == "#C":
                # Comments
                self.comments = self.comments + line

            elif line[0:2] == "#D":
                try:
                    self.scandate = time.strptime(line[2:].strip())
                except:
                    self.scandate = None
            elif line[0:3] == "#G4":
                try:
                    pos = line[3:].strip().split()
                    self.Qvec = np.array([float(pos[0]), float(pos[1]), float(pos[2])])
                    self.alphabeta = np.array([float(pos[4]), float(pos[5])])
                    self.wavelength = float(pos[3])
                    self.energy = 12398.4 / self.wavelength
                    self.omega = float(pos[6])
                    self.azimuth = float(pos[7])
                except:
                    print("**** Unable to read geometry information (G4)")
            elif line[0:3] == "#G1":
                try:
                    pos = line[3:].strip().split()
                    pos = np.array(map(float, pos))

                    self.Lattice = pos[0:6]
                    self.RLattice = pos[6:12]
                    self.or0 = pos[12:15]
                    self.or1 = pos[15:18]
                    sa = pos[18:-2].reshape(2, -1)
                    self.or0Angles = sa[0, :]
                    self.or1Angles = sa[1, :]
                    self.or0Lambda = pos[-2]
                    self.or1Lambda = pos[-1]
                except:
                    print("**** Unable to read geometry information (G1)")

            elif line[0:3] == "#G3":
                try:
                    pos = line[3:].strip().split()
                    pos = np.array(map(float, pos))

                    self.UB = pos.reshape(-1, 3)
                except:
                    print("**** Unable to read UB matrix (G3)")

            line = specfile._getLine()
            self.header = self.header + line

        if line[0:2] == "#L":
            # Comment line just before data
            self.cols = splitSpecString(line[3:])
            # print("---- %s" % line.strip())

        line = specfile._getLine()
        self.header = self.header + line

        # print("---- %s" % line.strip())

        while (line[0:2] != "#S") & (line != "") & (line[0:4] != "# CM"):
            if line[0] != "#":
                datum = np.array([])
                d = line.strip().split()
                if len(d) != 0:
                    for i in range(len(d)):
                        v = np.array([float(d[i])])
                        datum = np.concatenate((datum, v), 0)

                    if self.data.size == 0:
                        self.data = datum
                    else:
                        self.data = np.vstack((self.data, datum))

            elif line[0:2] == '#C':
                self.comments = self.comments + line
            else:
                self.header = self.header + line

            line = specfile._getLine()

        if self.data.ndim == 1:
            self.data = np.array([self.data])

        if mask is not None:
            print("---- Removing rows %s from data." % str(mask))
            self.data = np.delete(self.data, mask, axis=0)

        self.scanno = np.ones(self.data.shape[0], dtype=np.int) * self.scan
        self.scandatum = np.arange(self.data.shape[0])

        # Now set the motors
        self._setcols()


        print("---- Data is %i rows x %i cols." % (self.data.shape[0], self.data.shape[1]))

        return None

    def _setcols(self):
        if self.data.shape[0] > 0:
            for i in range(len(self.cols)):
                if len(self.data.shape) == 2:
                    self.scandata.setValue(self.cols[i], self.data[:, i])
                else:
                    self.scandata.setValue(self.cols[i], np.array([self.data[i]]))
            self.values = self.scandata.values

            # Now set the variables into the scan class from the data

            if self.setkeys:
                for i in self.scandata.values.keys():
                    iri = removeIllegals(i)
                    # if 0x02:
                    #     # print("oooo Setting variable %s (%s)" % (i, iri))
                    setattr(self, iri, self.scandata.values[i])

    def concatenate(self, a):
        # Could put check in here for cols matching ?!?

        self.header = self.header + a.header
        self.data = np.vstack((self.data, a.data))
        self.scanno = np.concatenate((self.scanno, a.scanno))
        self.scandatum = np.concatenate((self.scandatum, a.scandatum))

        for ext in self.datafile.userExtensions:
            ext.concatenateSpecScan(self, a)

        self._setcols()

    def bin(self, a, binbreak=None):
        """Bin the scans together adding the column values

        a is a SpecScan object of the file to bin.

        Note:
        This routine only bins the "data" portion of the scan. It returns the
        origional scans motors etc.

        """
        # First check if scans are the same.
        if self.cols != a.cols:
            raise Exception("Scan column headers are not the same.")
        self.header = self.header + a.header
        if binbreak != None:
            if binbreak in self.cols:
                flag = False
                for i in range(len(self.cols)):
                    if self.cols[i] == binbreak:
                        flag = True
                    if flag:
                        self.data[:, i] = self.data[:, i] + a.data[:, i]
            else:
                raise Exception("'%s' is not a column of the datafile." % binbreak)
        else:
            self.data = self.data + a.data
        self._setcols()

        return self

    def __getstate__(self):
        # Copy the dictionary and then return the copy
        return self.__dict__

    def __str__(self):
        return self.show()

    def show(self, prefix="", nperline=4):
        """Return string of statistics on SpecScan"""
        p = ""
        p = p + "Scan:\n\n"
        p = p + "\t%s\n\n" % self.scan
        p = p + "Datafile:\n\n"
        p = p + "\t%s\n\n" % self.datafile.file.name
        p = p + "Scan Command:\n\n"
        p = p + "\t%s\n\n" % self.scan_command
        p = p + "Scan Constants:\n\n"

        j = nperline
        typestoprint = [float, str, np.ndarray, int, np.float64]

        for d in self.__dict__:
            if not self.scandata.values.has_key(d):
                if typestoprint.count(type(getattr(self, d))):
                    p = p + "%-19s " % d
                    print d, type(getattr(self, d))
                    j -= 1
                    if j == 0:
                        p = p + "\n"
                        j = nperline

        p = p + "\n\n"
        p = p + self.scandata.show(prefix, nperline)
        return p

    def getYE(self, ycol=None, mcol=None):
        """Return an tuple of two arrays of y and e"""

        if type(ycol) == str:
            ycol = self.scan.cols.index(ycol)
        if type(mcol) == str:
            mcol = self.scan.cols.index(mcol)
        if ycol == None:
            ycol = -1
        if mcol == None:
            mcol = -2

        y = self.data[:, ycol]
        m = self.data[:, mcol]

        e = np.sqrt(y) / y
        y = y / m
        e = e * y

        return (y, e)

class SpecData(object):
    """Class defining the data contained in a scan """
    def __init__(self):
        self.values = {}

    def setValue(self, key, data, setdata=True):
        self.values[key] = data
        # if 0x20:
        #     # print("oooo Setting key %s" % key)

    def get(self, key):
        if self.values.has_key(key):
            return self.values[key]
        else:
            return None

    def __str__(self):
        return self.show()

    def show(self, prefix="", nperline=6):
        """Return string of statistics on data (motors, scalars)"""

        j = nperline
        p = ""
        p = p + prefix + "Motors:\n\n"
        p = p + prefix
        for i in self.values.keys():
            if self.values[i].size == 1:
                p = p + "%-19s " % i
                j -= 1
                if j == 0:
                    p = p + "\n" + prefix
                    j = nperline

        if j != nperline:
            p = p + "\n" + prefix

        p = p + "\n"

        p = p + prefix + "\n"
        p = p + prefix + "Scan Variables:\n"
        p = p + prefix + "\n"
        j = nperline
        for i in self.values.keys():
            if self.values[i].size > 1:
                p = p + "%-19s " % i
                j -= 1
                if j == 0:
                    p = p + "\n" + prefix
                    j = nperline

        p = p + "\n"
        return p

#########################
# End: pyspec classes   #
#########################

class SpecScanStack(object):
    ''' This class takes a scan in a spec file and binds both its images and
    scan information into a single object known as the specScanStack.
    The images are held in the skimage.io.ImageCollection, and read as needed,
    and the spec scan object can be accessed through one of the methods.
    '''

    def __init__(self, ImageDir, SpecFilePath):
        '''
        :param ImageDir: String
                    Absolute file path of directory where images scan folders are found.
        :param specFilePath: String
                    Absolute file path of spec file.
        '''

        self.specFile = SpecFilePath.split(os.sep)[-1]
        if os.path.exists(SpecFilePath):
            self.SpecFilePath = SpecFilePath
            self.specData = SpecDataFile(self.SpecFilePath)
        else:
            warn('Spec File not found! Exiting...')
            sys.exit(IOError)
        if os.path.exists(ImageDir):
            self.dir = os.path.join(ImageDir, self.specFile.split('.')[0])
        else:
            warn('Image Directory not found! Exiting...')
            sys.exit(IOError)

    def setScan(self, scanNumber):
        self.scanNumber = int(scanNumber)

    def getData(self):
        # Loading scan spec data.
        scan = self.specData[self.scanNumber]

        # Reading scan images.
        stackDirectory = os.path.join(self.dir,'S{:03d}'.format(self.scanNumber))
        time.sleep(0.75)
        print('Reading images in %s'%stackDirectory)
        images = ImageCollection(stackDirectory + os.sep + '*.tif')
        stack = np.array([imp for imp in images])
        print('Found %d images with dimensions: %s\n'%(stack.shape[0], format(stack.shape[1:])))

        # Checking dimensions of scan and # of images.
        if stack.shape[0] is not scan.data.shape[0]:
            message = '# of images: %d and # of points: %d in scan %s%03d do not match!'%(stack.shape[0],scan.data.shape[0],'S',self.scanNumber)
            warn(message)
            raise IndexError
        return scan,stack