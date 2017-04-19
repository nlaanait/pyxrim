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


wdir = os.getcwd()
with h5py.File(os.path.join(wdir, 'PTO_STO_20161005.hdf5'), 'r+') as f:
    for specf, datadir, date in zip(specfiles, dirs, datesL):
        # create group (date) and subgroups
        sd = SpecDataFile(os.path.join(datadir, specf))
        scans = int(sd.getStats().split(' ')[3])
        grp = f.create_group(date)
        grp.create_group('Raw')
        grp.create_group('Meta')
        grp.create_group('Process')

        # Loop over scans in the specfile
        for scanno in np.arange(1, scans + 1):
            try:
                # Assign scan images to 3D np.array
                ss = specScanStack(datadir, specf, int(scanno))
                data = ss.getStack()
                stack = np.array([imp for imp in data])
                # Assign 3D np.array to a dataset of 'Raw' subgroup
                subg = grp['Raw']
                dsetname = '{:s}{:03d}'.format('S', scanno)  # name of dataset
                dset = subg.create_dataset(dsetname, data=stack, dtype='uint16',
                                           compression='lzf')
                #                                           compression = 'gzip', compression_opts =9 )
                # Create then Assign attributes of dataset from specfile
                scan = ss.getScan()
                scandic = scan.__getstate__()
                keys = scandic.keys()
                invL = ['datafile', 'data', 'scanplotCCD', 'scandata', 'scandate', 'scanplot',
                        'values', 'ccdDarkFilenames', 'header', 'ccdFilenames', 'ccdAcquirePeriod', 'ccdAcquireTime',
                        'ccdNumAcquisitions']  # skip these keys.
                for key in keys:
                    try:
                        invL.index(key)
                    except ValueError:
                        #                    print key
                        attrs = dset.attrs
                        data = scandic[key]
                        attrs.create(key, data)
                    else:
                        pass
                print('Finished Writing Scan # %i' % (scanno))
            except IndexError:
                pass
