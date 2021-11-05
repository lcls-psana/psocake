# Cheetah-related data and methods
import sys
import h5py
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import namedtuple

class DetectorNotSupportedError(Exception):
    """Base class for other exceptions"""
    pass

@dataclass
class SupportedDetectors:
    supported = {'cspad','rayonix','jungfrau4m','epix10k2m'} # TODO: add pnccd for SPI

    @classmethod
    def parseDetectorName(cls, detName: str) -> str:
        """simplify detector name into psocake name and return lower case"""
        for det in cls.supported:
            if 'epix10k' in detName.lower() and '2m' in detName.lower():
                return 'epix10k2m'
            elif 'cxids' in detName.lower() and 'jungfrau' in detName.lower():
                return 'jungfrau4m' 
            elif det in detName.lower():
                return det
        raise DetectorNotSupportedError("{} is not supported in psocake".format(detName))

class DetectorDescriptor(ABC):

    @property
    @abstractmethod
    def psanaDim(self):
        """(seg, row, col)"""
        raise NotImplementedError

    @property
    @abstractmethod
    def quads(self):
        """"(numQuad, numAsicsPerQuad)"""
        raise NotImplementedError

    @property
    def tileDim(self):
        """"(dim0, dim1)"""
        ChDim = namedtuple('ChDim', ['dim0', 'dim1'])
        return ChDim(self.quads.numAsicsPerQuad * self.psanaDim.rows, self.quads.numQuad * self.psanaDim.cols)

    @abstractmethod
    def convert_peaks_to_cheetah(s, r, c):
        """convert psana peak positions to cheetah tile positions"""

    @abstractmethod
    def convert_peaks_to_psana(row2d, col2d):
        """convert cheetah tile peak positions to psana positions"""

    def pct(self, unassembled):
        """psana cheetah transform: convert psana unassembled detector to cheetah tile shape"""
        counter = 0
        img = np.zeros(self.tileDim)
        for quad in range(self.quads.numQuad):
            for seg in range(self.quads.numAsicsPerQuad):
                img[seg * self.psanaDim.rows:(seg + 1) * self.psanaDim.rows,
                    quad * self.psanaDim.cols:(quad + 1) * self.psanaDim.cols] = unassembled[counter, :, :]
                counter += 1

        return img

    def ipct(self, tile):
        """inverse psana cheetah transform: convert cheetah tile to psana unassembled detector shape"""
        calib = np.zeros((self.quads.numQuad * self.quads.numAsicsPerQuad, self.psanaDim.rows, self.psanaDim.cols))
        counter = 0
        for quad in range(self.quads.numQuad):
            for seg in range(self.quads.numAsicsPerQuad):
                calib[counter, :, :] = \
                    tile[seg * self.psanaDim.rows:(seg + 1) * self.psanaDim.rows,
                         quad * self.psanaDim.cols:(quad + 1) * self.psanaDim.cols]
                counter += 1
        return calib

class cspad(DetectorDescriptor):

    @property
    def psanaDim(self):
        """(seg, row, col)"""
        Psdim = namedtuple('Psdim', ['segs', 'rows', 'cols'])
        return Psdim(segs=32, rows=185, cols=388)

    @property
    def quads(self):
        """"(numQuad, numAsicsPerQuad)"""
        Quads = namedtuple('Quads', ['numQuad', 'numAsicsPerQuad'])
        return Quads(numQuad=4, numAsicsPerQuad=8)

    def convert_peaks_to_cheetah(self, s, r, c):
        """convert psana peak positions to cheetah tile positions"""
        if isinstance(s, np.ndarray):
            s = s.astype('int')
            r = r.astype('int')
            c = c.astype('int')
        row2d = (s % self.quads.numAsicsPerQuad) * self.psanaDim.rows + r  # where s%8 is a segment in quad number [0,7]
        col2d = (s / self.quads.numAsicsPerQuad) * self.psanaDim.cols + c  # where s/8 is a quad number [0,3]
        return row2d, col2d

    def convert_peaks_to_psana(self, row2d, col2d):
        """convert cheetah tile peak positions to psana positions"""
        s = (row2d / self.psanaDim.rows) + (col2d / self.psanaDim.cols * self.quads.numAsicsPerQuad)
        r = row2d % self.psanaDim.rows
        c = col2d % self.psanaDim.cols
        return s, r, c

class epix10k2m(DetectorDescriptor):

    @property
    def psanaDim(self):
        """(seg, row, col)"""
        Psdim = namedtuple('Psdim', ['segs', 'rows', 'cols'])
        return Psdim(segs=16, rows=352, cols=384)

    @property
    def quads(self):
        """"(numQuad, numAsicsPerQuad)"""
        Quads = namedtuple('Quads', ['numQuad', 'numAsicsPerQuad'])
        return Quads(numQuad=1, numAsicsPerQuad=16)

    def convert_peaks_to_cheetah(self, s, r, c):
        """convert psana peak positions to cheetah tile positions"""
        row2d = s * self.psanaDim.rows + r
        col2d = c
        return row2d, col2d

    def convert_peaks_to_psana(self, row2d, col2d):
        """convert cheetah tile peak positions to psana positions"""
        s = (row2d / self.psanaDim.rows) + (col2d / self.psanaDim.cols)
        r = row2d % self.psanaDim.rows
        c = col2d % self.psanaDim.cols
        return s, r, c

class jungfrau4m(DetectorDescriptor):

    @property
    def psanaDim(self):
        """(seg, row, col)"""
        Psdim = namedtuple('Psdim', ['segs', 'rows', 'cols'])
        return Psdim(segs=8, rows=512, cols=1024)

    @property
    def quads(self):
        """"(numQuad, numAsicsPerQuad)"""
        Quads = namedtuple('Quads', ['numQuad', 'numAsicsPerQuad'])
        return Quads(numQuad=1, numAsicsPerQuad=8)

    def convert_peaks_to_cheetah(self, s, r, c):
        """convert psana peak positions to cheetah tile positions"""
        row2d = s * self.psanaDim.rows + r
        col2d = c
        return row2d, col2d

    def convert_peaks_to_psana(self, row2d, col2d):
        """convert cheetah tile peak positions to psana positions"""
        s = (row2d / self.psanaDim.rows)
        r = row2d % self.psanaDim.rows
        c = col2d
        return s, r, c

class rayonix(DetectorDescriptor):

    @property
    def psanaDim(self):
        """(seg, row, col)"""
        Psdim = namedtuple('Psdim', ['segs', 'rows', 'cols'])
        return Psdim(segs=1, rows=1920, cols=1920)

    @property
    def quads(self):
        """"(numQuad, numAsicsPerQuad)"""
        Quads = namedtuple('Quads', ['numQuad', 'numAsicsPerQuad'])
        return Quads(numQuad=1, numAsicsPerQuad=1)

    def convert_peaks_to_cheetah(self, s, r, c):
        """convert psana peak positions to cheetah tile positions"""
        row2d = r
        col2d = c
        return row2d, col2d

    def convert_peaks_to_psana(self, row2d, col2d):
        """convert cheetah tile peak positions to psana positions"""
        s = 0
        r = row2d
        c = col2d
        return s, r, c

def invertBinaryImage(img):
    """For a binary image, swap 0s and 1s"""
    return -1*(img-1)

def readMask(fname: str, dset='/entry_1/data_1/mask'):
    """
    :param fname: full path to hdf5 static cheetah-shaped mask in
    :param dset: dataset containing the mask
    :return: mask: numpy array
    """
    if fname is None: return None
    with h5py.File(fname, 'r') as f:
        return invertBinaryImage(f[dset][()])

def saveCheetahFormatMask(outDir, detDesc, run=None, combinedMask=None):
    dim0, dim1 = detDesc.tileDim
    if run is not None:
        fname = outDir+'/r'+str(run).zfill(4)+'/staticMask.h5'
    else:
        fname = outDir + '/staticMask.h5'
    print("Saving static mask in Cheetah format: ", fname)
    with h5py.File(fname, 'w') as f:
        dset = f.create_dataset('/entry_1/data_1/mask', (dim0, dim1), dtype='int')
        # Convert calib image to cheetah image
        # This ensures mask displayed on GUI gets used in peak finding / indexing
        if combinedMask is None:
            img = np.ones((dim0, dim1))
        else:
            img = detDesc.pct(combinedMask)
        dset[:,:] = img
