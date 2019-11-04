# coding:utf-8
# Utility functions for Pytorch_Ext
# Created   :   8, 11, 2017
# Revised   :   8, 18, 2017
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'
import torch
import gzip, pickle
import socket, time
import numpy as np

def get_device(x):
    """
    Convenient & unified function for getting device a Tensor or Variable resides on
    :param x: torch Tensor or Variable
    :return: device ID, -1 for CPU; 0, 1, ... for GPU
    """
    if x.is_cuda is False:
        return -1
    else:
        return x.get_device()
		
def set_device(x, device):
    """
    Convenient & unified function for moving a Tensor or Variable to a specified device
    :param x: torch Tensor or Variable
    :param device: device ID, -1 for CPU; 0, 1, ... for GPU
    :return: Tensor or Variable on the specified device
    """
    if device < 0:
        if x.is_cuda:
            return x.cpu()
        else:
            return x
    else:
        return x.cuda(device)

def grad_clip(parameters, min, max):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    for p in parameters:
        p.grad.data = torch.clamp(p.grad.data, min=min, max=max)

class chunked_byte_writer(object):
    """
    This class is used for by-passing the bug in gzip/zlib library: when data length exceeds unsigned int limit, gzip/zlib will break
    file: a file object
    """
    def __init__(self, file, chunksize=4294967295):
        self.file = file
        self.chunksize = chunksize
    def write(self, data):
        for i in range(0, len(data), self.chunksize):
            self.file.write(data[i:i+self.chunksize])

class gpickle(object):
    """
    A pickle class with gzip enabled
    """
    @staticmethod
    def dump(data, filename, compresslevel=9):
        with gzip.open(filename, mode='wb', compresslevel=compresslevel) as f:
            pickle.dump(data, chunked_byte_writer(f))
            f.close()
    @staticmethod
    def load(filename):
        """
        The chunked read mechanism here is for by-passing the bug in gzip/zlib library: when
        data length exceeds unsigned int limit, gzip/zlib will break
        :param filename:
        :return:
        """
        buf = b''
        chunk = b'NULL'
        with gzip.open(filename, mode='rb') as f:
            while len(chunk) > 0:
                chunk = f.read(429496729)
                buf += chunk
        data = pickle.loads(buf)
        return data

    @staticmethod
    def loads(buf):
        return pickle.loads(buf)

    @staticmethod
    def dumps(data):
        return pickle.dumps(data)


class finite_memory_array(object):

    def __init__(self, array=None, shape=None, dtype=np.double):
        if array is not None:
            self.array = array
        elif shape is not None:
            self.array = np.zeros(shape=shape, dtype=dtype)
        else:
            raise ValueError('bad initialization para for Finite_Memory_Array')
        self.shape = self.array.shape
        self.curpos = 0
        self.first_round = True

    def update(self, value):
        self.array[:, self.curpos] = value
        self.curpos += 1
        if self.curpos >= self.shape[1]:
            self.curpos = 0
            self.first_round = False

    def clear(self):
        self.array[:] = 0
        self.curpos = 0
        self.first_round = True

    def get_current_position(self):
        return self.curpos

    @property
    def content(self):
        if self.first_round is True:
            return self.array[:, :self.curpos]
        else:
            return self.array

Finite_Memory_Array = finite_memory_array

def get_local_ip():
    """
    Get local host IP address
    :return:
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 0))                                # connecting to a UDP address doesn't send packets
    local_ip_address = s.getsockname()[0]
    return local_ip_address

def get_time_stamp():
    """
    Create a formatted string time stamp
    :return:
    """
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

class sys_output_tap:
    """
    Helper class to redirect sys output to file meanwhile keeping the output on screen
    Based on code snippet from https://github.com/smlaine2/tempens/blob/master/train.py#L30
    Example:
        #--- setup ---#
        stdout_tap = sys_output_tap(sys.stdout)
        stderr_tap = sys_output_tap(sys.stderr)
        sys.stdout = stdout_tap
        sys.stderr = stderr_tap
        #--- before training ---#
        stdout_tap.set_file(open(os.path.join(save_folder, 'stdout.txt'), 'wt'))
        stderr_tap.set_file(open(os.path.join(save_folder, 'stderr.txt'), 'wt'))
    """
    def __init__(self, stream, only_output_to_file=False):
        """

        :param stream: usually sys.stdout/sys.stderr
        :param only_output_to_file: flag. Whether disable output to original stream, default False.
        """
        self.stream = stream
        self.buffer = ''
        self.file = None
        self.only_output_to_file = only_output_to_file
        pass

    def write(self, s):
        if not self.only_output_to_file or self.file is None:
            self.stream.write(s)
            self.stream.flush()
        if self.file is not None:
            self.file.write(s)
            self.file.flush()
        else:
            self.buffer = self.buffer + s

    def set_file(self, f):
        assert(self.file is None)
        self.file = f
        self.file.write(self.buffer)
        self.file.flush()
        self.buffer = ''

    def flush(self):
        if not self.only_output_to_file or self.file is None:
            self.stream.flush()
        if self.file is not None:
            self.file.flush()

    def close(self):
        self.stream.close()
        if self.file is not None:
            self.file.close()
            self.file = None



if __name__ == '__main__':
    INFO = ['This is a collection of auxiliary functions for DL.\n',
            'Author: David Leon\n',
            'All rights reserved\n']
    print(*INFO)