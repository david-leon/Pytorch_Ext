# coding:utf-8
"""
Utility functions for Pytorch_Ext
Created   :   8, 11, 2017
Revised   :   8, 18, 2017
             12, 26, 2019  add freeze_module()/unfreeze_module()/get_trainable_parameters()
              7, 29, 2020  change default protocol of gpickle to v4

All rights reserved
"""

__author__ = 'dawei.leng'
import torch, gc, warnings
import gzip, pickle, hashlib
import socket
import numpy as np
from datetime import datetime

def freeze_module(module):
    """
    Freeze a module
    :param module: instance of nn.Module
    :return: No return
    """
    module.train(False)
    for param in module.parameters():
        param.requires_grad_(False)

def unfreeze_module(module):
    """
    Un-freeze a module
    :param module: instance of nn.Module
    :return: No return
    """
    module.train(True)
    for param in module.parameters():
        param.requires_grad_(True)

def get_trainable_parameters(module, with_name=False):
    """
    Retrieve only trainable parameters, for feeding optimizer
    :param module:
    :param with_name: if True, output in format of (name, tensor), else only tensor returned
    :return:
    """
    for name, tensor in module.named_parameters():
        if tensor.requires_grad:
            if with_name:
                yield name, tensor
            else:
                yield tensor

def set_value(t, v):
    """
    Set tensor value with numpy array
    :param t: tensor
    :param v: numpy array
    :return: No return
    """
    with torch.no_grad():
        t.copy_(torch.from_numpy(v))

def get_device(x):
    """
    Retrieve device from tensor or module
    :param x: Tensor or nn.Module
    :return: torch.device
    """
    if isinstance(x, torch.Tensor):
        return x.device
    elif isinstance(x, torch.nn.Module):
        device = None
        for p in x.parameters():
            device = p.device
            break
        if device is None:
            for p in x.buffers():
                device = p.device
                break
        return device
		
def grad_clip(parameters, min, max):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    for p in parameters:
        p.grad.data.clamp_(min=min, max=max)

def check_exception_msg(e, exception_descriptions=None):
    """
    Check whether exception `e` is related to CUDA OOM
    :param e: exception captured
    :param exception_descriptions: list of strings, each string is part of description of exception to be captured. Default = None, only CUDA OOM exceptions will be handled.
    """
    if exception_descriptions is None:
        exception_descriptions = ['out of memory',
                                  'CUBLAS_STATUS_ALLOC_FAILED',
                                  'CUBLAS_STATUS_NOT_INITIALIZED',  ]
    for msg in exception_descriptions:
        if msg in e.args[0]:
            return True
    return False

def torch_safe_run(fn, inputs, exception_descriptions=None):
    """
    Safe run against CUDA OOM, otherwise just raise the captured exception
    :param fn: pytorch function
    :param inputs: dict
    :param exception_descriptions: list of strings, each string is part of description of exception to be captured. Default = None, only CUDA OOM exceptions will be handled.
    :return: (status, result), in which status > 0 means OOM occurs; result is the return from `fn`
    """
    try:
        result = fn(**inputs)
        status = 0
        return status, result
    except Exception as e:
        if check_exception_msg(e, exception_descriptions=exception_descriptions):
            gc.collect()
            torch.cuda.empty_cache()
            status = 1
            return status, None
        else:
            raise e

class chunked_byte_writer(object):
    """
    This class is used for by-passing the bug in gzip/zlib library: when data length exceeds unsigned int32 limit,
    gzip/zlib will break
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
    def dump(data, filename, compresslevel=9, protocol=4):
        with gzip.open(filename, mode='wb', compresslevel=compresslevel) as f:
            pickle.dump(data, chunked_byte_writer(f), protocol=protocol)
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
    def loads(zipped_bytes):
        bytes = gzip.decompress(zipped_bytes)
        return pickle.loads(bytes)

    @staticmethod
    def dumps(data, compresslevel=9, protocol=4):
        zipped_bytes = gzip.compress(pickle.dumps(data, protocol=protocol), compresslevel=compresslevel)
        return zipped_bytes

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
    s.close()
    return local_ip_address

def get_time_stamp():
    """
    Create a formatted string time stamp
    :return:
    """
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')

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

class verbose_print(object):
    """
    `print` with verbose level filtering
    """
    def __init__(self, level=0, prefix=None):
        self.level = level
        self.prefix = prefix

    def __call__(self, *args, **kwargs):
        if self.prefix is not None:
            print(self.prefix+': ', end='')
        if 'l' not in kwargs:
            l = 0
        else:
            l = kwargs['l']
            kwargs.pop('l')
        if l < self.level:
            pass
        else:
            print(*args, **kwargs)

def get_file_md5(file=None, data=None):
    """
    Compute MD5 digest for binary file.
    :param file: file path, if given, `data` input will be ignored
    :param data: binary data, must be given if `file` is set to None
    :return: string, file's MD5 digest
    """
    hasher = hashlib.md5()
    if file is not None:
        with open(file, mode='rb') as f:
            data = f.read()
    if data is None:
        return None
    hasher.update(data)
    md5_digest = hasher.hexdigest()
    return md5_digest

if __name__ == '__main__':
    INFO = ['This is a collection of auxiliary functions for DL.\n',
            'Author: David Leon\n',
            'All rights reserved\n']
    print(*INFO)
