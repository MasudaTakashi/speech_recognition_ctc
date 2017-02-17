import argparse
import os
import sys
import math
import random
import chainer
import numpy as np
from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of examples in each mini-batch')
    """
    parser.add_argument('--bproplen', '-l', type=int, default=35,
                        help='Number of words in each mini-batch '
                             '(= length of truncated BPTT)')
    """
    parser.add_argument('--epoch', '-e', type=int, default=50,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.set_defaults(test=False)
    parser.add_argument('--unit', '-u', type=int, default=256,
                        help='Number of LSTM units in each layer')
    return parser.parse_args()


def print_progress(end, now):
    MAX_LEN = 50
    progress = 1.0 if now == end-1 else 1.0 * now / end
    BAR_LEN = MAX_LEN if now == end-1 else int(MAX_LEN * progress)
    progressbar_str =  ('[' + '=' * BAR_LEN +
                        ('>' if BAR_LEN < MAX_LEN else '=') +
                        ' ' * (MAX_LEN - BAR_LEN) +
                        '] %.1f%% (%d/%d)' % (progress * 100., now, end))
    sys.stderr.write('\r' +  progressbar_str)
    sys.stderr.flush()


class DatasetFromDirectory(chainer.dataset.DatasetMixin):
    def __init__(self, root='.', dtype=np.float32, label_dtype=np.int32):
        self.root = root
        self.directories = os.listdir(root)

    def __len__(self):
        return len(self.directories)

    def get_example(self, i):
        np_file = self.directories[i]
        np_file_path = os.path.join(self.root, np_file)
        sample = np.load(np_file_path)
        return sample["x"], sample["y"]


def get_shape(x):
    '''Returns the shape of a tensor as a tuple of
    integers or None entries.
    Note that this function only works with TensorFlow.
    '''
    shape = x.get_shape()
    return tuple([i.__int__() for i in shape])


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return (x, lengths)


def target_list_to_sparse_tensor(target_list):
    indices = []
    values = []
    for i, target in enumerate(target_list):
        for time, val in enumerate(target):
            indices.append([i, time])
            values.append(val)
    shape = [len(target_list), np.asarray(indices).max(0)[1]+1]
    return (np.array(indices), np.array(values), np.array(shape))


class BatchIterator(object):
    def __init__(self, batchsize, root='.', shuffle=True, dtype='float32', label_dtype='int32'):
        self.batchsize = batchsize
        self.root = root
        self.shuffle = shuffle
        # self.dtype = dtype
        # self.label_dtype = label_dtype
        self.files = os.listdir(root)
        self.n_samples = len(self.files)
        self.init_maxlen()

    def __len__(self):
        return math.ceil(1.0 * self.n_samples / self.batchsize)

    def size(self):
        return self.n_samples

    def init_maxlen(self):
        maxlen = 0
        for i in xrange(self.n_samples):
            np_file_path = os.path.join(self.root, self.files[i])
            sample = np.load(np_file_path)
            maxlen = max(len(sample["x"]), maxlen)
        self.set_maxlen(maxlen)
        return

    def get_maxlen(self):
        return self.maxlen

    def set_maxlen(self, maxlen):
        self.maxlen = maxlen
        return

    def __iter__(self):
        self.is_stop = False
        self._i = 0
        self.indexes = range(self.n_samples)
        if self.shuffle:
            random.shuffle(self.indexes)
        return self

    def next(self):
        if self.is_stop:
            raise StopIteration
        x_batches = []
        y_batches = []
        x_lens = []
        for _ in xrange(self.batchsize):
            index = self.indexes[self._i]
            np_file_path = os.path.join(self.root, self.files[index])
            sample = np.load(np_file_path)
            x_batches.append(sample["x"])
            y_batches.append(sample["y"])
            x_lens.append(len(sample["x"]))
            self._i += 1
            if self._i == self.n_samples:
                self.is_stop = True
                break
        (x_batches, seq_lengths) = pad_sequences(x_batches, maxlen=self.maxlen, dtype='float32', padding='post')
        y_batches = target_list_to_sparse_tensor(y_batches)
        return x_batches, y_batches, seq_lengths

class ResultWriter(object):
    def __init__(self, headers, result_dir='result'):
        if not os.path.exists(result_dir):
             os.mkdir(result_dir)
        self.filename = "{0}.txt".format(datetime.now().strftime('%Y%m%d%H%M%S'))
        self.filepath = os.path.join(result_dir, self.filename)
        self.file = open(self.filepath, 'w')
        self.set_header(headers)
        self.n_column = len(headers)

    def set_header(self, headers):
        self.file.write("#")
        for column in headers:
            self.file.write(str(column)+"\t")
        self.file.write("\n")

    def write(self, *args):
        if len(args) != self.n_column:
            raise ValueError("length of args must be equal to the number of column")
        for arg in args:
            self.file.write(str(arg)+"\t")
        self.file.write("\n")

    def close(self):
        self.file.close()
