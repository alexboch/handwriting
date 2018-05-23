from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves.urllib.request import urlretrieve
from six.moves import xrange as range
from PointsAndRectangles import *
import os
import sys
import numpy as np
from winreg import *
from sklearn.preprocessing import normalize

url = 'https://catalog.ldc.upenn.edu/desc/addenda/'
last_percent_reported = None

def normalized_frobenius(arr1,arr2):
    diff=arr1-arr2
    err=np.linalg.norm(diff)


def get_vectors_from_points(points,normalize_vectors=False):
    vectors=[]#список векторов
    if len(points)==1:
        #points.append(points[0])#Чтобы получился вектор из одной точки
        vectors.append([0,0])
    #for i in np.arange(len(points)-1):
    #    p1=points[i]#текущая точка
    #    p2=points[i+1]#следующая точка
    #    v=[p2[0]-p1[0],p2[1]-p1[1]]
    #    vectors.append(v)
    else:
        p1=points[0]
        for i in np.arange(1,len(points)):
            p2=points[i]
            v = [p2[0] - p1[0], p2[1] - p1[1]]
            vectors.append(v)
            p1=points[i]
    if normalize_vectors:
        vectors=normalize(vectors)  # нормализовать векторы
    return vectors

def get_bounds(points):
    """
    :param points:массив точек, каждая точка--массив вида{x,y}
    :return:
    """
    min_x=min(points,key=lambda p:p[0])[0]#Находит точку, берем первую координату(x)
    min_y=min(points,key=lambda p:p[1])[1]
    max_x=max(points,key=lambda p:p[0])[0]
    max_y = max(points, key=lambda p: p[1])[1]
    top_left=Point(min_x,min_y)
    bottom_right=Point(max_x,max_y)
    r=Rect(top_left,bottom_right)
    return r
    pass


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def getListSeparator():
    '''Retrieves the Windows list separator character from the registry'''
    aReg = ConnectRegistry(None, HKEY_CURRENT_USER)
    aKey = OpenKey(aReg, r"Control Panel\International")
    val = QueryValueEx(aKey, "sList")[0]
    return val



def decode_framewise(probabilities):
    """

    :param probabilities:Список/массив, элементом является список/массив вероятностей
    размерности 2: 0-й эл-т--это вероятность границы, 1-й эл-т--вероятность не-границы
    :return:Список индексов класса для всех моментов времени
    """
    max_t=len(probabilities[0])
    p=probabilities
    d=np.zeros(shape=(2,max_t))#Массив из нулей
    for t in np.arange(max_t):#Заполнение матрицы
        if t==0:
            d[1][t]=p[1][0]
            d[0][t]=p[0][0]
        else:
            d[0][t]=d[1][t-1]*p[0][t]
            d[1][t]=max(d[1][t-1],d[0][t-1])*p[1][t]
    #Восстановление пути
    class_numbers=[]
    class_index=np.argmax(d[:,max_t-1])
    class_numbers.append(class_index)
    for t in range(max_t-2,-1,-1):
        if class_index==0:#Если граница, можем переходить только на не-границу(индекс 1)
            class_index=1
        else:
            class_index=np.argmax(d[:,t])#Иначе выбираем, где макс. вероятность
        class_numbers.append(class_index)
    class_numbers.reverse()
    return class_numbers


def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for
    users with slow internet connections. Reports every 1% change in download
    progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent


def maybe_download(filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    if force or not os.path.exists(filename):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url + filename, filename,
                                  reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(filename)

    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        raise Exception(
                        'Failed to verify ' + filename + \
                        '. Can you get to it with a browser?')
    return filename

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

def pad_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    '''Pads each sequence to the same length: the length of the longest
    sequence.
        If maxlen is provided, any sequence longer than maxlen is truncated to
        maxlen. Truncation happens off either the beginning or the end
        (default) of the sequence. Supports post-padding (default) and
        pre-padding.

        Args:
            sequences: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger
            than maxlen either in the beginning or in the end of the sequence
            value: float, value to pad the sequences to the desired value.
        Returns
            x: numpy array with dimensions (number_of_sequences, maxlen)
            lengths: numpy array with the original sequence lengths
    '''
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

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
    return x, lengths
