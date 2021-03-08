import numpy as np
import time
import spektral
import tensorflow as tf
import spektralDataset
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.data import DisjointLoader, BatchLoader
from spektral.layers import MessagePassing
from spektral.transforms.normalize_adj import NormalizeAdj
from spektral.transforms.adj_to_sp_tensor import AdjToSpTensor

np.set_printoptions(linewidth=200)

'''
    Parameters
'''
learning_rate = 1e-3
epochs = 10
batch_size = 32
t0 = time.time()

'''
    Load Data
'''
file = ['root2networkOut/0 graphs.npz', 'root2networkOut/1 graphs.npz']
dataset = spektralDataset.TopDataset(max_samples=0, file_name_list=file)