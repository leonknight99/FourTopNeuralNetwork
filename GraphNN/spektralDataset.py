import os
import numpy as np
import scipy.sparse as sp

from spektral.data import Dataset, Graph
from spektral.transforms.adj_to_sp_tensor import AdjToSpTensor
from spektral.transforms.normalize_adj import NormalizeAdj

'''
    Loads the individual numpy files to convert the numpy graph arrays to the 'Graph' object present in the 
    Spektral library
'''


class TopDataset(Dataset):

    def __init__(self, max_samples, file_name_list, **kwargs):
        self.max_samples = max_samples
        self.file_name_list = file_name_list
        super().__init__(**kwargs)

    def download(self):
        os.mkdir(self.path)

    def read(self):

        x_listi, a_listi, e_listi, y_listi = [], [], [], []

        for file_name in self.file_name_list:
            data = np.load(file_name, allow_pickle=True)  # Opens files
            x_listi.append(data['x_list'])
            a_listi.append(data['a_list'])  # Appends each graph feature to its own feature list
            e_listi.append(data['e_list'])
            y_listi.append(data['y_list'])
        x_list = np.concatenate(x_listi)
        a_list = np.concatenate(a_listi)  # Joins all the lists together
        e_list = np.concatenate(e_listi)
        y_list = np.concatenate(y_listi)

        if self.max_samples != 0:
            n_samples = self.max_samples  # How many graphs are then outputted
        else:
            n_samples = len(y_list)

        def make_graph(i):

            # Node features
            x = x_list[i]

            # Adjacency matrix
            a = sp.csr_matrix(a_list[i])

            # Edge features
            e = e_list[i]

            # Labels
            y = np.array([y_list[i]])

            if int(a.nnz) != int(e.shape[0]):  # Check edge features and adjacency matrix are of the same order
                print(i, a.nnz, e.shape[0])
                print(a_list[i], e_list[i])

            return Graph(x=x, a=a, e=e, y=y)

        # We must return a list of Graph objects
        return [make_graph(n) for n in range(n_samples)]

