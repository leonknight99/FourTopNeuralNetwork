"""
This example shows how to perform regression of molecular properties with the
QM9 database, using a simple GNN in disjoint mode.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.data import DisjointLoader
from spektral.datasets import QM9
from spektral.layers import ECCConv, GlobalSumPool

################################################################################
# PARAMETERS
################################################################################
learning_rate = 1e-3  # Learning rate
epochs = 10  # Number of training epochs
batch_size = 32  # Batch size

################################################################################
# LOAD DATA
################################################################################
dataset = QM9(amount=1000)  # Set amount=None to train on whole dataset

# Parameters
F = dataset.n_node_features  # Dimension of node features
S = dataset.n_edge_features  # Dimension of edge features
n_out = dataset.n_labels  # Dimension of the target

# Train/test split
idxs = np.random.permutation(len(dataset))
split = int(0.9 * len(dataset))
idx_tr, idx_te = np.split(idxs, [split])
dataset_tr, dataset_te = dataset[idx_tr], dataset[idx_te]

loader_tr = DisjointLoader(dataset_tr, batch_size=batch_size, epochs=epochs)
loader_te = DisjointLoader(dataset_te, batch_size=batch_size, epochs=1)

################################################################################
# BUILD MODEL
################################################################################
X_in = Input(shape=(F,), name="X_in")
A_in = Input(shape=(None,), sparse=True, name="A_in")
E_in = Input(shape=(S,), name="E_in")
I_in = Input(shape=(), name="segment_ids_in", dtype=tf.int32)

X_1 = ECCConv(32, activation="relu")([X_in, A_in, E_in])
X_2 = ECCConv(32, activation="relu")([X_1, A_in, E_in])
X_3 = GlobalSumPool()([X_2, I_in])
output = Dense(n_out)(X_3)

# Build model
model = Model(inputs=[X_in, A_in, E_in, I_in], outputs=output)
opt = Adam(lr=learning_rate)
loss_fn = MeanSquaredError()


################################################################################
# FIT MODEL
################################################################################
@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
def train_step(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions)
        loss += sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


print("Fitting model")
current_batch = 0
model_loss = 0
for batch in loader_tr:
    outs = train_step(*batch)

    model_loss += outs
    current_batch += 1
    if current_batch == loader_tr.steps_per_epoch:
        print("Loss: {}".format(model_loss / loader_tr.steps_per_epoch))
        model_loss = 0
        current_batch = 0

################################################################################
# EVALUATE MODEL
################################################################################
print("Testing model")
model_loss = 0
for batch in loader_te:
    inputs, target = batch
    predictions = model(inputs, training=False)
    model_loss += loss_fn(target, predictions)
model_loss /= loader_te.steps_per_epoch
print("Done. Test loss: {}".format(model_loss))





'''

je = 0
            for n in range(nElectron):  # Adds electron node and
                G.nodes[1]['properties'] = [j[je] for j in i_e_prop]
                #G.add_node('electron')
                #G.nodes['electron']['properties'] = [j[je] for j in i_e_prop]  # [ptElectron[je], etaElectron[je]]
                je += 1
            jm = 0
            for n in range(nMuon):
                G.nodes[1]['properties'] = [j[jm] for j in i_m_prop]
                #G.add_node('muon')
                #G.nodes['muon']['properties'] = [j[jm] for j in i_m_prop]  # [ptMuon[jm], etaMuon[jm]]
                jm += 1
            jn = 0
            for n in range(nJet):  #Cutting off the first jet oops
                G.nodes[jn + 1]['properties'] = [j[jn] for j in i_j_prop]
                #G.add_edge('event', 'jet' + str(jn + 1))
                #G.nodes['jet' + str(jn + 1)]['properties'] = [j[jn] for j in i_j_prop]  # [ptJet[jn], etaJet[jn]]
                jn += 1


        keys_list = list(filter(lambda k: 'Selected' in k, list(eventsTreeDict.keys())))  # Filters to preselected events
        properties = ['mass', 'charge', 'eta', 'phi', '_pt', 'CSVV2']  # Properties to be added to Graph
        filter_list = list(filter(lambda k: any(contains(k, str(p)) for p in properties), keys_list))

        electron_props = list(filter(lambda k: 'Electron' in k, list(filter_list)))
        muon_props = list(filter(lambda k: 'Muon' in k, list(filter_list)))  # List of properties to add to nodes
        jet_props = list(filter(lambda k: 'dJet' in k, list(filter_list)))  # To eliminate the FatJets need d




def make_dataset(files, path):
    fn = 0
    graph_list = []
    for f in files:
        if fn == 1:
            break
        with np.load(os.path.join(path, f)) as data:
            x_list = data['x_list']
            a_list = data['a_list']
            y_list = data['y_list']
        print(x_list)

        for i in range(len(x_list)):
            graph_list.append(Graph(x=x_list[i], a=a_list[i], y=y_list[i]))
        print(graph_list)
        fn += 1


this_dir = os.getcwd()
file_path = os.path.join(this_dir, 'root2networkOut')
file_list = os.listdir(file_path)
print(file_list)

make_dataset(file_list, file_path)



nElectronList = np.array([1, 1])
nMuonList = np.array([0, 1])
nJetList = np.array([15, 15])

for i in range(len(nElectronList)):
    nElectron = int(nElectronList[i])
    nMuon = int(nMuonList[i])
    nJet = int(nJetList[i])
    nParticles = nJet + nMuon + nElectron

    a = np.zeros((nParticles, nParticles))
    a[:, 0] = 1
    a[0, :] = 1
    a[0, 0] = 0
    edge_list = []
    for n in range(2, int(nParticles)):
        edge_list.append((1, n))
        edge_list.append((n, 1))

    G = nx.Graph(event="4Top")
    G.add_edges_from(edge_list)

    for n in range(1, nElectron + 1):
        G.nodes[n]['particle'] = 'Electron'
    for n in range(nElectron + 1, nElectron + nMuon + 1):
        G.nodes[n]['particle'] = 'Muon'
    for n in range(nElectron + nMuon + 1, nElectron + nMuon + nJet):
        G.nodes[n]['particle'] = 'Jet'

    print(G.adj)
    print(G.nodes.data())
    try:
        pos = nx.nx_agraph.graphviz_layout(G)
    except ImportError:
        pos = nx.spring_layout(G, iterations=nParticles)
    nx.draw(G)
    nx.draw_networkx_labels(G, pos, font_size=20)
    plt.show()



edge_list = []
    for n in range(2, int(nParticles)):
        edge_list.append((1, n))
        edge_list.append((n, 1)
#eventsBranch = eventsTree.array("event", library="np"))
#eventsTree.show()
#filteredEventsTreeDict = dict(filter(lambda item: 'Selected' in item[0], eventsTreeDict.items()))
#filteredEventsTreeDict_noFat = dict(filter(lambda item: 'Fat' not in item[0], filteredEventsTreeDict.items()))

#nEventObjects = len(filteredEventsTreeDict_noFat['nSelectedElectron'])\
#                + len(filteredEventsTreeDict_noFat['nSelectedMuon']) + len(filteredEventsTreeDict_noFat['nSelectedJet'])

#electronT = dict(filter(lambda item: 'SelectedElectron' in item[0], eventsTreeDict.items()))
#muonT = dict(filter(lambda item: 'SelectedMuon' in item[0], eventsTreeDict.items()))
#jetT = dict(filter(lambda item: 'SelectedJet' in item[0], eventsTreeDict.items()))
#fatJetT = dict(filter(lambda item: 'SelectedFatJet' in item[0], eventsTreeDict.items()))
'''
