import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import uproot
import networkx as nx
import os
import time


def data_to_graph(inputDir, outputDir, type, exampleGraph=False):
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, outputDir)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    a_list, x_list, e_list, y_list = [], [], [], []
    # Empty lists to store adjacency matrix, node features, edge features and graph labels

    files_path = os.path.join(current_directory, inputDir)
    inputFiles_list = os.listdir(files_path)
    os.chdir(files_path)
    print(inputFiles_list)
    t0 = time.time()  # Timing of process
    g_err = 0  # Graphs with edge errors
    nF = len(inputFiles_list)
    nFx = 1

    for inputFile in inputFiles_list:

        file = uproot.open(str(inputFile))
        eventsTree = file["Events"]  # Grabs the events tree from the root Tfile

        eventsTreeDict = eventsTree.arrays(library="np")  # Event tree as a numpy array and python dictionary
        #  print(len(eventsTreeDict["event"]))

        nEvents = len(eventsTreeDict["event"])  # Number of events selected
        nElectron_list, nMuon_list, nJet_list = eventsTreeDict["nSelectedElectron"], eventsTreeDict["nSelectedMuon"], \
                                                eventsTreeDict["nSelectedJet"]  # Number of particles in events

        '''
            Required properties of the chosen event constituents for use on the graphs later
        '''

        features_list_e = ['SelectedElectron_pt', 'SelectedElectron_mass', 'SelectedElectron_charge',
                           'SelectedElectron_eta', 'SelectedElectron_phi']
        features_list_m = ['SelectedMuon_pt', 'SelectedMuon_mass', 'SelectedMuon_charge',
                           'SelectedMuon_eta', 'SelectedMuon_phi']
        features_list_j = ['SelectedJet_pt', 'SelectedJet_mass', 'SelectedJet_btagCSVV2',
                           'SelectedJet_eta', 'SelectedJet_phi']

        electron_list, muon_list, jet_list = [], [], []
        # Empty lists for the features of nodes and edges of graph
        for f in features_list_e:
            electron_list.append(eventsTreeDict[str(f)])
        for f in features_list_m:
            muon_list.append(eventsTreeDict[str(f)])  # Appends properties of events to lists
        for f in features_list_j:
            jet_list.append(eventsTreeDict[str(f)])

        jet_array = np.array(jet_list)
        electron_array = np.array(electron_list)  # Turns lists of arrays into arrays of arrays
        muon_array = np.array(muon_list)

        for i in range(nEvents):  # Iterate through graphs
            nElectron, nMuon, nJet = int(nElectron_list[i]), int(nMuon_list[i]), int(nJet_list[i])
            nParticles = nJet + nElectron + nMuon
            i_e_prop = electron_array[:, i]
            i_m_prop = muon_array[:, i]  # Selects each event's properties
            i_j_prop = jet_array[:, i]

            eta_list, phi_list = [], []  # For graph edge features

            G = nx.complete_graph(int(nParticles))  # Creates a fully connected graph
            G.graph['event'] = type

            if nElectron == 1:  # Adding electron properties to electron node
                G.nodes[0]['properties'] = [1, 0] + [j[0] for j in i_e_prop[:-2]] + [0]
                eta_list.append(i_e_prop[3][0])
                phi_list.append(i_e_prop[4][0])  # For edge features later
            elif nMuon == 1:  # Adding muon properties to muon node
                G.nodes[0]['properties'] = [-1, 0] + [j[0] for j in i_m_prop[:-2]] + [0]
                eta_list.append(i_m_prop[3][0])
                phi_list.append(i_m_prop[4][0])  # For edge features later
            else:
                print('Event has a muon and electron selected')  # Error would've occurred within event preselection
                continue

            for n in range(1, nParticles):  # Adding jet properties for each jet node in the event
                if i_j_prop[2][n-1] > 0.8838:
                    btag_bool = 1  # Three valued logic for b-jet, light jet or no lepton
                else:
                    btag_bool = -1
                G.nodes[n]['properties'] = [0, btag_bool] + [j[n - 1] for j in i_j_prop[:-3]] + [0, i_j_prop[2][n-1]]
                eta_list.append(i_j_prop[3][n-1])
                phi_list.append(i_j_prop[4][n-1])  # For edge features later

            edge_features_matrix = np.zeros([nParticles, nParticles])
            for xi in range(nParticles):
                for yj in range(nParticles):  # To work out the edge features "distance" between particles
                    edge_features_matrix[yj, xi] = np.sqrt((eta_list[xi] - eta_list[yj]) ** 2
                                                           + (phi_list[xi] - phi_list[yj]) ** 2)

            for edge in G.edges:  # Appends edge feature "distance" to each edge
                G.edges[edge]['d'] = edge_features_matrix[edge]

            '''
                Formatting graph for Spektral graph type
            '''

            graph_node_dict = nx.get_node_attributes(G, 'properties')
            node_features_array = np.array(list(graph_node_dict.values()))  # For node features

            edge_triu = np.array(nx.attr_sparse_matrix(G, edge_attr='d', rc_order=G.nodes).todense())
            edge_list = edge_triu.ravel()
            edge_list = edge_list[edge_list != 0]  # For edge features

            a = nx.convert_matrix.to_numpy_matrix(G)
            x = np.array(node_features_array)
            e = np.array([edge_list]).T
            #print(e)
            #e1 = nx.attr_sparse_matrix(G, edge_attr='d', rc_order=G.nodes)
            #print(e1)
            #  E = e.todense() # To turn back into an adjacency matrix for edge values
            y = G.graph.get('event')

            if np.count_nonzero(a) != e.shape[0]:
                #print('Graph Error\n', edge_features_matrix)
                g_err += 1
                continue

            a_list.append(a)
            x_list.append(x)
            e_list.append(e)  # To output data later for use in the Spektral GNN
            y_list.append(y)

            if exampleGraph:  # Printing example graphs
                try:
                    pos = nx.nx_agraph.graphviz_layout(G)
                except ImportError:
                    nParticles = nJet + nMuon + nElectron
                    pos = nx.spring_layout(G, iterations=nParticles)
                nx.draw(G, with_labels=True)
                #  nx.draw_networkx_labels(G, pos, font_size=20)
                plt.show()

        print(f'Processed: {inputFile} | {nFx} / {nF} | Time: {round(time.time() - t0, 5)}s | Events: {len(y_list)}')  # Timings
        nFx += 1

    '''
        Saving the full dataset to a npz file
    '''

    filename = os.path.join(final_directory, f'{type} graphs')
    np.savez(filename, x_list=x_list, a_list=a_list, e_list=e_list, y_list=y_list)
    print(f'Saved to file: {filename} | Total Time: {round(time.time() - t0, 5)}s | Total Events: {len(y_list)} '
          f'| Graph Errors: {g_err}')


output_directory = 'root2networkOut'
input_directory = 'root2networkIn'
top_quarks = 1  # 0 for 2, 1 for 4 events

data_to_graph(input_directory, output_directory, top_quarks)

