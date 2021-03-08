import numpy as np
import matplotlib.pyplot as plt
import uproot
import networkx as nx
import os
import time
from operator import contains


def data_to_graph(inputDir, outputDir, type, exampleGraph=False):
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, outputDir)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    a_list, x_list, y_list = [], [], []  # Empty lists to store adjacency matrix, node features and graph lables

    files_path = os.path.join(current_directory, inputDir)
    inputFiles_list = os.listdir(files_path)
    os.chdir(files_path)
    print(inputFiles_list)
    t0 = time.time()

    for inputFile in inputFiles_list:

        file = uproot.open(str(inputFile))
        eventsTree = file["Events"]  # Grabs the events tree from the root Tfile
        eventsTreeDict = eventsTree.arrays(library="np")  # Event tree as a numpy array and python dictionary

        keys_list = list(filter(lambda k: 'Selected' in k, list(eventsTreeDict.keys())))  # Filters to preselected events
        properties = ['mass', 'charge', 'eta', 'phi', '_pt', 'CSVV2']  # Properties to be added to Graph
        filter_list = list(filter(lambda k: any(contains(k, str(p)) for p in properties), keys_list))

        electron_props = list(filter(lambda k: 'Electron' in k, list(filter_list)))
        muon_props = list(filter(lambda k: 'Muon' in k, list(filter_list)))  # List of properties to add to nodes
        jet_props = list(filter(lambda k: 'dJet' in k, list(filter_list)))  # To eliminate the FatJets need d

        electron_list, muon_list, jet_list = [], [], []
        for p in electron_props:
            electron_list.append(eventsTreeDict[str(p)])
        for p in muon_props:
            muon_list.append(eventsTreeDict[str(p)])  # Appends properties of events to lists
        for p in jet_props:
            jet_list.append(eventsTreeDict[str(p)])

        nElectron_L, nMuon_L, nJet_L = eventsTreeDict["nSelectedElectron"], eventsTreeDict["nSelectedMuon"], eventsTreeDict["nSelectedJet"]

        jet_array = np.array(jet_list)
        electron_array = np.array(electron_list)  # Turns lists of arrays into arrays of arrays
        muon_array = np.array(muon_list)

        for i in range(len(nElectron_L)):  # Iterate through graphs
            nElectron, nMuon, nJet = int(nElectron_L[i]), int(nMuon_L[i]), int(nJet_L[i])
            i_e_prop = electron_array[:, i]
            i_m_prop = muon_array[:, i]  # Selects each event's properties
            i_j_prop = jet_array[:, i]

            G = nx.Graph(event=type)

            je = 0
            for n in range(nElectron):  # Adds electron node and
                G.add_edge('event', 'electron')
                G.nodes['electron']['properties'] = [j[je] for j in i_e_prop]  # [ptElectron[je], etaElectron[je]]
                je += 1
            jm = 0
            for n in range(nMuon):
                G.add_edge('event', 'muon')
                G.nodes['muon']['properties'] = [j[jm] for j in i_m_prop]  # [ptMuon[jm], etaMuon[jm]]
                jm += 1
            jn = 0
            for n in range(nJet):  #Cutting off the first jet oops
                G.add_edge('event', 'jet' + str(jn + 1))
                G.nodes['jet' + str(jn + 1)]['properties'] = [j[jn] for j in i_j_prop]  # [ptJet[jn], etaJet[jn]]
                jn += 1

            Graph_dict = nx.get_node_attributes(G, 'properties')
            features_array = np.array(list(Graph_dict.values()))

            a = nx.convert_matrix.to_numpy_matrix(G)
            x = np.insert(features_array, 0, 0, axis=0)
            y = G.graph.get('event')
            print(y)

            a_list.append(a)
            x_list.append(x)
            y_list.append(y)

            if exampleGraph:
                try:
                    pos = nx.nx_agraph.graphviz_layout(G)
                except ImportError:
                    nParticles = nJet + nMuon + nElectron
                    pos = nx.spring_layout(G, iterations=nParticles)
                nx.draw(G, with_labels=True)
                #  nx.draw_networkx_labels(G, pos, font_size=20)
                plt.show()

        print(f'Processed: {inputFile} | Time: {round(time.time() - t0, 5)} | Events: {len(y_list)}')

    filename = os.path.join(final_directory, f'{type} graphs test')
    np.savez(filename, x_list=x_list, a_list=a_list, y_list=y_list)
    print(f'Saved to file: {filename} | Total Time: {round(time.time() - t0, 5)} | Total Events: {len(y_list)}')


#name = "0CE9E1B8-A913-BC41-8A20-79692A089797_Skim.root"  # small file TTTT
#name = "0CE9E1B8-A913-BC41-8A20-79692A089797_Skim.root 2"  # bigger file TTTT
name = "A63864C1-F698-004A-ACE2-2EC42B1A56B5_Skim.root"  # TT file

output_directory = 'root2networkOut'
input_directory = 'root2networkIn'
top_quarks = 0  # 0 for 2, 1 for 4 events

data_to_graph(input_directory, output_directory, top_quarks)

