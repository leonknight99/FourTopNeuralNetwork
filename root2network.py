import numpy as np
import matplotlib.pyplot as plt
import uproot
import networkx as nx
import os


def data_import(inputFiles, outputDir, type, exampleGraph=False):
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, outputDir)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    file = uproot.open(str(inputFiles))
    eventsTree = file["Events"]  # Grabs the events tree from the root Tfile

    eventsTreeDict = eventsTree.arrays(library="np")  # Event tree as a numpy array and python dictionary

    nElectron_L, nMuon_L, nJet_L = eventsTreeDict["nSelectedElectron"], eventsTreeDict["nSelectedMuon"], eventsTreeDict["nSelectedJet"]
    Electronpt_L, Muonpt_L, Jetpt_L = eventsTreeDict["SelectedElectron_pt"], eventsTreeDict["SelectedMuon_pt"], eventsTreeDict["SelectedJet_pt"]
    Electroneta_L, Muoneta_L, Jeteta_L = eventsTreeDict["SelectedElectron_eta"], eventsTreeDict["SelectedMuon_eta"], eventsTreeDict["SelectedJet_eta"]

    a_list, x_list, y_list = [], [], []  # Empty lists to store adjacency matrix, node features and graph lables

    for i in range(len(nElectron_L)):  # Iterate through graphs
        nElectron, nMuon, nJet = int(nElectron_L[i]), int(nMuon_L[i]), int(nJet_L[i])
        ptElectron, ptMuon, ptJet = Electronpt_L[i], Muonpt_L[i], Jetpt_L[i]
        etaElectron, etaMuon, etaJet = Electroneta_L[i], Muoneta_L[i], Jeteta_L[i]

        G = nx.Graph(event=type)

        je = 0
        for n in range(1, nElectron + 1):  # Adds electron node and
            G.add_edge('event', 'electron')
            G.nodes['electron']['properties'] = [ptElectron[je], etaElectron[je]]
            je += 1
        jm = 0
        for n in range(nElectron + 1, nElectron + nMuon + 1):
            G.add_edge('event', 'muon')
            G.nodes['muon']['properties'] = [ptMuon[jm], etaMuon[jm]]
            jm += 1
        jn = 1
        for n in range(nElectron + nMuon + 1, nElectron + nMuon + nJet):
            G.add_edge('event', 'jet' + str(jn))
            G.nodes['jet' + str(jn)]['properties'] = [ptJet[jn], etaJet[jn]]
            jn += 1

        Graph_dict = nx.get_node_attributes(G, 'properties')
        features_array = np.array(list(Graph_dict.values()))

        a = nx.convert_matrix.to_numpy_matrix(G)
        x = np.insert(features_array, 0, 0, axis=0)
        y = G.graph.get('event')

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

    #print(a_list, '\n', x_list, '\n', y_list)

    filename = os.path.join(final_directory, f'graphs_{inputFiles[:-7]}2')
    np.savez(filename, x_list=x_list, a_list=a_list, y_list=y_list)

    return


#name = "0CE9E1B8-A913-BC41-8A20-79692A089797_Skim.root"  # small file TTTT
name = "0CE9E1B8-A913-BC41-8A20-79692A089797_Skim.root 2"  # bigger file TTTT
#name = "A63864C1-F698-004A-ACE2-2EC42B1A56B5_Skim.root"  # TT file

output_directory = 'root2networkOut'
top_quarks = 1  # 0 for 2, 1 for 4 events

data_import(name, output_directory, top_quarks)

