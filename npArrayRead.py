import os
import sys
import numpy as np
import matplotlib.pyplot as plt

array = np.load('cutarray.npy')
array = np.delete(array, 0, axis=0)

n_events, n_cuts = np.shape(array)
array_mean = np.mean(array, axis=0)

corr_matrix = np.corrcoef(array.T)

print '\n', abs(corr_matrix)
dim, dimi = np.shape(corr_matrix)

cut_names = ["No leptons in\nCMS detection", "nMuons > 1", "nElectrons > 1", "nLeptons != 1", "Electron nJets < 8",
             "Muon nJets < 7", "HT Cut"]

fig, ax = plt.subplots()
im = ax.imshow(abs(corr_matrix), cmap='YlGnBu', interpolation='nearest')

ax.set_xticks(np.arange(len(cut_names)))
ax.set_yticks(np.arange(len(cut_names)))
ax.set_xticklabels(cut_names)
ax.set_yticklabels(cut_names)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

corr_matrix_round = np.round(corr_matrix, 4)
for i in range(dim):
    for j in range(dim):
        ax.text(j, i, corr_matrix_round[i,j], ha='center', va='center', fontsize=8)

ax.set_title("Heatmap showing the correlation between cuts")
fig.tight_layout()
fig.colorbar(im, ax=ax)
plt.savefig('cut heat map.png')
