#!/usr/bin/env python
import os, sys
import numpy as np
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
from importlib import import_module
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Event, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module


class ExampleAnalysis(Module):
    def __init__(self, b, mn, mx, mc):
        self.writeHistFile = True
        self.nbins = b
        self.max_bin = mx
        self.min_bin = mn
        self.max_count = mc
        self.counter = 0

    def printing(self):
        print " Histogram: ", self.nbins, self.max_bin, self.min_bin

    def beginJob(self, histFile=None, histDirName=None):
        Module.beginJob(self, histFile, histDirName)

        self.h_vpt = ROOT.TH1F('Pt of SL', 'Transverse Momentum of the Single Lepton Events',
                               self.nbins, self.min_bin, self.max_bin)
        self.h_vpt.SetXTitle('Transverse Momentum')
        self.addObject(self.h_vpt)
        self.h_eta = ROOT.TH1F('Eta of SL', 'Pseudorapidity of the Single Lepton Events', 100, -5, 5)
        self.h_eta.SetXTitle('Pseudorapidity')
        self.addObject(self.h_eta)

        self.cut_array = 2 * np.ones(7)

    def endJob(self):
        np.save('cutarray', self.cut_array)

    def analyze(self, event):
        self.counter += 1
        #if self.counter > self.max_count:
         #   return False
        #else:
         #   print self.counter
        electrons = Collection(event, "Electron")
        muons = Collection(event, "Muon")
        jets = Collection(event, "Jet")
        eventSumMuons = ROOT.TLorentzVector()
        eventSumElectrons = ROOT.TLorentzVector()
        eventSumJets = ROOT.TLorentzVector()
        event_cut_array = np.ones(7)

        '''Select single lepton events'''

        n_leptons = 0
        for mu in muons:
            if mu.pt <= 24 or abs(mu.eta) >= 2.4:
                continue
            n_leptons += 1

        for el in electrons:
            if el.pt <= 32 or abs(el.eta) >= 2.1:
                continue
            n_leptons += 1

        if n_leptons == 0:
            event_cut_array[0] = 0  # No leptons cut

        '''Muons'''

        n_muons = 0
        for muon in muons:
            if muon.pt <= 26:  # Transverse momentum cut
                continue
            if abs(muon.eta) >= 2.1:  # Pseudorapidity cut
                continue
            if muon.pfRelIso04_all >= 0.15:
                continue
            n_muons += 1

            eventSumMuons += muon.p4()

        if n_muons >= 2:  # If something has been missed from previous cuts
            # More than one muon found cut
            event_cut_array[1] = 0

        '''Electrons'''

        n_electrons = 0
        for electron in electrons:
            if electron.pt <= 35:  # Transverse momentum cut
                continue
            if abs(electron.eta) >= 2.1:  # Pseudorapidity cut
                continue
            n_electrons += 1

            eventSumElectrons += electron.p4()

        if n_electrons >= 2:  # If something has been missed from previous cuts
            # More than one electron found cut
            event_cut_array[2] = 0

        if n_electrons + n_muons != 1:
            event_cut_array[3] = 0  # Too many leptons still in the event or no leptons found

        '''Jets'''

        HT = 0.0
        jet_n = 0

        for jet_i, jet in enumerate(jets):
            if jet.pt <= 30:  # Transverse momentum cut
                continue
            if abs(jet.eta) >= 2.5:  # Pseudorapidity cut
                continue

            HT += abs(jet.pt)
            jet_n += 1
            eventSumJets += jet.p4()
            #self.h_eta.Fill(jet.p4().Eta())
            #self.h_vpt.Fill(jet.p4().Pt())

        if n_electrons == 1 and jet_n < 8:
            event_cut_array[4] = 0  # Not enough jets in the electron event

        if n_muons == 1 and jet_n < 7:
            event_cut_array[5] = 0  # Not enough jets in the muon event

        if HT <= 500:  # Total scalar sum of all jets
            event_cut_array[6] = 0  # Event doesn't contain enough energy

        eventSum = eventSumMuons + eventSumElectrons + eventSumJets

          # Leftover events
        self.h_eta.Fill(eventSumMuons.Eta())  # fill histogram
        self.h_vpt.Fill(eventSumMuons.Pt())
        self.cut_array = np.vstack((self.cut_array, event_cut_array))
        return True


histogramBins = 100
histogramMax = 0
histogramMin = 1000
eventsMax = 100

an = ExampleAnalysis(histogramBins, histogramMin, histogramMax, eventsMax)
an.printing()

preselection = "nJet > 6"

#files = ["root://cms-xrd-global.cern.ch//store//mc/RunIIFall17NanoAODv7/TTTT_TuneCP5_PSweights_13TeV-amcatnlo-pythia8_correctnPartonsInBorn/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/100000/07FA7B6A-5EFA-8742-8C1C-D5DEE10D218B.root"]#,
files = ["root://cms-xrd-global.cern.ch//store/mc/RunIIFall17NanoAODv7/TTTT_TuneCP5_PSweights_13TeV-amcatnlo-pythia8_correctnPartonsInBorn/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/270000/0CE9E1B8-A913-BC41-8A20-79692A089797.root"]
#files = ["root://cms-xrd-global.cern.ch//store/mc/RunIIFall17NanoAODv7/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/130000/FAC33266-D958-C046-B736-E626D0D6F058.root"]
#files = ["root://cms-xrd-global.cern.ch//store/mc/RunIIFall17NanoAODv7/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/260000/A63864C1-F698-004A-ACE2-2EC42B1A56B5.root"]
#files = ["root://cms-xrd-global.cern.ch//store/mc/RunIIFall17NanoAODv7/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/260000/9E0DC7FB-15C3-E442-A594-AC7A3811865C.root",
#         "root://cms-xrd-global.cern.ch//store/mc/RunIIFall17NanoAODv7/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/260000/A63864C1-F698-004A-ACE2-2EC42B1A56B5.root "]

print "\nFile Name:\n", files, "\n"

output_path = str("/eos/user/l/lknight/TopQuarkProject/test/4")

histogram_file_name = str(os.path.basename(files[0]).replace(".root", "histOutMuons.root"))

p = PostProcessor('.', inputFiles=files, cut=preselection, branchsel=None,
                  modules=[ExampleAnalysis(histogramBins, histogramMin, histogramMax, eventsMax)],
                  noOut=True, histFileName=histogram_file_name, histDirName="plots")
p.run()


