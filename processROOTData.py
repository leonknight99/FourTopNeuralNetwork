#!/usr/bin/env python
import os
import sys
import numpy as np
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
from importlib import import_module
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Event, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

# https://github.com/cms-nanoAOD/nanoAOD-tools/blob/master/python/postprocessing/modules/common/collectionMerger.py
# said to do it
_rootLeafType2rootBranchType = {'UChar_t': 'b', 'Char_t': 'B', 'UInt_t': 'i', 'Int_t': 'I', 'Float_t': 'F',
                                'Double_t': 'D', 'ULong64_t': 'l', 'Long64_t': 'L', 'Bool_t': 'O'}


class ExampleAnalysis(Module):
    def __init__(self, b, mn, mx, mc):
        self.writeHistFile = True
        self.nbins = b
        self.max_bin = mx
        self.min_bin = mn
        self.max_count = mc
        self.counter = 0

        self.input = {"typeElectron": "Electron", "typeMuon": "Muon", "typeAK4": "Jet", "typeAK8": "FatJet"}
        self.output = {"typeElectron": "SelectedElectron", "typeMuon": "SelectedMuon",
                       "typeAK4": "SelectedJet", "typeAK8": "SelectedFatJet"}
        self.n_inputs = len(self.input)

        placeholder = []
        for elem in self.output:
            placeholder.append({})
        self.branchType = dict(zip(self.input.values(), placeholder))

    def beginJob(self, histFile=None, histDirName=None):
        Module.beginJob(self, histFile, histDirName)

        self.h_vpt = ROOT.TH1F('Pt of SL', 'Transverse Momentum of the Single Lepton Events',
                               self.nbins, self.min_bin, self.max_bin)
        self.h_vpt.SetXTitle('Transverse Momentum')
        self.addObject(self.h_vpt)
        self.h_eta = ROOT.TH1F('Eta of SL', 'Pseudorapidity of the Single Lepton Events', 100, -5, 5)
        self.h_eta.SetXTitle('Pseudorapidity')
        self.addObject(self.h_eta)
        self.h_cut_vals = ROOT.TH1F('Cuts', 'The events that have been cut at specific cuts', 10, 0, 10)
        self.addObject(self.h_cut_vals)

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):  # Copied from collectionMerger.py

        # Find list of activated branches in input tree
        _brlist_in = inputTree.GetListOfBranches()
        branches = set([_brlist_in.At(i) for i in range(_brlist_in.GetEntries())])
        branches = [x for x in branches if inputTree.GetBranchStatus(x.GetName())]

        placeholder = []
        for elem in self.input:
            placeholder.append([])  # Creates dictionary for i
        self.brlist_sep = dict(zip(self.input.keys(), placeholder))

        for key in self.input.keys():
            self.brlist_sep[key] = self.filterBranchNames(branches, self.input[key])

        self.out = wrappedOutputTree
        #print self.brlist_sep["typeElectron"]
        for ebr in self.brlist_sep["typeElectron"]:
            self.out.branch("%s_%s"%(self.output["typeElectron"], ebr),
                            _rootLeafType2rootBranchType[self.branchType[self.input["typeElectron"]][ebr]],
                            lenVar="n%s"%self.output["typeElectron"])

        for mbr in self.brlist_sep["typeMuon"]:
            self.out.branch("%s_%s"%(self.output["typeMuon"], mbr),
                            _rootLeafType2rootBranchType[self.branchType[self.input["typeMuon"]][mbr]],
                            lenVar="n%s"%self.output["typeMuon"])

        for j4br in self.brlist_sep["typeAK4"]:
            self.out.branch("%s_%s"%(self.output["typeAK4"], j4br),
                            _rootLeafType2rootBranchType[self.branchType[self.input["typeAK4"]][j4br]],
                            lenVar="n%s"%self.output["typeAK4"])

        for j8br in self.brlist_sep["typeAK8"]:
            self.out.branch("%s_%s"%(self.output["typeAK8"], j8br),
                            _rootLeafType2rootBranchType[self.branchType[self.input["typeAK8"]][j8br]],
                            lenVar="n%s"%self.output["typeAK8"])

        '''f = open("branch list.txt", "a")
        for i in xrange(self.branchlist.GetEntries()):
            f.write('\n\n' + str(self.branchlist.At(i)))
            print 'Branch list: ', self.branchlist.At(i)
        self.out = wrappedOutputTree
        f.close()'''

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        #print 'End: \n', wrappedOutputTree.GetListOfBranches().GetEntries()
        pass

    def filterBranchNames(self, branches, collection):
        out = []
        for br in branches:
            name = br.GetName()
            if not name.startswith(collection + '_'):
                continue
            out.append(name.replace(collection + '_', ''))
            self.branchType[collection][out[-1]] = br.FindLeaf(br.GetName()).GetTypeName()
        return out

    def analyze(self, event):
        self.counter += 1
        #if self.counter > self.max_count:
         #   return False
        #else:
         #   print self.counter

        electrons = Collection(event, "Electron")
        muons = Collection(event, "Muon")
        jets = Collection(event, "Jet")  # Collections
        fatjets = Collection(event, "FatJet")

        eventSumMuons = ROOT.TLorentzVector()
        eventSumElectrons = ROOT.TLorentzVector()  # 4 Vectors for histograms
        eventSumJets = ROOT.TLorentzVector()

        eIndex = []  # Electron collection
        mIndex = []  # Muon collection
        jIndex = []  # Jet collection
        n_0leptons = 0  # If no leptons are present
        n_leptons = 0  # Selected lepton count
        n_muons = 0  # Selected muon count
        n_electrons = 0  # Selected electron count
        n_jets = 0  # Selected jet count

        '''Cut for no leptons detected via CMS in the event cut'''

        for mu in muons:
            if mu.pt <= 24 or abs(mu.eta) >= 2.4:
                continue
            n_0leptons += 1

        for el in electrons:
            if el.pt <= 32 or abs(el.eta) >= 2.1:
                continue
            n_0leptons += 1

        if n_0leptons == 0:
            self.h_cut_vals.Fill(0.5)  # No leptons cut
            return False

        '''Muon collection'''

        for m, muon in enumerate(muons):
            if muon.pt <= 26:  # Transverse momentum cut
                continue
            if abs(muon.eta) >= 2.1:  # Pseudorapidity cut
                continue
            if muon.pfRelIso04_all < 0.15:  # Relative Isolation cut R = 0.4 cone
                n_muons += 1
                n_leptons += 1
                mIndex.append(m)
                eventSumMuons += muon.p4()

        '''Electron collection'''

        for e, electron in enumerate(electrons):
            if electron.pt <= 35:  # Transverse momentum cut
                continue
            if abs(electron.eta) >= 2.1:  # Pseudorapidity cut
                continue
            n_electrons += 1
            n_leptons += 1
            eIndex.append(e)
            eventSumElectrons += electron.p4()

        '''Single lepton selection cuts'''

        if n_electrons + n_muons != 1:
            self.h_cut_vals.Fill(3.5)  # Too many leptons still in the event or no leptons found
            return False

        '''Jets'''

        HT = 0.0
        n_BJets = 0

        for jet_i, jet in enumerate(jets):
            if jet.pt <= 30:  # Transverse momentum cut
                continue
            if abs(jet.eta) < 2.5:  # Pseudorapidity cut
                HT += jet.pt
                n_jets += 1
                jIndex.append(jet_i)
                eventSumJets += jet.p4()
                if jet.btagCSVV2 > 0.8838:
                    n_BJets += 1
            #self.h_eta.Fill(jet.p4().Eta())
            #self.h_vpt.Fill(jet.p4().Pt())

        if n_electrons == 1 and n_jets < 8:
            self.h_cut_vals.Fill(4.5)  # Not enough jets in the electron event
            return False
        if n_muons == 1 and n_jets < 7:
            self.h_cut_vals.Fill(5.5)  # Not enough jets in the muon event
            return False

        if n_BJets < 2:  # B Jets cut, requires at least 2 b-tagged jets at the medium working point
            self.h_cut_vals.Fill(6.5)
            return False

        if HT <= 500:  # Total scalar sum of all jets
            self.h_cut_vals.Fill(7.5)  # Event doesn't contain enough energy
            return False

        eventSum = eventSumMuons + eventSumElectrons + eventSumJets

        '''Filling Histograms'''

        self.h_cut_vals.Fill(9.5)  # Leftover events
        self.h_eta.Fill(eventSumMuons.Eta())  # fill histogram
        self.h_vpt.Fill(eventSumMuons.Pt())

        '''Writing to Output Branches'''

        for br in self.brlist_sep["typeElectron"]:
            out = []
            for elem in eIndex:  # for elem in eIndex
                out.append(getattr(electrons[elem], br))
            self.out.fillBranch("%s_%s"%(self.output["typeElectron"], br), out)

        for br in self.brlist_sep["typeMuon"]:
            out = []
            for elem in mIndex:  # for elem in mIndex
                out.append(getattr(muons[elem], br))
            self.out.fillBranch("%s_%s"%(self.output["typeMuon"], br), out)

        for br in self.brlist_sep["typeAK4"]:
            out = []
            for elem in jIndex:  # Type AK4 Jets that have been detected
                out.append(getattr(jets[elem], br))
            self.out.fillBranch("%s_%s"%(self.output["typeAK4"], br), out)

        for br in self.brlist_sep["typeAK8"]:
            out = []
            for elem in range(len(fatjets)):
                out.append(getattr(fatjets[elem], br))
            self.out.fillBranch("%s_%s"%(self.output["typeAK8"], br), out)
        #print 'done'
        return True


histogramBins = 100
histogramMax = 0
histogramMin = 1000
eventsMax = 40

preselection = "nJet > 6"

#files = ["root://cms-xrd-global.cern.ch//store//mc/RunIIFall17NanoAODv7/TTTT_TuneCP5_PSweights_13TeV-amcatnlo-pythia8_correctnPartonsInBorn/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/100000/07FA7B6A-5EFA-8742-8C1C-D5DEE10D218B.root"]#,
#files = ["root://cms-xrd-global.cern.ch//store/mc/RunIIFall17NanoAODv7/TTTT_TuneCP5_PSweights_13TeV-amcatnlo-pythia8_correctnPartonsInBorn/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/270000/0CE9E1B8-A913-BC41-8A20-79692A089797.root"]  # small event
#files = ["root://cms-xrd-global.cern.ch//store//mc/RunIIFall17NanoAODv7/TTTT_TuneCP5_PSweights_13TeV-amcatnlo-pythia8_correctnPartonsInBorn/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/100000/07FA7B6A-5EFA-8742-8C1C-D5DEE10D218B.root",
#         "root://cms-xrd-global.cern.ch//store/mc/RunIIFall17NanoAODv7/TTTT_TuneCP5_PSweights_13TeV-amcatnlo-pythia8_correctnPartonsInBorn/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/270000/0CE9E1B8-A913-BC41-8A20-79692A089797.root"]

#files = ["root://cms-xrd-global.cern.ch//store/mc/RunIIFall17NanoAODv7/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/130000/FAC33266-D958-C046-B736-E626D0D6F058.root"]
#files = ["root://cms-xrd-global.cern.ch//store/mc/RunIIFall17NanoAODv7/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/260000/A63864C1-F698-004A-ACE2-2EC42B1A56B5.root"]  # small event
#files = ["root://cms-xrd-global.cern.ch//store/mc/RunIIFall17NanoAODv7/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/260000/9E0DC7FB-15C3-E442-A594-AC7A3811865C.root",
#         "root://cms-xrd-global.cern.ch//store/mc/RunIIFall17NanoAODv7/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/260000/A63864C1-F698-004A-ACE2-2EC42B1A56B5.root "]

output_path = str("/eos/user/l/lknight/TopQuarkProject/test/3.1")  # Creates output directory for ROOT TTrees

#input_file = open("FourTopEvents.txt")  # List of data files to append through
input_file = open("TwoTopEvents.txt")

input_lines = input_file.read().splitlines()
files_list = ["root://cms-xrd-global.cern.ch/" + s for s in input_lines]  # Appends rest of file name information
print '\n\n', files_list, '\nlenght: ', len(files_list)

for file_index in range(len(files_list)):  # Iterates through files
    if file_index == 40:
        break
    files = [files_list[int(file_index)]]
    print "\nFile Name:\n", files, "\n"
    histogram_file_name = str(os.path.basename(files[0]).replace(".root", "histOutMuons.root"))

    p = PostProcessor(outputDir=output_path, inputFiles=files, cut=preselection, branchsel=None,
                  modules=[ExampleAnalysis(histogramBins, histogramMin, histogramMax, eventsMax)],
                  noOut=False, histFileName=histogram_file_name, histDirName="plots")
    p.run()
