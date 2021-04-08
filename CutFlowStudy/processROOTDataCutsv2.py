#!/usr/bin/env python
import os, sys
import numpy as np
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
from importlib import import_module
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Event, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.tools import deltaR

class ExampleAnalysis(Module):
    def __init__(self, b, mn, mx, mc, name, cbool=True):
        self.writeHistFile = True
        self.cumulative = cbool
        self.nbins = b
        self.max_bin = mx
        self.min_bin = mn
        self.max_count = mc
        self.name = name
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
        self.h_cut_vals = ROOT.TH1F('Cuts', 'The events that have been cut at specific cuts', 10, 0, 10)
        self.addObject(self.h_cut_vals)

        self.cut_array = 2 * np.ones(7)

    def endJob(self):
        np.save(str(self.cumulative) + 'cutarray' + str(self.name), self.cut_array)

    def analyze(self, event):
        self.counter += 1  # Testing Counter
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

        event_cut_array = np.ones(7)

        eIndex = []  # Electron collection
        mIndex = []  # Muon collection
        jIndex = []  # Jet collection
        crosslinkJetIndex = []  # For removal of jets associated to electrons and muons
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
            event_cut_array[0] = 0
            if not self.cumulative:
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
                crosslinkJetIndex.append(muon.jetIdx)
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
            crosslinkJetIndex.append(electron.jetIdx)
            eventSumElectrons += electron.p4()

        '''Single lepton selection cuts'''

        if n_electrons + n_muons != 1:
            self.h_cut_vals.Fill(3.5)  # Too many leptons still in the event or no leptons found
            event_cut_array[1] = 0
            if not self.cumulative:
                return False

        '''Jets'''

        HT = 0.0
        n_BJets = 0

        for jet_i, jet in enumerate(jets):
            if jet.pt <= 30:  # Transverse momentum cut per jet
                continue

            if abs(jet.eta) >= 2.5:  # Pseudorapidity cut per jet
                continue

            failedCleaning = False  # Jet cleaning per jet - making sure the jet isn't actually an electron or muon
            for m in mIndex:
                if deltaR(muons[m], jet) < 0.4:
                    #print deltaR(muons[m], jet)
                    failedCleaning = True
            for e in eIndex:
                if deltaR(electrons[e], jet) < 0.4:
                    #print deltaR(electrons[e], jet)
                    failedCleaning = True
            if failedCleaning:
                continue

            HT += jet.pt  # Summing the momentum for the jets in the event
            n_jets += 1
            jIndex.append(jet_i)
            eventSumJets += jet.p4()
            if jet.btagCSVV2 > 0.8838:  # Assigning b-tagged jets to the event
                n_BJets += 1

        if n_electrons == 1 and n_jets < 8:
            self.h_cut_vals.Fill(4.5)  # Not enough jets in the electron event
            event_cut_array[3] = 0
            if not self.cumulative:
                return False
        if n_muons == 1 and n_jets < 7:
            self.h_cut_vals.Fill(5.5)  # Not enough jets in the muon event
            event_cut_array[4] = 0
            if not self.cumulative:
                return False

        if n_BJets < 2:  # B Jets cut, requires at least 2 b-tagged jets at the medium working point
            self.h_cut_vals.Fill(6.5)
            event_cut_array[5] = 0
            if not self.cumulative:
                return False

        if HT <= 500:  # Total scalar sum of all jets
            self.h_cut_vals.Fill(7.5)  # Event doesn't contain enough energy
            event_cut_array[6] = 0
            if not self.cumulative:
                return False

        eventSumLeptons = eventSumMuons + eventSumElectrons

        '''Filling Histograms'''

        self.h_cut_vals.Fill(9.5)  # Leftover events
        self.h_eta.Fill(eventSumLeptons.Eta())  # fill histogram
        self.h_vpt.Fill(eventSumLeptons.Pt())
        self.cut_array = np.vstack((self.cut_array, event_cut_array))
        return True


histogramBins = 100
histogramMax = 0
histogramMin = 1000
eventsMax = 100

preselection = "nJet > 6"

#files = ["root://cms-xrd-global.cern.ch//store/mc/RunIIFall17NanoAODv7/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/70000/9E27F3B5-A39B-5947-8BB0-7540BA2F8DFE.root"]
files = ["root://cms-xrd-global.cern.ch//store/mc/RunIIFall17NanoAODv7/TTTT_TuneCP5_PSweights_13TeV-amcatnlo-pythia8_correctnPartonsInBorn/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/70000/6D75F151-87CB-3B4E-8C80-FB8D4E134968.root"]

cboolval = True  # False to remove events if they don't pass all cuts, True to see all the cuts that kick out the event

print "\nFile Name:\n", files, "\n"

output_path = str("/eos/user/l/lknight/TopQuarkProject/data_hist")

histogram_file_name = str(cboolval) + str(os.path.basename(files[0]).replace(".root", "histOut.root"))

p = PostProcessor('.', inputFiles=files, cut=None, branchsel=None,
                  modules=[ExampleAnalysis(histogramBins, histogramMin, histogramMax, eventsMax, name=str(os.path.basename(files[0]).replace(".root", "")), cbool=cboolval)],
                  noOut=True, histFileName=histogram_file_name, histDirName="plots")
p.run()


