import sys 
import uproot4 
import awkward1 as ak 
import numpy 
import math 
import time


def geteta(mupx, mupy,mupz):
    mup = numpy.sqrt(mupx**2 + mupy**2 + mupz**2)
    mueta = numpy.log((mup + mupz)/(mup - mupz))/2
    return (mueta)



def getphi(mupx, mupy):
    muphi = numpy.arctan2(mupy, mupx)
    return (muphi)



def getpt(mupx, mupy):
    mupt = numpy.sqrt(mupx**2 + mupy**2)
    return (mupt)

def Phi_mpi_pi(x):
    kPI=numpy.array(3.14159265)
    kPI = kPI.repeat(len(x))
    kTWOPI = 2 * kPI
    while (ak.any(x >= kPI)): x = x - kTWOPI;
    while (ak.any(x < -kPI)): x = x + kTWOPI; 
    return x;

def DeltaPhi(phi1,phi2):
    phi = Phi_mpi_pi(phi1-phi2)
    return abs(phi)

def getrecoil(nLep,leppt,lepphi,leppx_,leppy_,met_,metphi_):
    dummy=-9999.0
    WenuRecoilPt=dummy; WenurecoilPhi=-10.0;  We_mass=dummy;
    if (nLep >= 0):
        dphi = DeltaPhi(lepphi,metphi_)
        WenuRecoilPx = -( met_*numpy.cos(metphi_) + leppx_)
        WenuRecoilPy = -( met_*numpy.sin(metphi_) + leppy_)
        WenuRecoilPt = (numpy.sqrt(WenuRecoilPx**2  +  WenuRecoilPy**2))
        
    return WenuRecoilPt


tree_ = uproot4.open("/eos/cms/store/group/phys_exotica/bbMET/2016_SkimmedFiles/skim_setup_2016_v16_07-00/crab_ttHTobb_M125_13TeV_powheg_pythia8_200918_215950_0000_0.root")["outTree"].arrays()

print ((tree_))

cms_events = ak.zip({   "event": ak.zip({"run":tree_["st_runId"],"lumi":tree_["st_lumiSection"], "event": tree_["st_eventId"]   }),
                        "jets" : ak.zip({"px":tree_["st_THINjetPx"], "py":tree_["st_THINjetPy"], "pz":tree_["st_THINjetPz"], "e":tree_["st_THINjetEnergy"],
                                         "pt": getpt(tree_["st_THINjetPx"], tree_["st_THINjetPy"]), "eta":geteta(tree_["st_THINjetPx"], tree_["st_THINjetPy"], tree_["st_THINjetPz"]), 
                                         "phi":getphi(tree_["st_THINjetPx"], tree_["st_THINjetPy"]), "csv": tree_["st_THINjetDeepCSV"]              }),
                        "met": ak.zip({"met":tree_["st_pfMetCorrPt"], "phi": tree_["st_pfMetCorrPhi"], "trig": tree_["st_mettrigdecision"]   }),
                        "ele": ak.zip({"px":tree_["st_elePx"], "py":tree_["st_elePy"], "pz":tree_["st_elePz"], "e":tree_["st_eleEnergy"],
                                       "idL":tree_["st_eleIsPassLoose"], "idT":tree_["st_eleIsPassTight"], "q":tree_["st_eleCharge"],
                                       "pt": getpt(tree_["st_elePx"], tree_["st_elePy"]), "eta":geteta(tree_["st_elePx"], tree_["st_elePy"], tree_["st_elePz"]),
                                       "phi":getphi(tree_["st_elePx"], tree_["st_elePy"])          }),
                        "mu": ak.zip({"px":tree_["st_muPx"], "py":tree_["st_muPy"], "pz":tree_["st_muPz"], "e":tree_["st_muEnergy"],
                                      "idT":tree_["st_isTightMuon"], "q":tree_["st_muCharge"],
                                      "pt": getpt(tree_["st_muPx"], tree_["st_muPy"]), "eta":geteta(tree_["st_muPx"], tree_["st_muPy"], tree_["st_muPz"]),
                                      "phi":getphi(tree_["st_muPx"], tree_["st_muPy"]) }) ,
                        "tau": ak.zip({"ntau":tree_["st_nTau_discBased_TightEleTightMuVeto"]}),
                        "photon": ak.zip({ "npho":tree_["st_nPho"] }),
                        #"px":tree_["st_phoPx"],"py":tree_["st_phoPy"],"pz":tree_["st_phoPz"],"e":tree_["st_phoEnergy"], 
                        #"pt": getpt(tree_["st_phoPx"], tree_["st_phoPy"]), "eta":geteta(tree_["st_phoPx"], tree_["st_phoPy"], tree_["st_phoPz"])  }),
                    },
                    depth_limit=1)


print ("event loading done")

muons = cms_events.mu 
met   = cms_events.met
ele   = cms_events.ele 
jet   = cms_events.jets

## muons 
mu_sel = (muons.pt>30) & (muons.idT==True) & (numpy.abs(muons.eta)<2.4)
nMuTight = ak.sum(mu_sel, axis=-1)
nMuLoose = ak.sum ( (muons.pt>10), axis = -1 ) 
## electrons 
ele_sel = (ele.idT==True) & (ele.pt>30) & (numpy.abs(ele.eta)<2.5) 
nEleTight = ak.sum(ele_sel, axis=-1)
nEleLoose = ak.sum ( (ele.pt>10), axis = -1 )


## Recoil 
recoil_Wmunu =  getrecoil(nMuTight,  muons.pt, muons.phi, muons.px, muons.pt, met.met, met.phi)
recoil_Wenu  =  getrecoil(nEleTight, ele.pt,   ele.phi,   ele.px,   ele.pt,   met.met, met.phi)


dphi_jet_met = DeltaPhi(jet.phi, met.phi)
min_dphi_jet_met = ak.min(dphi_jet_met, axis=-1)


jet_loose_mask = (jet.pt > 30.0  ) & (numpy.abs(jet.eta)<2.5)
jet_tight_mask = (jet.pt > 50.0  ) & (numpy.abs(jet.eta)<2.5)
jet_b_mask     = (jet.csv > 0.6321) & (numpy.abs(jet.eta)<2.4)

n_jet_loose = ak.sum(jet_loose_mask,axis=-1)
n_jet_tight = ak.sum(jet_tight_mask, axis=-1)
n_jet_b     = ak.sum(jet_b_mask,axis=-1)

## zip all of them together 
mu_ = ak.zip( {"nMuLoose":nMuLoose,   "nMuTight":nMuTight,   "recoil":recoil_Wmunu} ) 
ele_= ak.zip( { "nEleLoose":nEleLoose, "nEleTight":nEleTight, "recoil":recoil_Wenu} )
jet_= ak.zip( {"min_dphi_jet_met":min_dphi_jet_met, "n_jet_loose":n_jet_loose, "n_jet_tight":n_jet_tight, "n_jet_b":n_jet_b})

## upto this it works fine, but if I try to make a zip of above 3 zips it does not work and complain about the broadcast nested list, which is not understood to me, 
## The size of these zip are same i.e. total number of events in the .root file , and took the structure of the variables present alredy in the awkward array. 


print ("zipping ", len(mu_), len(ele_), len(jet_))

#allevents = ak.zip ({ "basic": cms_events, "processed":cms_events_processed  })

print ("ele_jet", ak.zip ({  "ele":ele_, "jet":jet_  }))
print ("mu_jet", ak.zip ({  "mu":mu_, "jet":jet_  }))
print ("ele_mu", ak.zip ({  "ele":ele_, "mu":mu_  }))


## Since I need these variables in previous 3 zip to apply selection i try to make a zip of all three of them just to make one single object to play around in rest of the macro. 

## However, a combined zip start to throw error, and found that it is related to the last zip. 
'''
Traceback (most recent call last):
  File "test.py", line 87, in <module>
    print ("ele_mu", ak.zip ({  "ele":ele_, "mu":mu_  }))
  File "/afs/cern.ch/work/k/khurana/EXOANALYSIS/CMSSW_11_0_2/src/bbDMNanoAOD/analyzer/dependencies/lib/python3.6/site-packages/awkward1/operations/structure.py", line 348, in zip
    out = awkward1._util.broadcast_and_apply(layouts, getfunction, behavior)
  File "/afs/cern.ch/work/k/khurana/EXOANALYSIS/CMSSW_11_0_2/src/bbDMNanoAOD/analyzer/dependencies/lib/python3.6/site-packages/awkward1/_util.py", line 972, in broadcast_and_apply
    out = apply(broadcast_pack(inputs, isscalar), 0)
  File "/afs/cern.ch/work/k/khurana/EXOANALYSIS/CMSSW_11_0_2/src/bbDMNanoAOD/analyzer/dependencies/lib/python3.6/site-packages/awkward1/_util.py", line 745, in apply
    outcontent = apply(nextinputs, depth + 1)
  File "/afs/cern.ch/work/k/khurana/EXOANALYSIS/CMSSW_11_0_2/src/bbDMNanoAOD/analyzer/dependencies/lib/python3.6/site-packages/awkward1/_util.py", line 786, in apply
    nextinputs.append(x.broadcast_tooffsets64(offsets).content)
ValueError: in ListOffsetArray64, cannot broadcast nested list

(https://github.com/scikit-hep/awkward-1.0/blob/0.3.1/src/cpu-kernels/operations.cpp#L778)
'''
