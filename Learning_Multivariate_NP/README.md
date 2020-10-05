# Learning Multivariate New Physics

In this folder you will find the code to replicate Section 4 and 5 of "Learning Multivariate New Physics." (D'Agnolo R. T., Grosso G., Pierini M., Wulzer A., & Zanetti M., 2019, arXiv preprint arXiv:1912.12155).

### DataSets:
We study LHC di-muon production comparing SM to two well-known new physics scenarios:
• a resonant signal, represented by a Z′ decaying to μ+μ−
• a smooth signal given by a contact interaction that we call “EFT”

The data are stored in h5 files on the following repositories (a Cern account is needed to access them!)

• SM (Reference/Background)
```
/eos/project/d/dshep/BSM_Detection/DiLepton_SM
```
• Z' resonant New Physics (Signal)
```
/eos/project/d/dshep/BSM_Detection/DiLepton_Zprime200
/eos/project/d/dshep/BSM_Detection/DiLepton_Zprime300
/eos/project/d/dshep/BSM_Detection/DiLepton_Zprime600
```
• EFT smooth New Physics (Signal+Background)
```
/eos/project/d/dshep/BSM_Detection/DiLepton_EFT06
/eos/project/d/dshep/BSM_Detection/DiLepton_EFT06_2
/eos/project/d/dshep/BSM_Detection/DiLepton_EFT06_5
```
For each event 6 features are stored: PT1, PT2, ETA1, ETA2, DELTA_PHI, MASS.

### Instruction to run the code:
The main files to run the training are:
##### NPL_train_Zprime.py for Zprime-like signals or SM-like data (when the numebr of signal events is set to 0). 

  Arguments:
  
- [1]: output folder
    
- [2]: toy label (integer)
    
- [3]: signal events folder (ex: /eos/project/d/dshep/BSM_Detection/DiLepton_Zprime300)
    
##### NPL_train_EFT.py for EFT-like signals.\\

  Arguments:
  
 - [1]: output folder
    
 - [2]: toy label (integer)
    
 - [3]: data events folder (ex: /eos/project/d/dshep/BSM_Detection/DiLepton_EFT1)

#### Run multiple trainings in parallel using condor:
The instructions are in the file Run_jobs_instructions.txt.
The script is run_z_5D_toys.py

