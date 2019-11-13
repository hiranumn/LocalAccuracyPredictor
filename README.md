# ErrorPredictor.py

A script for predicting protein model accuracy.

```
usage: ErrorPredictor.py [-h] [--multiDecoy] [--multiDecoy2] [--noEnsemble] [--leavetemp] [--verbose] [--process PROCESS]
                         [--featurize] [--reprocess]
                         infolder outfolder

Error predictor network

positional arguments:
  infolder              input folder name full of pdbs
  outfolder             output folder name

optional arguments:
  -h, --help            show this help message and exit
  --multiDecoy, -mm     running multi-decoy network option (Default: False)
  --multiDecoy2, -mm2   running multi-decoy (trained w/o native in dist but w native in trainign set) model option
                        (Default: False)
  --noEnsemble, -ne     running without model ensembling (Default: False)
  --leavetemp, -lt      leaving temporary files (Default: False)
  --verbose, -v         verbose flag (Default: False)
  --process PROCESS, -p PROCESS
                        # of cpus to use for featurization (Default: 1)
  --featurize, -f       running only featurization (Default: False)
  --reprocess, -r       reprocessing all feature files (Default: False)
```

# Required softwares
- Python3.5-
- Pyrosetta
- Tensorflow 1.4

# Outputs
Output of the network is written in [input_file_name].npz.
You can extract the predictions as follows.

```
import numpy as np
x = np.load("testoutput.npz")
lddt = x["lddt"] # per residue lddt
estogram = x["estogram"] # per pairwise distance e-stogram
mask = x["mask"] # mask predicting native < 15
```

# What are the numbers you want to check?
See example.ipynb for how to interpret them.
```
from analyze import *
import numpy as np

# Get filenames of original pdb and predictions
predname = '/projects/ml/for/docking/output_pdbs/longxing_HEEH_14976_000000014_0001_0001.npz'
pdbname  = '/projects/ml/for/docking/pdbs/longxing_HEEH_14976_000000014_0001_0001.pdb'

# Load prediction
pred = np.load(predname)

# Get inter and intra interaction masks
# Returns interface map, chainA map, and chainB map
imap, [map1, map2] = get_interaction_map(pdbname)

# Good for analyzing monomer
global_lddt    = np.mean(get_lddt(pred["estogram"], pred["mask"]))
global_lddt2   = np.mean(x["lddt"]) # should be the same thing.

# Good for analyzing binder + target
interface_lddt = np.mean(get_lddt(pred["estogram"], np.multiply(imap, pred["mask"])))
chainA_lddt    = np.mean(get_lddt(pred["estogram"], np.multiply(map1, pred["mask"])))
chainB_lddt    = np.mean(get_lddt(pred["estogram"], np.multiply(map2, pred["mask"])))

# Best metric for binder + target (with chainA as binder of course)
score          = interface_lddt+chainA_lddt
```

# Updates
- Added some analysis code, 2019.11.6
- Distance matrix calculation speed-up, 2019.10.25
- v 0.0.1 released, 2019.10.19

# ToDo
- Engineering so that we can run it for bigger protein targets.
