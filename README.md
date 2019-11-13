# ErrorPredictor.py

A script for predicting protein model accuracy.

```
usage: ErrorPredictor.py [-h] [--pdb] [--multiDecoy] [--noEnsemble] [--leavetemp] [--verbose] [--process PROCESS]
                         [--gpu GPU] [--featurize] [--reprocess]
                         infolder outfolder

Error predictor network

positional arguments:
  infolder              input folder name full of pdbs
  outfolder             output folder name

optional arguments:
  -h, --help            show this help message and exit
  --pdb, -pdb           Running on a single pdb file instead of a folder (Default: False)
  --multiDecoy, -mm     running multi-multi model option (Default: False)
  --noEnsemble, -ne     running without model ensembling (Default: False)
  --leavetemp, -lt      leaving temporary files (Default: False)
  --verbose, -v         verbose flag (Default: False)
  --process PROCESS, -p PROCESS
                        # of cpus to use for featurization (Default: 1)
  --gpu GPU, -g GPU     gpu device to use (default gpu0)
  --featurize, -f       running only featurization (Default: False)
  --reprocess, -r       reprocessing all feature files (Default: False)

v0.0.1
```
# Example usages (for IPD people)
Type the following commands to activate tensorflow environment with pyrosetta3.
```
source activate tensorflow
source /software/pyrosetta3/setup.sh
```

Running on a folder of pdbs (foldername: ```samples```)
```
python ErrorPredictor.py -r -v samples outputs
```

Running on a single pdb file (inputname: ```input.pdb```)
```
python ErrorPredictor.py -r -v --pdb input.pdb output.npz
```

Only doing the feature processing (foldername: ```samples```)
```
python ErrorPredictor.py -r -v -f samples outputs
```

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

# Trouble shooting
- If ErrorPredictor.py returns an OOM (out of memory) error, your protein is probably too big. Try getting on titan instead of rtx2080 or run without gpu if running time is not your problem. You can also truncate your protein structures although it is not recommended. 

# Required softwares
- Python3.5-
- Pyrosetta
- Tensorflow 1.4 (not Tensorflow 2.0)

# Updates
- Reorganized code so that it is a python package, 2019.11.10
- Added some analysis code, 2019.11.6
- Distance matrix calculation speed-up, 2019.10.25
- v 0.0.1 released, 2019.10.19
