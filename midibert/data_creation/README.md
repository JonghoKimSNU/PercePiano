# Data Creation

All data in CP token are already in `Data/CP_data`, including the train, valid, test split.

You can also preprocess as below.

## 1. Download Dataset and Preprocess


## 2. Prepare Dictionary

```cd data_creation/prepare_data/dict/```
Run ```python make_dict.py```  to customize the events & words you'd like to add.

In this paper, we only use *Bar*, *Position*, *Pitch*, *Duration*.  And we provide our dictionaries in CP representation. (```data_creation/prepare_data/dict/CP.pkl```)

## 3. Prepare CP
Note that the CP tokens here only contain Bar, Position, Pitch, and Duration.  Please look into the repos below if you prefer the original definition of CP tokens.

All the commands are in ```scripts/prepare_data.sh```. You can directly edit the script and run it.

(Note that `export PYTHONPATH='.'` is necessary.)

