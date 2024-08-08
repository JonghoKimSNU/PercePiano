# MidiBERT-Piano
<p align="center">
    <br>
    <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/github/license/wazenmai/MIDI-BERT?logoColor=blue" /></a>
    <a href="http://arxiv.org/licenses/nonexclusive-distrib/1.0/"><img alt="ARXIV LICENSE" src="https://img.shields.io/badge/License-arxiv-lightgrey" /> </a>
    <a href=""><img alt="STAR" src="https://img.shields.io/github/stars/wazenmai/MIDI-BERT"/> </a>
    <a href="https://github.com/wazenmai/MIDI-BERT/issues"><img alt="ISSUE" src="https://img.shields.io/github/issues/wazenmai/MIDI-BERT" /></a>

</p>
Authors: <a href="https://sophia1488.github.io">Yi-Hui (Sophia) Chou</a>, <a href="https://github.com/wazenmai">I-Chun (Bronwin) Chen</a>

## Introduction
This is the official repository for the paper, [MidiBERT-Piano: Large-scale Pre-training for Symbolic Music Understanding](https://arxiv.org/pdf/2107.05223.pdf).

With this repository, you can
* pre-train a MidiBERT-Piano with your customized pre-trained dataset
* fine-tune & evaluate on 4 downstream tasks
* extract melody (mid to mid) using pre-trained MidiBERT-Piano

All the datasets employed in this work are publicly available.

## Installation
* Python3
* Install generally used packages for MidiBERT-Piano:
```python
git clone https://github.com/wazenmai/MIDI-BERT.git
cd MIDI-BERT
pip install -r requirements.txt
```

## Usage
Please see `scripts` folder, which includes bash file for
* prepare data
* pretrain
* finetune
* evaluation
* melody extraction

You may need to change the folder/file name or any config settings you prefer.


## Repo Structure
```
Data/
└── Dataset/       
    └── pop909/       
    └── .../
└── CP_data/
    └── pop909_train.npy
    └── *.npy

data_creation/
└── preprocess_pop909/
└── prepare_data/       # convert midi to CP_data 
    └── dict/           # CP dictionary 

melody_extraction/
└── skyline/
└── midibert/

MidiBERT/
└── *py

```

## More
For more details on 
* data preparation, please go to `data_creation` and follow Readme
* MidiBERT pretraining, finetuning, evaluation, please go to `MidiBERT` and follow Readme
* skyline, please go to `melody_extraction/skyline` and follow Readme

Note that Baseline (LSTM) and code in remi versions are removed for cleaness.  But you could find them in `main` branch.

## Citation

If you find this useful, please cite our paper.

```
@article{midibertpiano,
  title={{MidiBERT-Piano}: Large-scale Pre-training for Symbolic Music Understanding},
  author={Yi-Hui Chou and I-Chun Chen and Chin-Jui Chang and Joann Ching, and Yi-Hsuan Yang},
  journal={arXiv preprint arXiv:2107.05223},
  year={2021}
}
```

