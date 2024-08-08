# Percepiano

This is the implementation of PercePiano: Piano Performance Evaluation Dataset with Multi-level Perceptual Features. The code is based on [MidiBERT-Piano](https://github.com/wazenmai/MIDI-BERT) and [Virtuosonet](https://github.com/jdasam/virtuosoNet).


## Evaluation Case

- Example: [A segment from Beethoven_WoO80_var27](examples/Beethoven_WoO80_var27_8bars_3_15.wav)
- In the user study with our professional piano player, after listening to the music and predicting the scores based on the labels, we found that our model matched the label values better than the baseline.
- The quantitative results are shown below.

| Category                                              | Ours   | Baseline (Bi-LSTM) | Label  |
|-------------------------------------------------------|--------|----------|--------|
| Timing_Stable_Unstable                                | 0.4189 | 0.4397   | 0.3429 |
| Articulation_Short_Long                               | 0.3856 | 0.4776   | 0.3143 |
| Articulation_Soft_cushioned_Hard_solid                | 0.7237 | 0.7403   | 0.7429 |
| Pedal_Sparse/dry_Saturated/wet                        | 0.4694 | 0.6018   | 0.4571 |
| Pedal_Clean_Blurred                                   | 0.4457 | 0.5265   | 0.4286 |
| Timbre_Even_Colorful                                  | 0.4093 | 0.4714   | 0.3143 |
| Timbre_Shallow_Rich                                   | 0.4425 | 0.5390   | 0.3714 |
| Timbre_Bright_Dark                                    | 0.5998 | 0.5854   | 0.6000 |
| Timbre_Soft_Loud                                      | 0.7299 | 0.7250   | 0.8000 |
| Dynamic_Sophisticated/mellow_Raw/crude                | 0.7172 | 0.7172   | 0.7143 |
| Dynamic_Little_dynamic_range_Large_dynamic_range      | 0.4539 | 0.5832   | 0.3714 |
| Music_Making_Fast_paced_Slow_paced                    | 0.3869 | 0.4457   | 0.3714 |
| Music_Making_Flat_Spacious                            | 0.4282 | 0.4883   | 0.4571 |
| Music_Making_Disproportioned_Balanced                 | 0.4768 | 0.5116   | 0.5429 |
| Music_Making_Pure_Dramatic/expressive                 | 0.5047 | 0.5815   | 0.5143 |
| Emotion_&_Mood_Optimistic/pleasant_Dark               | 0.6524 | 0.6256   | 0.7143 |
| Emotion_&_Mood_Low_Energy_High_Energy                 | 0.7151 | 0.7362   | 0.7429 |
| Emotion_&_Mood_Honest_Imaginative                     | 0.3677 | 0.4279   | 0.4000 |
| Interpretation_Unsatisfactory/doubtful_Convincing     | 0.4223 | 0.4635   | 0.4000 |


## Data Preprocessing

### Labels
In `labels` folder, run
```
python map_midi_to_label.py
```
- Filename follows the format: [performance name]\_[# bars]bars\_[segment number]_[player number]

### MIDI & MUSICXML

#### Directory structure
In `virtuoso` folder,
```
virtuoso/
    data/
        └- all_2rounds/
            └- *.mid
        └- score_midi/
            └- *Score*.mid
        └- score_xml/
            └- *Score*.musicxml
```

### Align

#### Aligning Score MIDI - Performance MIDI
- Run v19 first because v22 makes error on first 3 notes
- Then align the remaining data by v22
- **Some paths are hard-coded. you should manually fix them!**
```
cd virtuoso/pyScoreParser/midi_utils
python copy_and_align_v19.py
python copy_and_align.py
```

### Align & Split 

#### Aligning ScoreXML - Score MIDI - Performance MIDI - Labels
- **Some paths are hard-coded. you should manually fix them!**
    - `labels/label_2round_mean_reg_19_with0_rm_highstd0.json` is the path for the labels.
```
cd virtuoso
python pyScoreParser/m2pf_dataset_compositionfold.py  # Piece split
python pyScoreParser/m2pf_dataset_performerfold.py  # Performer split
```

## Bi-LSTM + Ours Train
```
cd virtuoso
bash ./scripts/composition4fold/3_run_comp_multilevel_measure.sh # Piece split
bash ./scripts/performer4fold/0_run_performer_multilevel_measure.sh  # Performer split
```
## Bi-LSTM + Ours Eval
```
cd virtuoso
bash ./scripts/composition4fold/0_eval_comp_multilevel.sh # Piece split
bash ./scripts/performer4fold/0_eval_performer_multilevel.sh # Performer split
```

## MidiBERT

### HOW TO RUN
- pre-processed data from virtuosonet
```
cd virtuoso
python virtuoso/pyScoreParser/save_aligned_features.py
```

### Preprocessing

- Copy the dataset from virtuoso to midibert
```
cp -r virtuoso/data/all_2rounds midibert/Data/Dataset/percepiano
cd virtuoso
python virtuoso/virtuoso/pyScoreParser/save_aligned_features.py
```

- Resulting directory structure: 
```
Data/
└- Dataset/
    └- percepiano/
        └- *.mid
        └- virtuosonet/
            └- *.pkl
```

- Pre-processing for MidiBERT
    - Refer to: data_creation/README.md
```
cd midibert
export PYTHONPATH='.'
python data_creation/prepare_data/main_percepiano.py --dataset=percepiano --task=percepiano --output_dir=Data/CP_data/percepiano_random4split_addon --addons_path Data/Dataset/percepiano/virtuosonet --split performerfold

python data_creation/prepare_data/main_percepiano.py --dataset=percepino --task=percepiano --output_dir=Data/CP_data/percepiano_composition4split_addon --addons_path Data/Dataset/percepiano/virtuosonet --split composition4fold
```

### Train
- Download the pre-trained checkpoints from [MidiBERT-Piano](https://github.com/wazenmai/MIDI-BERT)
- Train e.g.
```
./scripts/finetune_cv/performer4fold/p_finetune_addons_04.sh
```
### Eval
- e.g.
./scripts/finetune_cv/performer4fold/eval_performer_addons.sh