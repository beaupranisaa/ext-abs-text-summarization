# Evaluating Truncation and Extractive Approaches in Text Summarization

### Create dataset
```sh
python3 shuffle_dataset.py --dataset 'xsum' --orig_source_length 512 --max_target_length 36 --seed 0
```

### Extract document
#### Unshuffled dataset
```sh
python3 extraction.py --approach 'head+tail0.5'
```

#### Shuffle dataset
```sh
python3 extraction.py --approach 'head+tail0.5' --shuffle True -seed 0
```

To combine the extracted document into train_set.csv, val_set.csv and test.csv

```sh
python3 helper/helper.py --approach 'head+tail0.5'
```

```sh
python3 helper/helper.py --approach 'head+tail0.5' --shuffle True -seed 0
```