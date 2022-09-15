# Evaluating Truncation and Extractive Approaches in Text Summarization

#### Create dataset
'''text
python3 shuffle_dataset.py --dataset 'xsum' --orig_source_length 512 --max_target_length 36 --seed 0
'''

#### Extract document
'''text
python3 extraction.py --approach 'head+tail0.5'
'''