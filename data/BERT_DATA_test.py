import os
from BERT_DATA import CombineBertData

split = "train"
files = "/scratch365/yding4/bert_project/bert_prep_working_dir/" \
        "hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en"



path = files
if not os.path.exists(path):
    raise FileNotFoundError(
        "Dataset not found: ({})".format(path)
    )

files = [os.path.join(path, f) for f in os.listdir(path)] if os.path.isdir(path) else [path]
print(files)
files = [f for f in files if split in f]
print(files)
assert len(files) > 0

self.datasets[split] = CombineBertData(files)