__DATE__ = '2020/12/26'
__AUTHOR__ = 'Yifan_Ding'
__E_MAIL = 'yding4@nd.edu'

# 1. load dataset, either from CONLL2003 or other entity linking datasets.
# 2. load model from a trained one, load model architecture and state_dict from a trained one.
# 3. predict loss, generate predicted label
# 4. compare predicted label with ground truth label to obtain evaluation results.

if args.test_file is not None:
    data_files["test"] = args.test_file
else:
    raise ValueError('Evaluation must specify test_file!')
