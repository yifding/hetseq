import torch

from hetseq.model import TransformersBertForELClassification, TransformersBertForTokenClassification
from hetseq.bert_modeling import BertConfig
from transformers import BertConfig as HGBertConfig
from transformers import BertForTokenClassification as HGBertForTokenClassification


class Emp(object):
    def __init__(self):
        pass


if __name__ == '__main__':
    args = Emp()
    args.config_file = '/scratch365/yding4/hetseq/preprocessing/uncased_L-12_H-768_A-12/bert_config.json'
    args.ent_vecs_filename = '/scratch365/yding4/EL_resource/data/deep_ed_PyTorch_data/generated/ent_vecs/ent_vecs__ep_78.pt'
    args.EntityEmbedding = torch.load(args.ent_vecs_filename, map_location='cpu')
    args.num_entity_labels = args.EntityEmbedding.shape[0]
    args.dim_entity_emb = args.EntityEmbedding.shape[1]
    args.num_labels = 3
    config = HGBertConfig.from_json_file(args.config_file)
    # config = BertConfig.from_json_file(args.config_file)
    '''
    model = TransformersBertForTokenClassification.from_pretrained('bert-base-uncased', config=config, num_labels=3)
    '''
    model = TransformersBertForELClassification.from_pretrained(
        'bert-base-uncased', config=config, args=args
    )

    '''
    model = HGBertForTokenClassification.from_pretrained(
        'bert-base-uncased', config=config,
    )
    '''


