from transformers import (
    BertPreTrainedModel,
    BertModel,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss


_OUT_DICT_ENTITY_ID = -1
_IGNORE_CLASSIFICATION_LABEL = -100
NER_LABEL_DICT = {'B': 0, 'I':1, 'O':2}


# https://stackoverflow.com/questions/50411191/how-to-compute-the-cosine-similarity-in-pytorch-for-all-rows-in-a-matrix-with-re
def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


# simple version of transformers' BERT implementation
# https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py#L1633
'''
@add_start_docstrings(
    """
    Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BERT_START_DOCSTRING,
)
'''


class TransformersBertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super().__init__(config)
        self.num_labels = num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class TransformersBertForELClassification(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.config = config
        self.args = args
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.num_entity_labels = args.num_entity_labels
        self.dim_entity_emb = args.dim_entity_emb
        self.entity_classifier = nn.Linear(config.hidden_size, self.dim_entity_emb)

        self.init_weights()

        # **YD** TODO args.EntityEmbedding to be added.
        if args.ent_emb_no_freeze:
            self.entity_emb = nn.Embedding.from_pretrained(args.EntityEmbedding, freeze=False)
        else:
            self.entity_emb = nn.Embedding.from_pretrained(args.EntityEmbedding, freeze=True)

        assert len(self.entity_emb.weight.shape) == 2
        assert self.entity_emb.weight.shape[0] == self.num_entity_labels
        assert self.entity_emb.weight.shape[1] == self.dim_entity_emb

        self.activate = torch.tanh

    '''
    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, entity_labels=None, checkpoint_activations=False):

        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output
    '''

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        entity_labels=None,

        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,

        **kwargs,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)

        # **YD** entity branch forward.
        entity_logits = self.entity_classifier(sequence_output)
        # **YD** may not require activation function
        entity_logits = self.activate(entity_logits)

        # entity_logits = F.normalize(entity_logits, 2, 2)
        # entity_logits = torch.matmul(entity_logits, self.entity_emb.weight.T)
        # entity_logits = torch.log(entity_logits)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            entity_loss_fct = nn.CosineEmbeddingLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                '''
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
                '''
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                ner_loss = loss_fct(active_logits, active_labels)

                '''
                entity_labels[entity_labels == _OUT_DICT_ENTITY_ID] = _IGNORE_CLASSIFICATION_LABEL
                assert entity_labels.requires_grad is False
                entity_active_logits = entity_logits.view(-1, self.num_entity_labels)
                entity_active_labels = torch.where(
                    active_loss, entity_labels.view(-1),
                    torch.tensor(entity_loss_fct.ignore_index).type_as(entity_labels)
                )
                entity_loss = entity_loss_fct(entity_active_logits, entity_active_labels)
                '''

                # entity_active_loss = (labels.view(-1) == NER_LABEL_DICT['B']) | active_loss
                entity_active_loss = (entity_labels.view(-1) > 0)
                entity_active_logits = entity_logits.view(-1, self.dim_entity_emb)[entity_active_loss]
                entity_active_labels = entity_labels.view(-1)[entity_active_loss]

                entity_loss = entity_loss_fct(
                    entity_active_logits,
                    self.entity_emb.weight[entity_active_labels],
                    torch.tensor(1).type_as(entity_labels)
                )

                print('ner_loss', ner_loss, 'entity_loss', entity_loss)
                if torch.isnan(entity_loss):
                    loss = ner_loss
                else:
                    loss = ner_loss + entity_loss
                assert not torch.isnan(loss)
            else:
                # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                raise ValueError("mask has to not None ")

            return loss
        else:
            # **YD** you want to obtain the cosine similarity scores between tokens' hidden embeddings with the
            # entity embeddings within the given entity dictionary.

            # before, entity_logts.shape = [batch_size(=8 by default), num_tokens, entity_embed_length(=300 by deep-ed)]
            # print(entity_logits.shape[0], entity_logits.shape[1], entity_logits.shape[2])
            # assert entity_logits.shape[0] == self.args.batch_size
            assert entity_logits.shape[2] == self.dim_entity_emb

            """
            # **YD** two for loops are slow, may improve to vectorization
            re_logits = torch.zeros(entity_logits.shape[0], entity_logits.shape[1], self.num_entity_labels)
            for i in range(entity_logits.shape[0]):
                for j in range(entity_logits.shape[1]):
                    re_logits[i][j] = F.cosine_similarity(entity_logits[i][j].unsqueeze(0), self.entity_emb.weight)
            """

            re_logits = sim_matrix(entity_logits.view(-1, self.dim_entity_emb), self.entity_emb.weight)
            entity_logits = re_logits.view(entity_logits.shape[0], entity_logits.shape[1], self.num_entity_labels)

            return logits, entity_logits