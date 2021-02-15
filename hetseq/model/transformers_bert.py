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
MAX_LEN_MENTION = 4

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

        # **YD** support weight for entity linking loss.
        self.entity_loss_weight = args.entity_loss_weight if hasattr(args, 'entity_loss_weight') else 1

        self.num_entity_labels = args.num_entity_labels
        self.dim_entity_emb = args.dim_entity_emb
        self.entity_classifier = nn.Linear(config.hidden_size, self.dim_entity_emb)

        self.init_weights()

        # **YD** TODO args.EntityEmbedding to be added.
        if hasattr(args, "ent_emb_no_freeze") and args.ent_emb_no_freeze:
            self.entity_emb = nn.Embedding.from_pretrained(args.EntityEmbedding, freeze=False)
        else:
            self.entity_emb = nn.Embedding.from_pretrained(args.EntityEmbedding, freeze=True)

        assert len(self.entity_emb.weight.shape) == 2
        assert self.entity_emb.weight.shape[0] == self.num_entity_labels
        assert self.entity_emb.weight.shape[1] == self.dim_entity_emb

        self.activate = torch.tanh

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
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                ner_loss = loss_fct(active_logits, active_labels)

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
                    loss = ner_loss + self.entity_loss_weight * entity_loss
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

            re_logits = sim_matrix(entity_logits.view(-1, self.dim_entity_emb), self.entity_emb.weight)
            entity_logits = re_logits.view(entity_logits.shape[0], entity_logits.shape[1], self.num_entity_labels)

            return logits, entity_logits


class TransformersBertForELClassificationCrossEntropy(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.config = config
        self.args = args
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        # **YD** support weight for entity linking loss.
        self.entity_loss_weight = args.entity_loss_weight if hasattr(args, 'entity_loss_weight') else 1

        self.num_entity_labels = args.num_entity_labels
        self.dim_entity_emb = args.dim_entity_emb
        self.entity_classifier = nn.Linear(config.hidden_size, self.dim_entity_emb)

        self.init_weights()

        # **YD** TODO args.EntityEmbedding to be added.
        if hasattr(args, "ent_emb_no_freeze") and args.ent_emb_no_freeze:
            self.entity_emb = nn.Embedding.from_pretrained(args.EntityEmbedding, freeze=False)
        else:
            self.entity_emb = nn.Embedding.from_pretrained(args.EntityEmbedding, freeze=True)

        assert len(self.entity_emb.weight.shape) == 2
        assert self.entity_emb.weight.shape[0] == self.num_entity_labels
        assert self.entity_emb.weight.shape[1] == self.dim_entity_emb

        self.activate = torch.tanh
        self.loss_fct = nn.CrossEntropyLoss()
        if hasattr(args, 'entity_loss_type'):
            if args.entity_loss_type == 'CrossEntropyLoss':
                self.entity_loss_fct = nn.CrossEntropyLoss()
            else:
                raise ValueError('Unknown entity loss function')
        else:
            self.entity_loss_fct = nn.CrossEntropyLoss()

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
        entity_logits = sim_matrix(
            entity_logits.view(-1, self.dim_entity_emb),
            self.entity_emb.weight,
        ).view(entity_logits.shape[0], entity_logits.shape[1], self.num_entity_labels)

        if labels is not None:

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(self.loss_fct.ignore_index).type_as(labels)
                )
                ner_loss = self.loss_fct(active_logits, active_labels)

                # entity_active_loss = (labels.view(-1) == NER_LABEL_DICT['B']) | active_loss
                entity_active_loss = (entity_labels.view(-1) > 0)
                entity_active_logits = entity_logits.view(-1, self.num_entity_labels)[entity_active_loss]
                entity_active_labels = entity_labels.view(-1)[entity_active_loss]

                entity_loss = self.entity_loss_fct(
                    entity_active_logits,
                    entity_active_labels,
                )

                print('ner_loss', ner_loss, 'entity_loss', entity_loss)
                # there may be no entity in a sentence in the training process.
                if torch.isnan(entity_loss):
                    loss = ner_loss
                else:
                    loss = ner_loss + self.entity_loss_weight * entity_loss
                assert not torch.isnan(loss)
            else:
                raise ValueError("mask has to not None ")

            return loss
        else:
            return logits, entity_logits


class TransformersBertForNERSymmetry(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.config = config
        self.args = args
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.left_classifier = nn.Linear(config.hidden_size, 2)
        self.right_classifier = nn.Linear(config.hidden_size, 2)

        self.init_weights()

        self.left_loss_fct = nn.CrossEntropyLoss()
        self.right_loss_fct = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,

        entity_th_ids=None,
        left_mention_masks=None,
        right_mention_masks=None,
        left_entity_masks=None,
        right_entity_masks=None,

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

        left_logits = self.left_classifier(sequence_output)
        right_logits = self.right_classifier(sequence_output)

        if entity_th_ids is not None:
            assert attention_mask is not None

            active_left_loss = left_mention_masks.view(-1) == 1
            active_right_loss = right_mention_masks.view(-1) == 1

            active_left_logits = left_logits.view(-1, 2)
            active_right_logits = right_logits.view(-1, 2)

            '''
            active_left_labels = left_entity_masks.view(-1)[active_left_loss]
            active_right_labels = right_entity_masks.view(-1)[active_right_loss]
            '''
            active_left_labels = torch.where(
                active_left_loss, left_entity_masks.view(-1), torch.tensor(-100).type_as(left_entity_masks)
            )
            active_right_labels = torch.where(
                active_right_loss, right_entity_masks.view(-1), torch.tensor(-100).type_as(right_entity_masks)
            )

            left_loss = self.left_loss_fct(active_left_logits, active_left_labels)
            right_loss = self.right_loss_fct(active_right_logits, active_right_labels)

            loss = left_loss + right_loss

            return loss
        else:
            return left_logits, right_logits


class TransformersBertForELSymmetry(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.config = config
        self.args = args
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.left_classifier = nn.Linear(config.hidden_size, 2)
        self.right_classifier = nn.Linear(config.hidden_size, 2)

        # **YD** entity related parameters
        self.entity_loss_weight = args.entity_loss_weight if hasattr(args, 'entity_loss_weight') else 1
        self.num_entity_labels = args.num_entity_labels
        self.dim_entity_emb = args.dim_entity_emb

        self.entity_classifier = nn.Linear(2 * config.hidden_size, self.dim_entity_emb)

        self.init_weights()

        self.left_loss_fct = nn.CrossEntropyLoss()
        self.right_loss_fct = nn.CrossEntropyLoss()

        # **YD** args.EntityEmbedding to be added.
        if hasattr(args, "ent_emb_no_freeze") and args.ent_emb_no_freeze:
            self.entity_emb = nn.Embedding.from_pretrained(args.EntityEmbedding, freeze=False)
        else:
            self.entity_emb = nn.Embedding.from_pretrained(args.EntityEmbedding, freeze=True)

        assert len(self.entity_emb.weight.shape) == 2
        assert self.entity_emb.weight.shape[0] == self.num_entity_labels
        assert self.entity_emb.weight.shape[1] == self.dim_entity_emb

        self.activate = torch.tanh

        self.entity_loss_fct = nn.CosineEmbeddingLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,

        entity_th_ids=None,

        # **YD** left_mention_masks and right_mention_masks are deprecated
        # left_mention_masks=None,
        # right_mention_masks=None,

        left_entity_masks=None,
        right_entity_masks=None,

        span_masks=None,
        span_cand_entities=None,

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

        left_logits = self.left_classifier(sequence_output)
        right_logits = self.right_classifier(sequence_output)

        # **YD** compute ED logits
        el_sequence_output = sequence_output.view(-1, self.config.hidden_size)

        batch_size = left_logits.shape[0]
        num_token = left_logits.shape[1]
        total_token = batch_size * num_token

        # **YD** max_len_mention has been implemented
        max_len_mention = span_masks.shape[2]
        ent_left_side = torch.repeat_interleave(torch.arange(total_token), max_len_mention)
        ent_right_side = torch.arange(total_token - 1 + max_len_mention).unfold(0, max_len_mention, 1).reshape(-1)
        ent_right_side[ent_right_side >= total_token] = 0

        entity_output = torch.cat((el_sequence_output[ent_left_side], el_sequence_output[ent_right_side]), axis=1)
        entity_output = self.entity_classifier(entity_output)
        entity_output = self.activate(entity_output)

        if entity_th_ids is not None:
            assert attention_mask is not None

            # **YD** compute NER loss
            active_left_logits = left_logits.view(-1, 2)
            active_right_logits = right_logits.view(-1, 2)

            active_left_labels = left_entity_masks.view(-1)
            active_right_labels = right_entity_masks.view(-1)

            left_loss = self.left_loss_fct(active_left_logits, active_left_labels)
            right_loss = self.right_loss_fct(active_right_logits, active_right_labels)

            ner_loss = left_loss + right_loss

            # **YD** compute ED loss
            # **YD** max_len_mention has not implemented
            entity_output = entity_output.view(total_token, max_len_mention, 2 * dim_emb)

            ent_left_index = left_entity_masks.view(-1).nonzero(as_tuple=True)[0]
            ent_right_index = right_entity_masks.view(-1).nonzero(as_tuple=True)[0]
            ent_true_right_index = ent_right_index - ent_left_index

            entity_output = entity_output[ent_left_index, ent_true_right_index]

            # **YD** candidate entities can be utilized to compute hinge loss, as features to predict NER and etc.
            # cand_entity_output = span_cand_entities[ent_left_index, ent_true_right_index]

            entity_active_loss = (active_left_labels > 0)
            entity_active_labels = entity_th_ids.view(-1)[entity_active_loss]

            entity_loss = self.entity_loss_fct(
                entity_output,
                self.entity_emb.weight[entity_active_labels],
                torch.tensor(1).type_as(entity_output)
            )

            if torch.isnan(entity_loss):
                loss = ner_loss
            else:
                loss = ner_loss + self.entity_loss_weight * entity_loss

            return loss

        else:
            # **YD** entity_output.shape = [batch_size, len_tokens, max_len_mentions, dim_entity_emb]
            assert entity_output.shape[3] == self.dim_entity_emb
            re_logits = sim_matrix(entity_output.view(-1, self.dim_entity_emb), self.entity_emb.weight)
            entity_logits = re_logits.view(
                entity_output.shape[0],
                entity_output.shape[1],
                entity_output.shape[2],
                self.num_entity_labels,
            )

            return left_logits, right_logits, entity_logits