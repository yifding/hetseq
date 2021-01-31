import torch
import torch.nn as nn
import torch.nn.functional as F


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


if __name__ == '__main__':
    batch_size = 8
    num_tokens = 20
    dim_emb = 100
    num_ent = 10
    hidden_state = torch.rand(batch_size, num_tokens, dim_emb)
    ent_embedding = nn.Embedding(num_ent, dim_emb)

    re_logits = sim_matrix(hidden_state.view(-1, dim_emb), ent_embedding.weight)
    entity_logits = re_logits.view(batch_size, num_tokens, num_ent)
    print(entity_logits.shape)
    print(batch_size, num_tokens, num_ent)

