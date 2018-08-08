import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import word2vec

def get_emb(emb_path):
    model = word2vec.load(emb_path)
    vec = torch.from_numpy(model.vectors)
    vec = F.normalize(vec)
    return vec, model.vocab

def batch_dot(tensor_a, tensor_b):
    dot_vec = torch.bmm(
        tensor_a.view(-1, 1, tensor_a.size(1)),
        tensor_b.view(-1, tensor_b.size(1), 1)
    )
    dot_vec = dot_vec.view(-1)
    return dot_vec

output_model_vec, vocab = get_emb("./random_start_output_emb.txt")
#  pretrained_model_vec, _ = get_emb("./text8.mik.style.vec.txt")
input_model_vec, _ = get_emb("./random_start_input_emb.txt")

#  cos_out_pretrain = batch_dot(output_model_vec, pretrained_model_vec)
#  cos_input_pretrain = batch_dot(input_model_vec, pretrained_model_vec)
cos_out_input = batch_dot(output_model_vec, input_model_vec)
print(cos_out_input.mean())
