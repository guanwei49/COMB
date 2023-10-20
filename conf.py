import torch
# device = torch.device("cpu")
# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# model parameter setting
batch_size = 64
d_model = 64
n_layers_agg= 2
n_layers = 2
n_heads = 4
ffn_hidden = 128
drop_prob = 0.1

n_epochs=20
lr=0.0002 #learning rate