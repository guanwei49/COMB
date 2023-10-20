import torch
from torch import nn

from models.encoder_layer import EncoderLayer, EncoderLayer_agg
from models.transformer_embedding import TransformerEmbedding


class Encoder_single(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)    #不跨属性的attention
        return x

class Encoder_single_agg(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer_agg(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self,  x, all_x, src_mask):
        for layer in self.layers:
            x = layer(x,all_x, src_mask)  # 跨属性的attention ：自己属性值的信息，查询所有属性的属性值的信息
        return x

class Encoder(nn.Module):
    def __init__(self, enc_voc_sizes, max_len, d_model, ffn_hidden, n_head, n_layers,n_layers_agg, drop_prob, device):
        super().__init__()
        first_encoders=[]
        self.attribute_dims = enc_voc_sizes
        for i, dim in enumerate(enc_voc_sizes):
            first_encoders.append(Encoder_single(int(dim + 1), max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device))
        aggregate_encoders=[]

        for i, dim in enumerate(enc_voc_sizes):
            aggregate_encoders.append(
                Encoder_single_agg(d_model, ffn_hidden, n_head, n_layers_agg, drop_prob))

        self.first_encoders = nn.ModuleList(first_encoders)
        self.aggregate_encoders = nn.ModuleList(aggregate_encoders)

    def forward(self, Xs, src_mask):
        '''
            :param Xs:是多个属性，每一个属性作为一个X ：[batch_size, time_step]
            :return:
        '''
        enc_output = []
        agg_enc_output=[]
        src_mask = src_mask.unsqueeze(1).unsqueeze(1)
        for i, dim in enumerate(self.attribute_dims):
            X = Xs[i]
            enc_output_ = self.first_encoders[i](X,src_mask)
            enc_output.append(enc_output_)
        all_x= torch.cat(enc_output, dim=1)
        agg_mask = src_mask.repeat((1,1, 1, len(self.attribute_dims)))
        for i, dim in enumerate(self.attribute_dims):
            agg_enc_output_ = self.aggregate_encoders[i](enc_output[i],all_x, agg_mask)  #[batch_size,seq_len,d_model] 聚合不同属性之间的信息
            agg_enc_output.append(agg_enc_output_)
        agg_enc_output = torch.cat(agg_enc_output, dim=1)

        return agg_enc_output  #隐藏层
