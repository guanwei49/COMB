import torch
from torch import nn

from models.decoder_layer import DecoderLayer, DecoderLayer_attr
from models.transformer_embedding import TransformerEmbedding


class Decoder_single_act(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, act, enc_src, trg_mask, src_mask):
        act = self.emb(act)
        act_embed = act

        for layer in self.layers:
            act = layer(act, enc_src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(act)
        return output, act_embed

class Decoder_single_attr(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.device=device
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)
        DecoderLayers=[DecoderLayer_attr(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)]
        for _ in range(n_layers - 1):
            DecoderLayers.append(DecoderLayer(d_model=d_model,
                                                      ffn_hidden=ffn_hidden,
                                                      n_head=n_head,
                                                      drop_prob=drop_prob))

        self.layers = nn.ModuleList(DecoderLayers)

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, current_act, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)
        trg_first =torch.cat((trg, current_act),dim=1)  #将当前属性信息，与控制流信息连接在一起
        eye = torch.eye( trg_mask.shape[-2], trg_mask.shape[-1], device=self.device).unsqueeze(0).unsqueeze(0)
        trg_mask_first = torch.cat((trg_mask,eye.repeat( trg_mask.shape[0], trg_mask.shape[1],1,1)),dim=-1)  ##考虑当前活动信息以及这个属性前面出现的所有属性值
        for i,layer in enumerate(self.layers):
            if i==0:
                trg = layer(current_act,trg_first, enc_src, trg_mask_first, src_mask)
            else:
                trg = layer(trg, enc_src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output

class Decoder(nn.Module):
    def __init__(self, dec_voc_sizes, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        decoders=[]
        self.device=device
        self.attribute_dims = dec_voc_sizes
        for i, dim in enumerate(dec_voc_sizes):
            if i == 0:
                decoders.append(Decoder_single_act(int(dim + 1), max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device))
            else:
                decoders.append(
                    Decoder_single_attr(int(dim + 1), max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device))

        self.decoders = nn.ModuleList(decoders)

    def forward(self, Xs, enc_output,mask):
        '''
            :param Xs:是多个属性，每一个属性作为一个X ：[batch_size, seq_len]
            :param enc_output:是多个属性的encoder输出 : [batch_size, seq_len*num_attr,d_model]
            :param mask: [batch_size,seq_len]
            :return:
        '''
        dec_output = []
        act_embed = None
        for i, dim in enumerate(self.attribute_dims):
            X = Xs[i]
            src_mask = mask.unsqueeze(1).unsqueeze(1).repeat((1, 1, 1, len(self.attribute_dims)))
            trg_pad_mask = mask.unsqueeze(1).unsqueeze(3)
            trg_sub_mask = torch.tril(torch.ones(mask.shape[-1], mask.shape[-1])).type(torch.ByteTensor).to(
                self.device)
            trg_mask = trg_sub_mask & trg_pad_mask
            if i == 0:
                dec_output_, act_embed = self.decoders[i](X,enc_output,trg_mask,src_mask)
                dec_output.append(dec_output_)
            else:
                pad = torch.zeros(act_embed.shape[0],1, act_embed.shape[-1],device=self.device)
                act_embed = torch.cat((act_embed[:,1:,:],pad),dim=1)  #第一个活动不要，向后移动一位： 不重建起始字符
                dec_output_ = self.decoders[i](act_embed,X,enc_output,trg_mask,src_mask)
                dec_output.append(dec_output_)

        return dec_output
