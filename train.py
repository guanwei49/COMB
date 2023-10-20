import numpy as np
import torch.optim as optim
from tqdm import tqdm
import itertools

from conf import *
from models.decoder import Decoder
from models.encoder import Encoder


def train(dataloader,attribute_dims, max_len, b1=0.5 ,b2=0.999):
    '''
    :param dataloader:
    :param attribute_dims:  Number of attribute values per attribute : list
    :param max_len:  max length of traces
    :param b1: adam: decay of first order momentum of gradient
    :param b2: adam: decay of first order momentum of gradient
    :return:
    '''

    encoder = Encoder(attribute_dims, max_len, d_model, ffn_hidden, n_heads, n_layers, n_layers_agg, drop_prob, device)
    decoder = Decoder(attribute_dims, max_len, d_model, ffn_hidden, n_heads, n_layers, drop_prob, device)

    encoder.to(device)
    decoder.to(device)

    optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(),decoder.parameters()),lr=lr, betas=(b1, b2))

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(n_epochs/2), gamma=0.1)

    print("*"*10+"training"+"*"*10)
    for epoch in range(int(n_epochs)):
        train_loss = 0.0
        train_num = 0
        for i, Xs in enumerate(tqdm(dataloader)):
            mask = Xs[-1]
            Xs = Xs[:-1]
            mask = mask.to(device)
            for k ,X in enumerate(Xs):
                Xs[k] = X.to(device)

            optimizer.zero_grad()

            enc_output = encoder(Xs, mask)
            fake_X = decoder(Xs, enc_output,mask)

            loss = 0.0
            for ij in range(len(attribute_dims)):
                # --------------
                # 除了每一个属性的起始字符之外,其他重建误差
                # ---------------
                pred = torch.softmax(fake_X[ij][:, :-1, :], dim=2).flatten(0, -2) #最后一个预测无意义
                true = Xs[ij][:, 1:].flatten()

                corr_pred = pred.gather(1, true.view(-1, 1)).flatten().to(device).reshape(-1,
                                                                                          fake_X[0].shape[1] - 1)

                cross_entropys = -torch.log(corr_pred)
                loss += cross_entropys.masked_select((mask[:, 1:])).mean()

            train_loss += loss.item() * Xs[0].shape[0]
            train_num +=Xs[0].shape[0]
            loss.backward()
            optimizer.step()
        ## 计算一个epoch在训练集上的损失和精度
        train_loss_epoch=train_loss / train_num
        print(f"[Epoch {epoch+1:{len(str(n_epochs))}}/{n_epochs}] "
              f"[loss: {train_loss_epoch:3f}]")
        scheduler.step()

    return encoder,decoder

