import numpy as np
import torch
from tqdm import tqdm

from conf import device


def detect(encoder,decoder, dataloader, attribute_dims, attr_Shape):
    encoder.eval()
    decoder.eval()
    pos=0
    with torch.no_grad():
        attr_level_abnormal_scores=np.zeros(attr_Shape)

        print("*" * 10 + "detecting" + "*" * 10)

        for Xs in tqdm(dataloader):
            mask = Xs[-1]
            Xs = Xs[:-1]
            for k,tempX in enumerate(Xs):
                Xs[k] = tempX.to(device)
            mask=mask.to(device)

            enc_output = encoder(Xs, mask)
            fake_X = decoder(Xs, enc_output, mask)

            for attr_index in range(len(attribute_dims)):
                fake_X[attr_index]=torch.softmax(fake_X[attr_index][:, :-1, :],dim=2)

        #求异常分数
            for attr_index in range(len(attribute_dims)):
                truepos = Xs[attr_index][:, 1:].flatten()
                p = fake_X[attr_index].reshape((truepos.shape[0],-1)).gather(1, truepos.view(-1, 1)).squeeze()
                p_distribution = fake_X[attr_index].reshape((truepos.shape[0],-1))

                p_distribution = p_distribution + 1e-8  # 避免出现概率为0

                attr_level_abnormal_scores[pos: pos + Xs[attr_index].shape[0], 1: ,attr_index] = \
                    ((torch.sum(torch.log(p_distribution) * p_distribution, 1) - torch.log(p)).reshape((Xs[attr_index].shape[0],-1))*(mask[:,1:])).detach().cpu()
            pos += Xs[attr_index].shape[0]


        trace_level_abnormal_scores = attr_level_abnormal_scores.max((1, 2))
        event_level_abnormal_scores = attr_level_abnormal_scores.max((2))
        return  trace_level_abnormal_scores,event_level_abnormal_scores,attr_level_abnormal_scores
