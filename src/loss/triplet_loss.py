import numpy as np
import torch


def triplet_loss(alpha=0.2):
    def _triplet_loss(y_pred, Batch_size):
        anchor, positive, negative = \
            y_pred[:int(Batch_size)], y_pred[int(Batch_size):int(2 * Batch_size)], y_pred[int(2 * Batch_size):]

        pos_dist = torch.sqrt(torch.sum(torch.pow(anchor - positive, 2), axis=-1))
        neg_dist = torch.sqrt(torch.sum(torch.pow(anchor - negative, 2), axis=-1))

        keep_all = (neg_dist - pos_dist < alpha).cpu().numpy().flatten()  # 大于这个alpha说明距离已经很远了，没必要管
        hard_triplets = np.where(keep_all == 1)  # 其实是keep_all == True

        pos_dist = pos_dist[hard_triplets]
        neg_dist = neg_dist[hard_triplets]

        basic_loss = pos_dist - neg_dist + alpha
        loss = torch.sum(basic_loss) / torch.max(torch.tensor(1), torch.tensor(len(hard_triplets[0])))  # 向量元素求和变成标量
        return loss

    return _triplet_loss
