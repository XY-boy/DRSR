import torch
import torch.nn as nn
import torch.distributed as dist
from .gather import GatherLayer

class NT_Xent(nn.Module):
    """NT_Xent loss for simclr."""

    def __init__(self, batch_size=32, temperature=0.5, device="GPU"):
        super(NT_Xent, self).__init__()
        # self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        # self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        """Mask correlated samples.
        :param batch_size: batch size of the dataset
        :type batch_size: int
        """
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """Calculate the compare loss."""
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)
        self.batch_size = z_i.shape[0]
        self.mask = self.mask_correlated_samples(self.batch_size)
        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
# class NT_Xent(nn.Module):
#     def __init__(self, batch_size, temperature=0.5, world_size=1):
#         super(NT_Xent, self).__init__()
#         self.batch_size = batch_size
#         self.temperature = temperature
#         self.world_size = world_size
#
#         self.mask = self.mask_correlated_samples(batch_size, world_size)
#         self.criterion = nn.CrossEntropyLoss(reduction="sum")
#         self.similarity_f = nn.CosineSimilarity(dim=2)
#
#     def mask_correlated_samples(self, batch_size, world_size):
#         N = 2 * batch_size * world_size
#         mask = torch.ones((N, N), dtype=bool)
#         mask = mask.fill_diagonal_(0)
#         for i in range(batch_size * world_size):
#             mask[i, batch_size * world_size + i] = 0
#             mask[batch_size * world_size + i, i] = 0
#         return mask
#
#     def forward(self, z_i, z_j):
#         """
#         We do not sample negative examples explicitly.
#         Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples
#         within a minibatch as negative examples.
#         """
#         N = 2 * self.batch_size * self.world_size
#
#         z = torch.cat((z_i, z_j), dim=0)
#         if self.world_size > 1:
#             z = torch.cat(GatherLayer.apply(z), dim=0)
#
#         sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
#
#         sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
#         sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)
#
#         # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
#         positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
#         negative_samples = sim[self.mask].reshape(N, -1)
#
#         print(positive_samples.size())
#         labels = torch.zeros(N).to(positive_samples.device).long()
#         logits = torch.cat((positive_samples, negative_samples), dim=1)
#         loss = self.criterion(logits, labels)
#         loss /= N
#         return loss