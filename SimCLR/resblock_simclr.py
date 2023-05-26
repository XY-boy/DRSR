import torch.nn as nn
import torchvision


class Resb_SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResBlock used in DASR to obtain hi = f(x ̃i) = ResBlock(x ̃i)
    where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder):
        super(Resb_SimCLR, self).__init__()

        self.encoder = encoder
        # self.encoder_j = encoder

    def forward(self, x_i, x_j):
        h_i, mlp_i = self.encoder(x_i)
        h_j, mlp_j = self.encoder(x_j)

        return h_i, h_j, mlp_i, mlp_j