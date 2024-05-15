import torch

# Set the class for Sinc activation
class Sinc(nn.Module):
    def __init__(self, beta):
        super(Sinc, self).__init__()
        self.beta = beta

    def forward(self, x):
        return torch.sinc(self.beta*x)
    