import torch.nn as nn

class SignToGlossConv3DModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SignToGlossConv3DModel, self).__init__()


class BasicConv3D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=(1,1,1), padding=(0,0,0)):
        super(SignToGlossConv3DModel, self).__init__()
        
        self.conv = nn.Conv3d(in_dim, out_dim, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm3d(out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        return x

