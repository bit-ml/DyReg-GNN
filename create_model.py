import torch
from torch.nn import functional as F
from ops.dyreg import DynamicGraph, dyregParams

class SpaceTimeModel(torch.nn.Module):
    def __init__(self):
        super(SpaceTimeModel, self).__init__()
        dyreg_params = dyregParams()
        dyregParams.offset_lstm_dim = 32
        self.dyreg = DynamicGraph(dyreg_params,
                    backbone_dim=32, node_dim=32, out_num_ch=32,
                    H=16, W=16, 
                    iH=16, iW=16,
                    project_i3d=False,
                    name='lalalal')


        self.fc = torch.nn.Linear(32, 10)

    def forward(self, x):
        dx = self.dyreg(x)
        # you can initialize the dyreg branch as identity function by normalisation, 
        #   as done in DynamicGraphWrapper found in ./ops/dyreg.py 
        x = x + dx
        # average over time and space: T, H, W
        x = x.mean(-1).mean(-1).mean(-2)
        x = self.fc(x)
        return x



class ConvSpaceTimeModel(torch.nn.Module):
    def __init__(self):
        super(ConvSpaceTimeModel, self).__init__()
        dyreg_params = dyregParams()
        dyregParams.offset_lstm_dim = 32
        self.conv1 = torch.nn.Conv3d(in_channels=3, out_channels=32, kernel_size=[1,3,3], stride=[1,2,2],padding=[0,1,1])
        self.conv2 = torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=[1,3,3], stride=[1,2,2],padding=[0,1,1])
        self.dyreg = DynamicGraph(dyreg_params,
                    backbone_dim=32, node_dim=32, out_num_ch=32,
                    H=16, W=16, 
                    iH=16, iW=16,
                    project_i3d=False,
                    name='lalalal')

        self.conv3 = torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=[1,3,3], stride=[1,2,2],padding=[0,1,1])
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        input = x.permute(0,2,1,3,4).contiguous()
        dx = self.dyreg(input)
        dx = dx.permute(0,2,1,3,4).contiguous()
        # you can initialize the dyreg branch as identity function by normalisation, 
        #   as done in DynamicGraphWrapper found in ./ops/dyreg.py 
        x = x + dx
        x = F.relu(self.conv3(x))
        # average over time and space: T, H, W
        x = x.mean(-1).mean(-1).mean(-1)
        x = self.fc(x)
        return x

B = 8
T = 10
C = 32
H = 16
W = 16
x = torch.ones(B,T,C,H,W)
st_model = SpaceTimeModel()
out1 = st_model(x)


B = 8
T = 10
C = 3
H = 64
W = 64
x = torch.ones(B,C,T,H,W)
conv_st_model = ConvSpaceTimeModel()
out2 = conv_st_model(x)
[print(f'{k} {v.shape}') for k,v in conv_st_model.named_parameters()]

out = out1 + out2
print('done')