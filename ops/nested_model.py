"""
Optional: Data Parallelism
==========================
**Authors**: `Sung Kim <https://github.com/hunkim>`_ and `Jenny Kang <https://github.com/jennykang>`_

In this tutorial, we will learn how to use multiple GPUs using ``DataParallel``.

It's very easy to use GPUs with PyTorch. You can put the model on a GPU:

.. code:: python

    device = torch.device("cuda:0")
    model.to(device)

Then, you can copy all your tensors to the GPU:

.. code:: python

    mytensor = my_tensor.to(device)

Please note that just calling ``my_tensor.to(device)`` returns a new copy of
``my_tensor`` on GPU instead of rewriting ``my_tensor``. You need to assign it to
a new tensor and use that tensor on the GPU.

It's natural to execute your forward, backward propagations on multiple GPUs.
However, Pytorch will only use one GPU by default. You can easily run your
operations on multiple GPUs by making your model run parallelly using
``DataParallel``:

.. code:: python

    model = nn.DataParallel(model)

That's the core behind this tutorial. We will explore it in more detail below.
"""


######################################################################
# Imports and parameters
# ----------------------
#
# Import PyTorch modules and define parameters.
#

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pdb
# Parameters and DataLoaders
ch_dim = 128

input_size = (16,14,14,ch_dim)
output_size = 2

batch_size = 30
data_size = 100


######################################################################
# Device
#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######################################################################
# Dummy DataSet
# -------------
#
# Make a dummy (random) dataset. You just need to implement the
# getitem
#

class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size[0], size[1], size[2],size[3])

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)


######################################################################
# Simple Model
# ------------
#
# For the demo, our model just gets an input, performs a linear operation, and
# gives an output. However, you can use ``DataParallel`` on any model (CNN, RNN,
# Capsule Net etc.)
#
# We've placed a print statement inside the model to monitor the size of input
# and output tensors.
# Please pay attention to what is printed at batch rank 0.
#


class LayerNormAffine(nn.Module):
    def __init__(self,input, size=128):
        super(LayerNormAffine, self).__init__()
        self.norm = nn.LayerNorm(128,elementwise_affine=False).to(input.device)
        # nn.init.constant_(self.norm.weight, 0)
        # nn.init.constant_(self.norm.bias, 0)

        self.scale = torch.nn.Parameter(torch.Tensor(size=[1,1,1,size])).to(input.device)
        self.bias = torch.nn.Parameter(torch.Tensor(size=[1,1,1,size])).to(input.device)
        
        # pdb.set_trace()
        # self.norm = self.norm.to(input.device)
        # self.scale = scale.to(input.device)
        # self.bias = bias.to(input.device)
        
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.scale, 1)

        # self.register_parameter('bias', self.bias)
        # self.register_parameter('scale', self.scale)

    def forward(self, input):
        # input = self.norm(input) * self.params['scale'] + self.params['bias']
        input = self.norm(input) * self.scale + self.bias

        
        return input


class Model2(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model2, self).__init__()
        self.fc = nn.Linear(ch_dim, ch_dim)
        self.norm_dict = nn.ModuleDict({})
        self.module_norm_dict = nn.ModuleList()

        self.model2_bias = torch.nn.Parameter(torch.Tensor(size=[1,1,1,128]))
        nn.init.constant_(self.model2_bias, 1)
        
        # self.init_norm = LayerNormAffine()


    def get_norm(self, input, name, zero_init=False):
        # input: B * T x C x H x W

        if name not in self.norm_dict:
            self.model2_asdasdiasdadksa = torch.nn.Parameter(torch.Tensor(size=[1,1,1,128]))
            nn.init.constant_(self.model2_asdasdiasdadksa, 1)


            norm = LayerNormAffine(input)
            self.norm_dict[name] = norm
            # print(self)
        else:
            norm = self.norm_dict[name]

        input = self.norm_dict[name](input)
        return input

    def forward(self, input):
        pdb.set_trace()
        input = self.get_norm(input, 'norm1') 
        input = input +  self.model2_bias

        output = self.fc(input)
        print("\tIn Model2: input size", input.size(),
              "output size", output.size())

        return output


class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.model2 = Model2(input_size, ch_dim)
        self.fc = nn.Linear(ch_dim, output_size)

    def forward(self, input):
        input = self.model2(input)
        output = self.fc(input)
        print(f"Forward model: {self}")
        # print(f'Model parameters: {self.parameters()}')
        #for p in self.parameters():
        for name, param in model.named_parameters():
            print(f'param: {name}')
            print(f'param: {param.device}')

        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output


######################################################################
# Create Model and DataParallel
# -----------------------------
#
# This is the core part of the tutorial. First, we need to make a model instance
# and check if we have multiple GPUs. If we have multiple GPUs, we can wrap
# our model using ``nn.DataParallel``. Then we can put our model on GPUs by
# ``model.to(device)``
#

model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  print(f"Model: {model}")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)


######################################################################
# Run the Model
# -------------
#
# Now we can see the sizes of input and output tensors.
#

for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())


######################################################################
# Results
# -------
#
# If you have no GPU or one GPU, when we batch 30 inputs and 30 outputs, the model gets 30 and outputs 30 as
# expected. But if you have multiple GPUs, then you can get results like this.
#
# 2 GPUs
# ~~~~~~
#
# If you have 2, you will see:
#
# .. code:: bash
#
#     # on 2 GPUs
#     Let's use 2 GPUs!
#         In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#         In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#         In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#         In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
#         In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
#     Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
#
# 3 GPUs
# ~~~~~~
#
# If you have 3 GPUs, you will see:
#
# .. code:: bash
#
#     Let's use 3 GPUs!
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#     Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
#
# 8 GPUs
# ~~~~~~~~~~~~~~
#
# If you have 8, you will see:
#
# .. code:: bash
#
#     Let's use 8 GPUs!
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#     Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
#


######################################################################
# Summary
# -------
#
# DataParallel splits your data automatically and sends job orders to multiple
# models on several GPUs. After each model finishes their job, DataParallel
# collects and merges the results before returning it to you.
#
# For more information, please check out
# https://pytorch.org/tutorials/beginner/former\_torchies/parallelism\_tutorial.html.
#