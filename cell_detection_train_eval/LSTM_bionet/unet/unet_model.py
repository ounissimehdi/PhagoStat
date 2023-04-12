# ------------------------------------------------------------------------------
#
#
#                                 P̶̖̥͈͈͇̼͇̈́̈́̀͋͒̽͊͘͠h̴͙͈̦͗́̓a̴̧͗̾̀̅̇ḡ̶͓̭̝͓͎̰͓̦̭̎́͊̐̂͒͠ơ̵̘͌̿̽̑̈̾Ś̴̠́̓̋̂̃͒͆̚t̴̲͓̬͎̾͑͆͊́̕a̸͍̫͎̗̞͆̇̀̌̑͂t̸̳̼̘͕̯̠̱̠͂̔̒̐̒̕͝͝
#
#
#                                PhagoStat
#                Advanced Phagocytic Activity Analysis Tool
# ------------------------------------------------------------------------------
# Copyright (C) 2023 Mehdi OUNISSI <mehdi.ounissi@icm-institute.org>
#               Sorbonne University, Paris Brain Institute - ICM, CNRS, Inria,
#               Inserm, AP-HP, Paris, 75013, France.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ------------------------------------------------------------------------------
# Note on Imported Packages:
# The packages used in this work are imported as is and not modified. If you
# intend to use, modify, or distribute any of these packages, please refer to
# the requirements.txt file and the respective package licenses.
# ------------------------------------------------------------------------------

import torch.nn.functional as F

from .unet_parts import *

from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ConvTranspose2d, ReLU, MaxPool2d, Sigmoid, Parameter
from torch import tensor, cat
import torch


class BiONet_LSTM_ready(Module):

    def __init__(self,
                 num_classes: int = 1,
                 iterations: int = 1,
                 multiplier: float = 1.0,
                 num_layers: int = 4,
                 integrate: bool = False):

        super(BiONet_LSTM_ready, self).__init__()
        # Parameters
        self.iterations = iterations
        self.multiplier = multiplier
        self.num_layers = num_layers
        self.integrate = integrate
        self.batch_norm_momentum = 0.01
        # Generate channel parameters, where the channel starts from the amount output by the first Encoder until the semantic vector
        self.filters_list = [int(64 * (2 ** i) * self.multiplier) for i in range(self.num_layers + 1)]
        # Preprocess the convolution block, do not participate in the loop, and the final output is 64*256*256
        self.pre_transform_conv_block = Sequential(
            # Looking at the code implementation here, it should always be the same as the number of layers output by the first Encoder
            Conv2d(num_classes, self.filters_list[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            BatchNorm2d(self.filters_list[0], momentum=self.batch_norm_momentum),
            Conv2d(self.filters_list[0], self.filters_list[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # Generate f[1]*1024*1024
            ReLU(),
            BatchNorm2d(self.filters_list[0], momentum=self.batch_norm_momentum),
            Conv2d(self.filters_list[0], self.filters_list[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # Generate f[1]*1024*1024
            ReLU(),
            BatchNorm2d(self.filters_list[0], momentum=self.batch_norm_momentum),
            MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        )
        self.reuse_convs = []  # encoder multiplexed convolution kernel, each encoder corresponds to a tuple (3 convolution kernels in total, excluding ReLU)
        self.encoders = []  # A list of encoders. Since part of the encoder does not participate in the loop, each encoder is a tuple (Sequential, DOWN of two CONVs)
        self.reuse_deconvs = []  # Decoder multiplexed convolution and deconvolution kernels, each decoder corresponds to a tuple (3 convolution kernels in total, excluding ReLU)
        self.decoders = []  # A list of decoders. Since part of the decoder does not participate in the loop, each decoder is a tuple (Sequential, UP of two CONVs)
        for iteration in range(self.iterations):
            for layer in range(self.num_layers):

                # Create the encoders section. Although part of the code can be written together, in order to look clear (and the constructor is not very efficient), so consider the encoder and decoder separately
                # constants related to the level
                in_channel = self.filters_list[layer] * 2 # Since there is incoming data from the output part, it is necessary to double the input channel
                mid_channel = self.filters_list[layer]
                out_channel = self.filters_list[layer + 1]
                # Create encoders model
                if iteration == 0:
                    # Create and add multiplexed convolution kernel
                    # Only the last convolution kernel is responsible for raising the channel
                    conv1 = Conv2d(in_channel, mid_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    conv2 = Conv2d(mid_channel, mid_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    conv3 = Conv2d(mid_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    self.reuse_convs.append((conv1, conv2, conv3))

                # create encoder
                # First construct two CONVs
                convs = Sequential(
                    self.reuse_convs[layer][0],
                    ReLU(),
                    BatchNorm2d(mid_channel, momentum=self.batch_norm_momentum),
                    self.reuse_convs[layer][1],
                    ReLU(),
                    BatchNorm2d(mid_channel, momentum=self.batch_norm_momentum)
                )
                # build DOWN
                down = Sequential(
                    self.reuse_convs[layer][2],
                    ReLU(),
                    BatchNorm2d(out_channel, momentum=self.batch_norm_momentum),
                    MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
                )
                self.add_module("iteration{0}_layer{1}_encoder_convs".format(iteration, layer), convs)
                self.add_module("iteration{0}_layer{1}_encoder_down".format(iteration, layer), down)
                self.encoders.append((convs, down))


                # Create the part of decoders, modeled after encoders
                # Constants related to the level, note that mid_channel is not needed in this part, because the first convolution kernel has already increased the dimension
                in_channel = self.filters_list[self.num_layers - layer] + self.filters_list[self.num_layers - 1 - layer]
                out_channel = self.filters_list[self.num_layers - 1 - layer]
                
                # Create the decoders model
                if iteration == 0:
                    # Create and add multiplexed convolution kernel
                    # Increase the number of channels from the first convolution kernel
                    conv1 = Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    conv2 = Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    conv3 = ConvTranspose2d(out_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2),
                                            output_padding=(1, 1))  # This is a bit different from tensorflow. For the complete shape, you need to use output to make up for it
                    self.reuse_deconvs.append((conv1, conv2, conv3))

                # create encoder
                # First construct two CONVs
                convs = Sequential(
                    self.reuse_deconvs[layer][0],
                    ReLU(),
                    BatchNorm2d(out_channel, momentum=self.batch_norm_momentum),
                    self.reuse_deconvs[layer][1],
                    ReLU(),
                    BatchNorm2d(out_channel, momentum=self.batch_norm_momentum)
                )
                
                # Construct UP
                up = Sequential(
                    self.reuse_deconvs[layer][2],
                    ReLU(),
                    BatchNorm2d(out_channel, momentum=self.batch_norm_momentum)
                )
                self.add_module("iteration{0}_layer{1}_decoder_convs".format(iteration, layer), convs)
                self.add_module("iteration{0}_layer{1}_decoder_up".format(iteration, layer), up)
                self.decoders.append((convs, up))
        # create middle layer
        self.middles = Sequential(
            Conv2d(self.filters_list[-1], self.filters_list[-1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            BatchNorm2d(self.filters_list[-1], momentum=self.batch_norm_momentum),
            Conv2d(self.filters_list[-1], self.filters_list[-1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            BatchNorm2d(self.filters_list[-1], momentum=self.batch_norm_momentum),
            ConvTranspose2d(self.filters_list[-1], self.filters_list[-1], kernel_size=(3, 3), padding=(1, 1),
                            stride=(2, 2), output_padding=(1, 1)),
            ReLU(),
            BatchNorm2d(self.filters_list[-1], momentum=self.batch_norm_momentum)
        )
        self.post_transform_conv_block = Sequential(
            Conv2d(self.filters_list[0] * self.iterations, self.filters_list[0], kernel_size=(3, 3), padding=(1, 1),
                   stride=(1, 1)) if self.integrate else Conv2d(self.filters_list[0],
                                                                self.filters_list[0], kernel_size=(3, 3),
                                                                padding=(1, 1), stride=(1, 1)),
            ReLU(),
            BatchNorm2d(self.filters_list[0], momentum=self.batch_norm_momentum),
            Conv2d(self.filters_list[0], self.filters_list[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ReLU(),
            BatchNorm2d(self.filters_list[0], momentum=self.batch_norm_momentum),
            Conv2d(self.filters_list[0], 1, kernel_size=(1, 1), stride=(1, 1)),
            # Sigmoid(),
        )

    def forward(self, x: tensor) -> tensor:
        enc = [None for i in range(self.num_layers)]
        dec = [None for i in range(self.num_layers)]
        all_output = [None for i in range(self.iterations)]
        x = self.pre_transform_conv_block(x)
        e_i = 0
        d_i = 0
        for iteration in range(self.iterations):
            for layer in range(self.num_layers):
                if layer == 0:
                    x_in = x
                x_in = self.encoders[e_i][0](cat([x_in, x_in if dec[-1 - layer] is None else dec[-1 - layer]], dim=1))
                enc[layer] = x_in
                x_in = self.encoders[e_i][1](x_in)
                e_i = e_i + 1
            x_in = self.middles(x_in)
            for layer in range(self.num_layers):
                x_in = self.decoders[d_i][0](cat([x_in, enc[-1 - layer]], dim=1))
                dec[layer] = x_in
                x_in = self.decoders[d_i][1](x_in)
                d_i = d_i + 1
            all_output[iteration] = x_in
        if self.integrate:
            x_in = cat(all_output, dim=1)
        logits = self.post_transform_conv_block(x_in)
        return x_in, logits

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=False),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# This is the overall framework
### TITLE : LSTM workflow (LSTM_WTL)
class LSTM_MMUnet(nn.Module):
    def __init__(self, unet_model_path, input_nc=1, output_nc=1, ngf=64, temporal=3, UNet_train=False, device_name='cuda'):
        super(LSTM_MMUnet, self).__init__()
        self.temporal   = temporal
        self.UNet_train = UNet_train 
        self.unet = BiONet_LSTM_ready(num_classes= 1, iterations= 1, multiplier= 1.0, num_layers = 4, integrate = False)
        
        
        if unet_model_path != '':
            device = torch.device(device_name)
            self.unet.load_state_dict(torch.load(unet_model_path, map_location= device))#, map_location='cuda'

        #UNet_LSTM_ready(n_channels=input_nc, n_classes=output_nc, bilinear=True)

        # Putting the UNet model into the requested mode(train or freez)
        if self.UNet_train: self.unet.train()
        else: self.unet.eval()

        self.lstm0 = LSTM0(in_c=output_nc , ngf=ngf)
        self.lstm = LSTM(in_c=output_nc , ngf=ngf)
        self.out = nn.Conv2d(ngf, output_nc, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        :param x:  5D tensor    bz * temporal * 4 * 240 * 240
        :return:
        """
        lstm_output = []
        unet_output = []
        cell = None
        hide = None
        for t in range(self.temporal):
            im_t = x[:, t, ...]                # bz * temporal * W * H
            #print("Time : ",t," ",im_t.shape)
            #a = x.cpu().detach().numpy()
            #print(a.shape)
            if self.UNet_train:
                unet_last,  out_unet_mask = self.unet(im_t)
            else:
                with torch.no_grad():
                    #self.unet.eval()
                    unet_last,  out_unet_mask = self.unet(im_t)  # bz * 64 * W * H,  bz * 1 * W * H

            out_unet_mask = torch.sigmoid(out_unet_mask)
            unet_output.append(out_unet_mask)
            lstm_in = torch.cat((out_unet_mask, unet_last), dim=1) # bz * 65 * W * H

            if t == 0:
                cell, hide = self.lstm0(lstm_in)   # bz * ngf(64) * W * H
            else:
                cell, hide = self.lstm(lstm_in, cell, hide)

            out_lstm_mask = self.out(hide)
            lstm_output.append(out_lstm_mask)

        return torch.stack(unet_output, dim=1), torch.stack(lstm_output, dim=1)

if __name__ == "__main__":
    batch_size = 1
    num_classes = 1
    input_channels = 1
    ngf = 64

    unet_model_path  = '../../../border_loss_f1_guided/'
    unet_model_path += 'exp_12/experiments/EXP_ONLY_BORDER_BCE_EP_20_ES_200_BS_1_LR_0.0001_RS_2022/'
    unet_model_path += 'ckpts/best_model.pth'

    net = LSTM_MMUnet(unet_model_path, input_nc=1, output_nc=1, ngf=64, temporal=3, UNet_train=False, device_name='cuda')
    image = torch.randn(batch_size, 3, 1, 128, 128)    # bz * temporal * RGB channels * W * H  (To be addapted for 3D when needed)
    print(net)
    mmout, predict = net(image)
    print(mmout.shape)    # (1, 3, 1, 128, 128)
    print(predict.shape)  # (1, 3, 1, 128, 128)