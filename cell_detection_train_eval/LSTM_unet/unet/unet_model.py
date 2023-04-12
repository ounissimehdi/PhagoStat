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

class UNet_LSTM_ready(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_LSTM_ready, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        conv_num = 64

        self.inc = DoubleConv(n_channels, conv_num)
        self.down1 = Down(conv_num, conv_num*2)
        self.down2 = Down(conv_num*2, conv_num*4)
        self.down3 = Down(conv_num*4, conv_num*8)
        self.down4 = Down(conv_num*8, conv_num*16)

        self.mid = DoubleConv(conv_num*16, conv_num*16)

        self.up1 = Up(conv_num*16, conv_num*8)
        self.up2 = Up(conv_num*8, conv_num*4)
        self.up3 = Up(conv_num*4, conv_num*2)
        self.up4 = Up(conv_num*2, conv_num)
        self.outc = OutConv(conv_num, n_classes)

    def forward(self, x):
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.mid(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return x, logits

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
        self.unet = UNet_LSTM_ready(n_channels=input_nc, n_classes=output_nc)
        
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