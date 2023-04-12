# ------------------------------------------------------------------------------
#
#
#                                P̶̖̥͈͈͇̼͇̈́̈́̀͋͒̽͊͘͠h̴͙͈̦͗́̓a̴̧͗̾̀̅̇ḡ̶͓̭̝͓͎̰͓̦̭̎́͊̐̂͒͠ơ̵̘͌̿̽̑̈̾Ś̴̠́̓̋̂̃͒͆̚t̴̲͓̬͎̾͑͆͊́̕a̸͍̫͎̗̞͆̇̀̌̑͂t̸̳̼̘͕̯̠̱̠͂̔̒̐̒̕͝͝
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

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention_U_Net(nn.Module):
    def __init__(self,n_channels=1,n_classes=1):
        super(Attention_U_Net,self).__init__()

        conv_num = 24
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=n_channels,ch_out=conv_num)
        self.Conv2 = conv_block(ch_in=conv_num,ch_out=conv_num*2)
        self.Conv3 = conv_block(ch_in=conv_num*2,ch_out=conv_num*4)
        self.Conv4 = conv_block(ch_in=conv_num*4,ch_out=conv_num*8)
        self.Conv5 = conv_block(ch_in=conv_num*8,ch_out=conv_num*16)


        #################
        self.Conv_middle = conv_block(ch_in=conv_num*16,ch_out=conv_num*16)
        #################

        self.Up1 = up_conv(ch_in=conv_num*16,ch_out=conv_num*8)
        self.Att1 = Attention_block(F_g=conv_num*8,F_l=conv_num*8,F_int=conv_num*4)
        self.Up_conv1 = conv_block(ch_in=conv_num*16, ch_out=conv_num*8)

        self.Up2 = up_conv(ch_in=conv_num*8,ch_out=conv_num*4)
        self.Att2 = Attention_block(F_g=conv_num*4,F_l=conv_num*4,F_int=conv_num*2)
        self.Up_conv2 = conv_block(ch_in=conv_num*8, ch_out=conv_num*4)
        
        self.Up3 = up_conv(ch_in=conv_num*4,ch_out=conv_num*2)
        self.Att3 = Attention_block(F_g=conv_num*2,F_l=conv_num*2,F_int=64)
        self.Up_conv3 = conv_block(ch_in=conv_num*4, ch_out=conv_num*2)
        
        self.Up4 = up_conv(ch_in=conv_num*2,ch_out=conv_num)
        self.Att4 = Attention_block(F_g=conv_num,F_l=conv_num,F_int=int(conv_num/2))
        self.Up_conv4 = conv_block(ch_in=conv_num*2, ch_out=conv_num)

        self.Conv_1x1 = nn.Conv2d(conv_num,n_classes,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # Middle section
        x5 = self.Conv_middle(x5)

        # decoding + concat path
        x = self.Up1(x5)
        x4 = self.Att1(g=x,x=x4)
        x = torch.cat((x4,x),dim=1)        
        x = self.Up_conv1(x)
        
        x = self.Up2(x)
        x3 = self.Att2(g=x,x=x3)
        x = torch.cat((x3,x),dim=1)
        x = self.Up_conv2(x)

        x = self.Up3(x)
        x = self.Att3(g=x,x=x2)
        x = torch.cat((x2,x),dim=1)
        x = self.Up_conv3(x)

        x = self.Up4(x)
        x1 = self.Att4(g=x,x=x1)
        x = torch.cat((x1,x),dim=1)
        x = self.Up_conv4(x)

        x = self.Conv_1x1(x)

        return x

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        # x = F.dropout(x, p=0.5)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        # x = F.dropout(x, p=0.5)
        return x


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.LeakyReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi



def test_model():
    x = torch.randn((1,1,2048,2048))
    model = Attention_U_Net(n_channels=1, n_classes=1)
    print(model)
    m = torch.nn.LogSoftmax(dim=0)
    preds = m(model(x))
    print(f'input shape: {x.shape}')
    print(f'output shape: {preds.shape}')

if __name__ == "__main__":
    test_model()