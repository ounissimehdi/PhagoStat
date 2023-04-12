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


# Add comment : this is the first LSTM to be used in the begining of the sequence only 
class LSTM0(nn.Module):
	def __init__(self, in_c=1, ngf=32):
		super(LSTM0, self).__init__()
		self.conv_gx_lstm0 = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1)
		self.conv_ix_lstm0 = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1)
		self.conv_ox_lstm0 = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1)

	def forward(self, xt):
		"""
		:param xt:      bz * 5(num_class) * 240 * 240
		:return:
			hide_1:    bz * ngf(32) * 240 * 240
			cell_1:    bz * ngf(32) * 240 * 240
		"""
		gx = self.conv_gx_lstm0(xt)
		ix = self.conv_ix_lstm0(xt)
		ox = self.conv_ox_lstm0(xt)

		gx = torch.tanh(gx)
		ix = torch.sigmoid(ix)
		ox = torch.sigmoid(ox)

		cell_1 = torch.tanh(gx * ix)
		hide_1 = ox * cell_1
		return cell_1, hide_1

class LSTM(nn.Module):
	def __init__(self, in_c=1, ngf=64):
		super(LSTM, self).__init__()
		self.conv_ix_lstm = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1, bias=True)
		self.conv_ih_lstm = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False)

		self.conv_fx_lstm = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1, bias=True)
		self.conv_fh_lstm = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False)

		self.conv_ox_lstm = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1, bias=True)
		self.conv_oh_lstm = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False)

		self.conv_gx_lstm = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1, bias=True)
		self.conv_gh_lstm = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False)

	def forward(self, xt, cell_t_1, hide_t_1):
		"""
		:param xt:          bz * (5+32) * 240 * 240
		:param hide_t_1:    bz * ngf(32) * 240 * 240
		:param cell_t_1:    bz * ngf(32) * 240 * 240
		:return:
		"""
		gx = self.conv_gx_lstm(xt)         # output: bz * ngf(32) * 240 * 240
		gh = self.conv_gh_lstm(hide_t_1)   # output: bz * ngf(32) * 240 * 240
		g_sum = gx + gh
		gt = torch.tanh(g_sum)

		ox = self.conv_ox_lstm(xt)          # output: bz * ngf(32) * 240 * 240
		oh = self.conv_oh_lstm(hide_t_1)    # output: bz * ngf(32) * 240 * 240
		o_sum = ox + oh
		ot = torch.sigmoid(o_sum)

		ix = self.conv_ix_lstm(xt)              # output: bz * ngf(32) * 240 * 240
		ih = self.conv_ih_lstm(hide_t_1)        # output: bz * ngf(32) * 240 * 240
		i_sum = ix + ih
		it = torch.sigmoid(i_sum)

		fx = self.conv_fx_lstm(xt)              # output: bz * ngf(32) * 240 * 240
		fh = self.conv_fh_lstm(hide_t_1)        # output: bz * ngf(32) * 240 * 240
		f_sum = fx + fh
		ft = torch.sigmoid(f_sum)

		cell_t = ft * cell_t_1 + it * gt        # bz * ngf(32) * 240 * 240
		hide_t = ot * torch.tanh(cell_t)            # bz * ngf(32) * 240 * 240

		return cell_t, hide_t