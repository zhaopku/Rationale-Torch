import torch
from torch import nn
import numpy

class EncoderCell(nn.Module):
	def __init__(self, hidden_size, input_size, forget_bias=1.0, threshold=0.5):
		super(EncoderCell, self).__init__()
		self.hidden_size = hidden_size
		self.input_size = input_size

		self.forget_bias = forget_bias
		self.threshold = threshold

		self.fully_connected = nn.Linear(in_features=self.hidden_size+self.input_size, out_features=4*self.hidden_size)

	def forward(self, cur_embed, last_hidden, last_cell, mask):
		"""

		:param cur_embed: [batch_size, embedding_size]
		:param last_hidden: [batch_size, hidden_size]
		:param last_cell: [batch_size, hidden_size]
		:param mask: [batch_size], probs of read
		:return:
		"""

		# [batch_size, embedding_size+hidden_size]
		inputs = torch.cat([cur_embed, last_hidden], dim=1)

		gate_inputs = self.fully_connected(inputs)

		# i = input_gate, j = new_input, f = forget_gate, o = output_gate
		# [batch_size, hidden_size]
		i, j, f, o = torch.chunk(gate_inputs, chunks=4, dim=1)

		# [batch_size, hidden_size]
		new_cell = torch.add(torch.mul(last_cell, torch.sigmoid(torch.add(f, self.forget_bias))),
		               torch.mul(torch.sigmoid(i), torch.tanh(j)))

		new_hidden = torch.mul(torch.tanh(new_cell), torch.sigmoid(o))

		# [batch_size, 1]
		mask = torch.unsqueeze(mask, dim=1)

		# [batch_size, hidden_size] 1: read, 0: skip
		mask_cell = torch.add(torch.mul(mask, new_cell), torch.mul((1 - mask), last_cell))
		mask_hidden = torch.add(torch.mul(mask, new_hidden), torch.mul((1 - mask), last_hidden))

		return mask_hidden, mask_cell
