import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Generator(nn.Module):
	def __init__(self, args):
		super(Generator, self).__init__()
		self.args = args

		# actually, this is an LSTM-RNN
		if self.args.elmo:
			input_size = 1024
		else:
			input_size = self.args.embedding_size

		self.rnn = nn.LSTM(input_size=input_size, hidden_size=self.args.hidden_size,
		              batch_first=True, dropout=1.0-self.args.drop_out, num_layers=self.args.gen_layers,
		              bidirectional=self.args.gen_bidirectional)

		# skip or not, 2 classes
		in_features = self.args.hidden_size
		if self.args.gen_bidirectional:
			in_features *= 2
		self.hidden = nn.Linear(in_features=in_features, out_features=2, bias=True)

	@staticmethod
	def gumbel_softmax(logits, temperature):
		# g = -log(-log(u)), u ~ U(0, 1)

		noise = torch.rand_like(logits)

		noise.add_(1e-9).log_().neg_()
		noise.add_(1e-9).log_().neg_()

		if torch.cuda.is_available():
			noise = noise.cuda()
		x = (logits + noise) / temperature
		x = F.softmax(x.view(-1, x.size()[-1]), dim=-1)

		# probs: [batch_size, max_steps, 2]
		probs = x.view_as(logits)

		return probs

	def sample(self, probs):
		hard_mask = torch.gt(probs, self.args.threshold)

		if self.training:
			mask = probs
		else:
			mask = hard_mask

		return mask

	def forward(self, x):
		"""

		:param x: [batch_size, max_steps, embedding_size]
		:return:
		"""
		# outputs: [batch_size, max_steps, hidden_size]
		outputs, (h_n, c_n) = self.rnn(x)
		# logits: [batch_size, max_steps, 2]
		logits = self.hidden(outputs)

		probs = self.gumbel_softmax(logits, self.args.temperature)

		# prob of the corresponding words being read
		probs_read = probs[:, :, 1]
		# probs_read: [batch_size, max_steps]
		# mask: [batch_size, max_steps]
		mask = self.sample(probs_read)
		mask = mask.float()
		return mask
