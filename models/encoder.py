import torch
from torch import nn
import numpy as np
from models.encoder_cell import EncoderCell
from torch.nn.functional import softmax

class Encoder(nn.Module):
	def __init__(self, args):
		super(Encoder, self).__init__()
		self.args = args

		if self.args.elmo:
			input_size = 1024
		else:
			input_size = self.args.embedding_size

		self.cell = EncoderCell(hidden_size=self.args.hidden_size, input_size=input_size,
		                        threshold=self.args.threshold)

		self.output_layer = nn.Linear(in_features=self.args.hidden_size, out_features=2)

	def forward(self, embedded, mask, lengths):
		"""

		:param embedded: [batch_size, max_steps, embedding_size]
		:param mask: [batch_size, max_steps], probs of reading
		:param lengths: [batch_size]
		:return:
		"""
		batch_size = embedded.size()[0]
		max_steps = embedded.size()[1]

		last_hidden = torch.zeros(batch_size, self.args.hidden_size)
		last_cell = torch.zeros(batch_size, self.args.hidden_size)

		if torch.cuda.is_available():
			last_hidden = last_hidden.cuda()
			last_cell = last_cell.cuda()

		outputs = []

		for step in range(max_steps):
			cur_embed = embedded[:, step, :].squeeze()
			cur_mask = mask[:, step]

			mask_hidden, mask_cell = self.cell(cur_embed=cur_embed, last_hidden=last_hidden,
			                                     last_cell=last_cell, mask=cur_mask)

			outputs.append(mask_hidden)

		# [batch_size, max_steps, hidden_size]
		outputs = torch.stack(outputs).permute(1, 0, 2)

		# [batch_size, hidden_size]

		last_relevant_outputs = []

		for idx in range(batch_size):
			last_relevant_outputs.append(outputs[idx, lengths[idx]-1, :])

		last_relevant_outputs = torch.stack(last_relevant_outputs)

		logits = self.output_layer(last_relevant_outputs)

		probs = softmax(logits, dim=-1)

		# 0: democrat
		# 1: republican
		predictions = torch.argmax(probs, dim=1)

		return probs, predictions



