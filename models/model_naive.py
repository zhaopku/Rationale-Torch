import torch
from torch import nn
import numpy as np

class ModelNaive(nn.Module):
	def __init__(self, args, text_data):
		super(ModelNaive, self).__init__()
		self.args = args
		self.text_data = text_data

		self.embedding_layer = nn.Embedding.from_pretrained(torch.tensor(self.text_data.pre_trained_embedding),
		                                                    freeze=False)
		self.rnn = nn.LSTM(input_size=self.args.embedding_size, hidden_size=self.args.hidden_size,
		              batch_first=True, dropout=0.0, num_layers=1, bidirectional=False)

		self.hidden = nn.Linear(in_features=self.args.hidden_size, out_features=2, bias=True)

	def forward(self, word_ids, length):
		"""

		:param word_ids: [batch_size, max_steps]
		:param length: [batch_size]
		:return:
		"""
		# [batch_size, max_steps, embedding_size]
		embedded = self.embedding_layer(word_ids.long())
		# [batch_size, max_steps, hidden_size]
		outputs, (h_n, c_n) = self.rnn(embedded)

		last_relevant_outputs = torch.index_select(outputs, dim=1, index=length)

		logits = self.hidden(last_relevant_outputs)

		return logits





