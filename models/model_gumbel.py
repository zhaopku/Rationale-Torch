import math
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids
from models.generator import Generator
from models.encoder import Encoder

# elmo params
options_file = "~/allennlp_cache/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "~/allennlp_cache/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

class ModelGumbel(nn.Module):
	def __init__(self, args, text_data):

		super(ModelGumbel, self).__init__()
		self.args = args
		self.text_data = text_data
		# embedding layer
		if self.args.pre_embedding and not self.args.elmo:
			# pre_trained embeddings are 300 dimensional, trainable
			self.embedding_layer = nn.Embedding.from_pretrained(torch.Tensor(self.text_data.pre_trained_embedding), freeze=False)

		elif self.args.elmo:
			self.embedding_layer = Elmo(options_file, weight_file, 1, dropout=1.0-self.args.drop_out,
			                            requires_grad=self.args.train_elmo)
		else:
			self.embedding_layer = nn.Embedding(num_embeddings=self.text_data.getVocabularySize(),
			                                          embedding_dim=self.args.embedding_size)

		# first generator
		self.generator = Generator(args=self.args)

		# then encoder
		self.encoder = Encoder(args=self.args)

	def forward(self, x, lengths):
		"""

		:param x: [batch_size, max_steps]
		:param lengths: [batch_size]
		:return:
		"""
		# embedded: [batch_size, max_steps, embedding_size]

		embedded = self.embedding_layer(x.long())
		if self.args.elmo:
			embedded = embedded['elmo_representations'][0]
		# mask: [batch_size, max_steps], continuous during training, discrete during val & test
		# 1: read
		# 0: skip
		mask = self.generator(embedded)

		# predictions: [batch_size, 2], predictions of labels
		probs, predictions = self.encoder(embedded, mask, lengths)

		return predictions, probs, mask



