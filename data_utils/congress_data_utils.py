import torch
from torch.utils.data.dataset import Dataset
from allennlp.modules.elmo import Elmo, batch_to_ids
import random

class Sample:
	def __init__(self):
		# id is file name
		self.id = None

		# padded
		self.words = None
		self.word_ids = []

		# True length
		self.length = -1

		# republican of democrats
		self.label = None


class Batch:

	def __init__(self, samples):
		self.samples = samples
		self.batch_size = len(samples)

class CongressDataSet(Dataset):
	def __init__(self, samples, elmo, is_training, max_steps):
		super(CongressDataSet, self).__init__()
		self.samples = samples
		self.elmo = elmo
		self.is_training = is_training
		self.max_steps = max_steps

	def __getitem__(self, index):
		sample = self.samples[index]
		#return sample.word_ids, sample.word_ids

		# during training, cut the samples when it is too long
		if self.is_training and sample.length > self.max_steps:
			start_idx = random.randint(0, sample.length-self.max_steps-1)

			words = sample.words[start_idx:start_idx+self.max_steps]
			word_ids = sample.word_ids[start_idx:start_idx+self.max_steps]
			length = self.max_steps

		else:
			words = sample.words[:self.max_steps]
			word_ids = sample.word_ids[:self.max_steps]
			length = sample.length


		if self.elmo:
			word_ids = batch_to_ids([words])
			word_ids.squeeze_(dim=0)
		else:
			word_ids = torch.tensor(word_ids)

		return sample.id, word_ids, length, sample.label

	def __len__(self):
		return len(self.samples)

	def set_is_training(self, is_training):
		self.is_training = is_training
