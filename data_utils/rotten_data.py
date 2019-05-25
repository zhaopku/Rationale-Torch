import os
from collections import defaultdict, Counter
import numpy as np
import random
from tqdm import tqdm
import nltk
from torch.utils.data.dataset import Dataset
import torch

class Sample:
	def __init__(self, data, words, steps, label, length, id):
		self.word_ids = data[0:steps]
		self.sentence = words[0:steps]
		self.length = length
		self.label = label
		self.id = id


class Batch:
	def __init__(self, samples):
		self.samples = samples
		self.batch_size = len(samples)

class RottenDataSet(Dataset):
	def __init__(self, samples):
		super(RottenDataSet, self).__init__()
		self.samples = samples

	def __getitem__(self, index):
		sample = self.samples[index]

		return sample.id, torch.tensor(sample.word_ids), sample.length, sample.label

	def __len__(self):
		return len(self.samples)

class RottenData:
	def __init__(self, args):
		self.args = args

		#note: use 20k most frequent words
		self.UNK_WORD = 'unk'
		self.PAD_WORD = '<pad>'
		self.BLANK = '<blank>'
		self.NEWLINE = '<newline>'

		# list of batches
		self.train_batches = []
		self.val_batches = []
		self.test_batches = []

		self.word2id = {}
		self.id2word = {}

		self.train_samples = None
		self.valid_samples = None
		self.test_samples = None
		self.pre_trained_embedding = None

		self.train_samples, self.val_samples, self.test_samples = self._create_data()

		# [num_batch, batch_size, maxStep]
		self.train_batches = self._create_batch(self.train_samples)
		self.val_batches = self._create_batch(self.val_samples)

		# note: test_batches is none here
		self.test_batches = self._create_batch(self.test_samples)

		print('Dataset created')

	def construct_dataset(self, elmo=None):
		self.training_dataset = RottenDataSet(samples=self.train_samples)
		self.val_dataset = RottenDataSet(samples=self.val_samples)
		self.test_dataset = RottenDataSet(samples=self.test_samples)


	def getVocabularySize(self):
		assert len(self.word2id) == len(self.id2word)
		return len(self.word2id)

	def _create_batch(self, all_samples, tag='test'):
		all_batches = []
		if tag == 'train':
			random.shuffle(all_samples)
		if all_samples is None:
			return all_batches

		num_batch = len(all_samples)//self.args.batch_size + 1
		for i in range(num_batch):
			samples = all_samples[i*self.args.batch_size:(i+1)*self.args.batch_size]

			if len(samples) == 0:
				continue

			batch = Batch(samples)
			all_batches.append(batch)

		return all_batches

	def _create_samples(self, file_path):

		oov_cnt = 0
		cnt = 0
		with open(file_path, 'r') as file:
			lines = file.readlines()
			all_samples = []
			for idx, line in enumerate(tqdm(lines)):
				line = line.strip()
				label = int(line[-1])
				line = line[0:-1].strip()

				words = nltk.word_tokenize(line)
				word_ids = []

				words = words[:self.args.max_length]
				length = len(words)
				cnt += length
				for word in words:
					if word in self.word2id.keys():
						id_ = self.word2id[word]
					else:
						id_ = self.word2id[self.UNK_WORD]
						print('Check!')
					if id_ == self.word2id[self.UNK_WORD] and word != self.UNK_WORD:
						oov_cnt += 1
					word_ids.append(id_)
				while len(word_ids) < self.args.max_length:
					word_ids.append(self.word2id[self.PAD_WORD])
				while len(words) < self.args.max_length:
					words.append(self.PAD_WORD)
				sample = Sample(data=word_ids, words=words,
								steps=self.args.max_length, label=label, length=length, id=idx)
				all_samples.append(sample)

		return all_samples, oov_cnt, cnt

	def create_embeddings(self):
		words = self.word2id.keys()

		glove_embed = {}

		with open(self.args.embedding_file, 'r') as glove:
			lines = glove.readlines()
			for line in tqdm(lines, desc='glove'):
				splits = line.split()
				word = splits[0]
				if len(splits) > 301:
					word = ''.join(splits[0:len(splits) - 300])
					splits[1:] = splits[len(splits) - 300:]
				if word not in words:
					continue
				embed = [float(s) for s in splits[1:]]
				glove_embed[word] = embed

		embeds = []
		for word_id in range(len(self.id2word)):
			word = self.id2word[word_id]
			if word in glove_embed.keys():
				embed = glove_embed[word]
			else:
				embed = glove_embed[self.UNK_WORD]
				self.word2id[word] = self.word2id[self.UNK_WORD]
			embeds.append(embed)

		embeds = np.asarray(embeds)

		return embeds

	def _create_data(self):

		train_path = os.path.join(self.args.data_dir, self.args.dataset, self.args.train_file)
		val_path = os.path.join(self.args.data_dir, self.args.dataset, self.args.val_file)
		test_path = os.path.join(self.args.data_dir, self.args.dataset, self.args.test_file)

		print('Building vocabularies for {} dataset'.format(self.args.dataset))
		self.word2id, self.id2word = self._build_vocab(train_path, val_path, test_path)

		print('Creating pretrained embeddings!')
		self.pre_trained_embedding = self.create_embeddings()

		print('Building training samples!')
		train_samples, train_oov, train_cnt = self._create_samples(train_path)
		val_samples, val_oov, val_cnt = self._create_samples(val_path)
		test_samples, test_oov, test_cnt = self._create_samples(test_path)

		print('OOV rate for train = {:.2%}'.format(train_oov*1.0/train_cnt))
		print('OOV rate for val = {:.2%}'.format(val_oov*1.0/val_cnt))
		print('OOV rate for test = {:.2%}'.format(test_oov*1.0/test_cnt))

		return train_samples, val_samples, test_samples

	@staticmethod
	def _read_sents(filename):
		with open(filename, 'r') as file:
			all_words = []
			lines = file.readlines()
			for idx, line in enumerate(tqdm(lines)):
				# if idx == 100000:
				#     break

				line = line.strip()[:-1]
				words = nltk.word_tokenize(line)
				all_words.extend(words)

		return all_words

	def _build_vocab(self, train_path, val_path, test_path):

		all_train_words = self._read_sents(train_path)
		all_val_words = self._read_sents(val_path)
		all_test_words = self._read_sents(test_path)

		all_words = all_train_words + all_val_words + all_test_words

		print('Number of unique words = ', len(list(set(all_words))))

		counter = Counter(all_words)

		count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

		# keep the most frequent vocabSize words, including the special tokens
		# -1 means we have no limits on the number of words
		if self.args.vocab_size != -1:
			count_pairs = count_pairs[0:self.args.vocab_size-2]

		count_pairs.append((self.UNK_WORD, 100000))
		count_pairs.append((self.PAD_WORD, 100000))

		if self.args.vocab_size != -1:
			assert len(count_pairs) == self.args.vocab_size

		words, _ = list(zip(*count_pairs))
		word_to_id = dict(zip(words, range(len(words))))

		id_to_word = {v: k for k, v in word_to_id.items()}

		return word_to_id, id_to_word

	def get_batches(self, tag='train'):
		if tag == 'train':
			return self._create_batch(self.train_samples, tag='train')
		elif tag == 'val':
			return self.val_batches
		else:
			return self.test_batches
