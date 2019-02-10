import torch
from tqdm import tqdm
import argparse
import os
import torch.optim as optimizer
from torch.utils.data import DataLoader
from models import utils
from models.congress_data import CongressData
import pickle as p
from models.model_gumbel import ModelGumbel
from models.loss import GeneratorLoss
import numpy as np

class Train:
	def __init__(self):
		self.args = None
		self.training_set = None
		self.val_set = None
		self.train_loader = None
		self.val_loader = None
		self.model = None

		self.result_dir = None

		self.cur_best_val_acc = 0.0

	@staticmethod
	def parse_args(args):
		parser = argparse.ArgumentParser()

		parser.add_argument('--result_dir', type=str, default='result', help='result directory')
		parser.add_argument('--model_dir', type=str, default='saved_models', help='result directory')
		parser.add_argument('--test_dir', type=str, default='test_result')
		# data location
		data_args = parser.add_argument_group('Dataset options')

		data_args.add_argument('--dataset_name', type=str, default='dataset', help='a TextData object')

		data_args.add_argument('--dataset', type=str, default='congress')

		data_args.add_argument('--data_dir', type=str, default='data', help='dataset directory, save pkl here')
		data_args.add_argument('--embedding_file', type=str, default='glove.840B.300d.txt')
		data_args.add_argument('--vocab_size', type=int, default=-1, help='vocab size, use the most frequent words')

		data_args.add_argument('--congress_dir', type=str, default='/Users/mengzhao/congress_data/gpo/H')
		data_args.add_argument('--max_length', type=int, default=1000, help='max length of samples')
		data_args.add_argument('--data_size', type=int, default=1000, help='number of subdirs to include')

		# neural network options
		nn_args = parser.add_argument_group('Network options')
		nn_args.add_argument('--embedding_size', type=int, default=300)
		nn_args.add_argument('--hidden_size', type=int, default=200)
		nn_args.add_argument('--gen_layers', type=int, default=2)
		nn_args.add_argument('--gen_bidirectional', action='store_true')

		nn_args.add_argument('--max_steps', type=int, default=1000, help='number of steps in RNN')
		nn_args.add_argument('--n_classes', type=int, default=2)
		nn_args.add_argument('--dependent', action='store_true', help='two kinds of rationales, only independent is supported at the moment')
		nn_args.add_argument('--r_unit', type=str, default='lstm', choices=['lstm', 'rcnn'], help='only support lstm at the moment')

		# training options
		trainingArgs = parser.add_argument_group('Training options')
		trainingArgs.add_argument('--rl', action='store_true', help='whether or not to use REINFORCE algorithm')
		trainingArgs.add_argument('--pre_embedding', action='store_true')
		trainingArgs.add_argument('--elmo', action='store_true')
		trainingArgs.add_argument('--train_elmo', action='store_true')
		trainingArgs.add_argument('--drop_out', type=float, default=1.0, help='dropout rate for RNN (keep prob)')
		trainingArgs.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')

		# using different batch_size for training and evaluation
		trainingArgs.add_argument('--batch_size', type=int, default=20, help='batch size')
		trainingArgs.add_argument('--test_batch_size', type=int, default=20, help='test batch size')

		trainingArgs.add_argument('--epochs', type=int, default=2000, help='most training epochs')
		trainingArgs.add_argument('--load_model', action='store_true', help='whether or not to use old models')
		trainingArgs.add_argument('--theta', type=float, default=0.0, help='for #choices')
		trainingArgs.add_argument('--gamma', type=float, default=0.0, help='for continuity')
		trainingArgs.add_argument('--temperature', type=float, default=0.5, help='gumbel softmax temperature')
		trainingArgs.add_argument('--threshold', type=float, default=0.5, help='threshold for producing hard mask')
		trainingArgs.add_argument('--test_model', action='store_true')

		return parser.parse_args(args)

	def construct_data(self):
		self.data_dir = os.path.join(self.args.data_dir, self.args.dataset)
		self.dataset_name = utils.construct_dir(prefix=self.args.dataset, args=self.args, create_dataset_name=True)
		dataset_file_name = os.path.join(self.data_dir, self.dataset_name)

		if not os.path.exists(dataset_file_name):
			if self.args.dataset == 'rotten':
				raise NotImplementedError
			elif self.args.dataset == 'congress':
				self.text_data = CongressData(args=self.args)
			else:
				print('Cannot recognize {}'.format(self.args.dataset))
				raise NotImplementedError

			with open(dataset_file_name, 'wb') as datasetFile:
				p.dump(self.text_data, datasetFile)
			print('dataset created and saved to {}, exiting ...'.format(dataset_file_name))
			exit(0)
		else:
			with open(dataset_file_name, 'rb') as datasetFile:
				self.text_data = p.load(datasetFile)
			print('dataset loaded from {}'.format(dataset_file_name))

		self.text_data.construct_dataset(elmo=self.args.elmo)

		self.train_loader = DataLoader(dataset=self.text_data.training_dataset, num_workers=1, batch_size=self.args.batch_size, shuffle=False)
		self.val_loader = DataLoader(dataset=self.text_data.val_dataset, num_workers=1, batch_size=self.args.test_batch_size, shuffle=False)
		self.test_loader = DataLoader(dataset=self.text_data.test_dataset, num_workers=1, batch_size=self.args.test_batch_size, shuffle=False)


	def construct_model(self):
		self.model = ModelGumbel(args=self.args, text_data=self.text_data)
		self.optimizer = optimizer.Adam(self.model.parameters(), lr=self.args.learning_rate)

		self.generator_loss = GeneratorLoss(args=self.args)
		self.encoder_loss = torch.nn.BCELoss(reduction='none')

	def construct_out_dir(self):
		self.model_dir = utils.construct_dir(prefix=self.args.model_dir, args=self.args, create_dataset_name=False)
		self.out_dir = utils.construct_dir(prefix=self.args.test_dir, args=self.args, create_dataset_name=False)
		self.result_file = self.model_dir.split('/')[-1]
		self.out_path = os.path.join(self.args.result_dir, self.result_file)

		if not os.path.exists(self.args.result_dir):
			os.makedirs(self.args.result_dir)

		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)

		if not os.path.exists(self.out_dir):
			os.makedirs(self.out_dir)


	def main(self, args=None):
		print('PyTorch Version {}, GPU enabled {}'.format(torch.__version__, torch.cuda.is_available()))
		self.args = self.parse_args(args=args)

		self.construct_data()

		self.construct_model()

		self.construct_out_dir()

		# self.model_path = './saved_models/up_2.pth'
		# if torch.cuda.is_available():
		# 	(generate_state_dict, discriminator_state_dict) = torch.load(self.model_path, map_location='gpu')
		# else:
		# 	(generate_state_dict, discriminator_state_dict) = torch.load(self.model_path, map_location='cpu')
		#
		# self.generator.load_state_dict(generate_state_dict)
		# self.discriminator.load_state_dict(discriminator_state_dict)

		with open(self.out_path, 'w') as self.out:

			for k, v in vars(self.args).items():
				self.out.write('{} = {}\n'.format(str(k), str(v)))
			self.out.write('\n\n')

			self.train_loop()

	def train_loop(self):

		# put model to GPU if available
		if torch.cuda.is_available():
			self.model.cuda()

		# for idx, (id_, word_ids, lengths, labels) in enumerate(tqdm(self.val_loader)):
		# 	print()

		for e in range(self.args.epochs):
			# switch the model to training mode
			self.model.train()

			train_results = {'accuracy':0.0, 'read_rate_per_sample':0.0, 'selection_loss': 0.0,
			              'transition_loss': 0.0, 'generator_loss': 0.0, 'encoder_loss':0.0, 'n_samples':0}
			all_read_rates = []
			all_lengths = []

			for idx, (id_, word_ids, lengths, labels) in enumerate(tqdm(self.train_loader)):
				# put data to GPU
				cur_batch_size = lengths.size()[0]
				if torch.cuda.is_available():

					word_ids = word_ids.cuda()
					lengths = lengths.cuda()
					labels = labels.cuda()


				self.model.zero_grad()
				# 0: democrat
				# 1: republican
				predictions, probs, mask = self.model(word_ids, lengths)
				probs_republican = probs[:, 1]

				selection_loss, transitions_loss, generator_loss, valid_mask = self.generator_loss(mask, lengths)
				encoder_loss = self.encoder_loss(probs_republican.float(), labels.float())

				train_results['selection_loss'] += selection_loss.data.sum()
				train_results['transition_loss'] += transitions_loss.data.sum()
				train_results['generator_loss'] += generator_loss.data.sum()
				train_results['encoder_loss'] += encoder_loss.data.sum()

				train_results['n_samples'] += cur_batch_size
				train_results['accuracy'] += torch.sum(predictions.data.int() == labels.data.int())
				all_read_rates.extend(valid_mask.cpu().data.numpy())
				all_lengths.extend(lengths.cpu().data.numpy())

				generator_loss_avg = generator_loss.sum() / cur_batch_size
				encoder_loss_avg = encoder_loss.sum() / cur_batch_size

				loss = generator_loss_avg + encoder_loss_avg
				# loss = encoder_loss_avg

				loss.backward()
				# for p in self.model.parameters():
				# 	print(p.grad)

				self.optimizer.step()

			train_results['read_rate_per_sample'] = np.sum(all_read_rates) / np.sum(all_lengths)

			train_results['accuracy'] = float(train_results['accuracy']) / train_results['n_samples']

			result_line = 'Epoch %d, Train, ' % e
			for k, v in train_results.items():
				result_line += '{} = {}, '.format(k, v)
			print(result_line)
			self.out.write(result_line+'\n\n')
			self.out.flush()

			continue

			self.validate(epoch=e, mode='val')
			self.validate(epoch=e, mode='test')


	def validate(self, epoch, mode='test'):
		if mode == 'test':
			loader = self.test_loader
		else:
			loader = self.val_loader

		with torch.no_grad():
			self.model.eval()

			train_results = {'accuracy':0.0, 'read_rate_per_sample':0.0, 'selection_loss': 0.0,
			              'transition_loss': 0.0, 'generator_loss': 0.0, 'encoder_loss':0.0, 'n_samples':0}
			all_read_rates = []
			all_lengths = []

			for idx, (id_, word_ids, lengths, labels) in enumerate(tqdm(loader)):
				# put data to GPU
				cur_batch_size = lengths.size()[0]
				if torch.cuda.is_available():
					word_ids = word_ids.cuda()
					lengths = lengths.cuda()
					labels = labels.cuda()

				# 0: democrat
				# 1: republican
				predictions, probs, mask = self.model(word_ids, lengths)
				probs_republican = probs[:, 1]

				selection_loss, transitions_loss, generator_loss, valid_mask = self.generator_loss(mask, lengths)
				encoder_loss = self.encoder_loss(probs_republican.float(), labels.float())

				train_results['selection_loss'] += selection_loss.data.sum()
				train_results['transition_loss'] += transitions_loss.data.sum()
				train_results['generator_loss'] += generator_loss.data.sum()
				train_results['encoder_loss'] += encoder_loss.data.sum()

				train_results['n_samples'] += cur_batch_size
				train_results['accuracy'] += torch.sum(predictions.data.int() == labels.data.int())
				all_read_rates.extend(valid_mask.cpu().data.numpy())
				all_lengths.extend(lengths.cpu().data.numpy())

				# if idx == 0:
				# 	break

			train_results['read_rate_per_sample'] = np.sum(all_read_rates) / np.sum(all_lengths)

			train_results['accuracy'] = float(train_results['accuracy']) / train_results['n_samples']

			result_line = '\t {} \t'.format(mode)
			for k, v in train_results.items():
				result_line += '{} = {}, '.format(k, v)
			print(result_line)
			self.out.write(result_line+'\n')
			self.out.flush()

			if train_results['accuracy'] >= self.cur_best_val_acc and mode != 'test':
				save_path = os.path.join(self.model_dir, 'model.pth')
				print('saving models at {}'.format(save_path))
				self.out.write('saving models at {}\n'.format(save_path))
				torch.save(self.model.state_dict(), os.path.join(self.model_dir, 'model.pth'))

			# if train_results['accuracy'] >= self.cur_best_val_acc and mode == 'test':
			# 	for idx, (id_, word_ids, lengths, labels) in enumerate(tqdm(loader)):
			# 		with open(os.path.join(self.out_dir, id_), 'w') as file:
			# 			masks = all_read_rates[idx]
			#
			# 			if sample.label == all_predictions[idx]:
			# 				file.write('correct\n')
			# 			else:
			# 				file.write('wrong\n')
			#
			# 			for i, word in enumerate(sample.words[:sample.length]):
			# 				file.write('{}({}) '.format(word, masks[i]))
			# 				if i % 20 == 0:
			# 					file.write('\n')
