import os
import torch

def sequence_mask(lengths, max_len):
	"""
	tensorflow style sequence_mask

	example:

	lengths = torch.Tensor([1, 2, 4, 3, 4])
	max_len = 10

	mask = sequence_mask(lengths, max_len)

	mask:
	tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
	        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
	        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
	        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.uint8)

	:param lengths: [batch_size]
	:param max_len: scalar
	:return:
	"""

	x = torch.arange(max_len).expand(lengths.size()[0], max_len)
	y = lengths.unsqueeze_(-1).expand(-1, max_len) - 1

	if torch.cuda.is_available():
		x = x.cuda()
		y = y.cuda()

	mask = torch.le(input = x.long(), other = y.long())

	if torch.cuda.is_available():
		mask = mask.cuda()

	return mask

def construct_dir(prefix, args, create_dataset_name):

	if create_dataset_name:
		file_name = ''
		file_name += prefix + '-'
		file_name += str(args.vocab_size) + '-'
		file_name += str(args.data_size) + '-'
		file_name += str(args.max_length) + '.pkl'
		return file_name

	path = ''
	path += 'lr_'
	path += str(args.learning_rate)
	path += '_bt_'
	path += str(args.batch_size) + '-' + str(args.test_batch_size)
	path += '_d_' + str(args.drop_out)
	path += '_pe_' + str(args.pre_embedding)
	path += '_elmo_' + str(args.elmo)
	path += '_train_elmo_' + str(args.train_elmo)
	path += '_t_' + str(args.temperature)
	path += '_th_' + str(args.threshold)
	path += '_s_' + str(args.max_steps)

	path += '_gamma_' + str(args.gamma)
	path += '_theta_' + str(args.theta)

	return os.path.join(prefix, path)