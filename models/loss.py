import torch
from torch import nn
from models.utils import sequence_mask

class GeneratorLoss(nn.Module):
	def __init__(self, args):
		super(GeneratorLoss, self).__init__()

		self.selection_weight = args.theta
		self.transition_weight = args.gamma

	def forward(self, mask, length):
		"""
		:param mask: [batch_size, max_steps]
		:param length: [batch_size], true length of each sample
		:return:
		"""
		# average read per sample

		mask_for_valid = sequence_mask(lengths=length, max_len=mask.size()[1])

		valid_mask = mask * mask_for_valid.float()

		selection_loss = torch.sum(valid_mask, dim=1) / length.squeeze_().float()

		# [batch_size, max_steps]

		padding = torch.zeros(mask.size()[0], 1)
		if torch.cuda.is_available():
			padding = padding.cuda()

		mask_shift_right = torch.cat([padding, mask[:, :-1]], dim=1)
		transitions = torch.abs(mask - mask_shift_right).float()
		transitions *= mask_for_valid.float()

		transitions_loss = torch.sum(transitions, dim=1) / length.float()

		generator_loss = self.selection_weight*selection_loss + self.transition_weight*transitions_loss

		return selection_loss, transitions_loss, generator_loss, valid_mask
