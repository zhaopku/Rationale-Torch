from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "~/allennlp_cache/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "~/allennlp_cache/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

# Compute two different representation for each token.
# Each representation is a linear weighted combination for the
# 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
elmo = Elmo(options_file, weight_file, 2, dropout=0)

# use batch_to_ids to convert sentences to character ids
sentences = [['First', 'sentence', '.', 'china', 'swiss']]
character_ids = batch_to_ids(sentences)

embeddings = elmo(character_ids)

print()