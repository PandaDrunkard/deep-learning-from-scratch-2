import sys, os
sys.path.append(os.pardir)
from dataset import ptb

corpus, word_to_id, id_to_word = ptb.load_data('train')

print(corpus.shape)

for word in ('car', 'happy', 'lexus'):
    print('%s : %d' % (word, word_to_id[word]))