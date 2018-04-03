import sys, os
sys.path.append(os.pardir)

from common.np import np
from common.util import preprocess, create_co_matrix, ppmi

text = 'you say goodbye and I say hello.'
corpus, w2id, id2w = preprocess(text)
vocab_size = len(w2id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

U, S, V = np.linalg.svd(W)

print('W=>' + str(W.shape))
print(W)
print('U=>' + str(U.shape))
print(np.round(U, 3))
print('S=>' + str(S.shape))
print(np.round(S, 3))
print('V=>' + str(V.shape))
print(np.round(V, 3))