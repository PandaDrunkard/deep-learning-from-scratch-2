import unittest

import sys, os
sys.path.append(os.pardir)
from common.np import *
from common.util import im2col, col2im, clip_grads, preprocess, \
    convert_on_hot, create_co_matrix, cos_similarity, most_similar

class UtilTest(unittest.TestCase):
    def test_im2col_transforms(self):
        x = np.random.randn(5, 1, 28, 28)
        col = im2col(x, 4, 4, stride=1, pad=0)

        self.assertSequenceEqual((3125, 16), col.shape)

    def test_col2im_transforms(self):
        col = np.random.randn(3125, 16)
        x = col2im(col, (5, 1, 28, 28), 4, 4, stride=1, pad=0)

        self.assertSequenceEqual((5, 1, 28, 28), x.shape)

    def test_preprocess_returns_corpus(self):
        text = 'you say goodbye and I say hello.'
        corpus, word_to_id, id_to_word = preprocess(text)

        ex_corpus = np.array([0,1,2,3,4,1,5,6])
        ex_w2id = {'you': 0, 'say': 1, 'goodbye': 2, 'and': 3, 'i': 4, 'hello': 5, '.': 6}
        ex_id2w = {0: 'you', 1: 'say', 2: 'goodbye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'}

        self.assertTrue((ex_corpus == corpus).all())
        self.assertDictEqual(ex_w2id, word_to_id)
        self.assertDictEqual(ex_id2w, id_to_word)
    
    def test_convert_one_hot(self):
        text = 'you say goodbye and I say hello.'
        corpus, w2id, id2w = preprocess(text)

        one_hot = convert_on_hot(corpus, len(w2id))

        self.assertTrue((np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ]) == one_hot).all())

    def test_create_co_matrix(self):
        text = 'you say goodbye and I say hello.'
        corpus, w2id, id2w = preprocess(text)

        co_matrix = create_co_matrix(corpus, len(w2id), window_size=1)

        self.assertTrue((np.array([
            [0, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 1, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0]
        ]) == co_matrix).all())

    def test_cos_similarity(self):
        text = 'you say goodbye and I say hello.'
        corpus, w2id, id2w = preprocess(text)

        vocab_size = len(w2id)

        C = create_co_matrix(corpus, vocab_size)

        c0 = C[w2id['you']]
        c1 = C[w2id['i']]

        expected_c0 = 0.9999999800000005
        expected_c1 = 0.7071067691154799

        self.assertEqual(cos_similarity(c0, c0), expected_c0)
        self.assertEqual(cos_similarity(c0, c1), expected_c1)

    def test_most_similar(self):
        text = 'you say goodbye and I say hello.'
        corpus, w2id, id2w = preprocess(text)
        vocab_size = len(w2id)
        C = create_co_matrix(corpus, vocab_size)

        most_similar('you', w2id, id2w, C, top=5)

if __name__ == '__main__':
    unittest.main()