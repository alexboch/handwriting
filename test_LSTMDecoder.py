from unittest import TestCase
import prepare_data as prepdata
from decoder_factories import *
import constants

class TestLSTMDecoder(TestCase):
    def test_decoding_very_small(self):
        all_chars = [chr(x + 1040) for x in range(65)]  # Все символы русского алфавита
        all_chars.append(constants.CONNECTION_LABEL)  # Метка соединения
        all_chars.append(constants.NOISE_LABEL)  # Метка шума
        full_alphabet = prepdata.LabelsAlphabet(all_chars)
        ld=LSTMDecoder(num_units = 20, num_layers = 1, num_features = 2, num_classes = 69, learning_rate = 1e-2, batch_size = 1,alphabet=full_alphabet)
        train_word = prepdata.WordData()
        train_word.point_list.extend([(1, 1), (0, 0), (-1, 1), (0, 0.1)])

        train_word.labels_list.extend(['а', 'и', 'а', 'и'])
        train_word.labels_list = full_alphabet.encode_char_labels(train_word.labels_list)
        ld.train([train_word], 150,True, model_name="small_test.ckpt")
        test_word = prepdata.WordData()
        test_word.point_list.extend([(0, 0), (-1, 0.99)])
        test_word.labels_list.extend(['и', 'а'])
        test_word.labels_list = full_alphabet.encode_char_labels(test_word.labels_list)
        labels, probs = ld.label([test_word.point_list], "Models/small_test.ckpt")
        #char_labels = full_alphabet.decode_numeric_labels(labels)
        assert list(labels)==test_word.labels_list

