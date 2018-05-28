from unittest import TestCase
import prepare_data as prepdata
from decoder_factories import *
from datetime import *
import constants

class TestLSTMDecoder(TestCase):
    def test_decoding_very_small(self):
        try:
            all_chars = [chr(x + 1040) for x in range(65)]  # Все символы русского алфавита
            all_chars.append(constants.CONNECTION_LABEL)  # Метка соединения
            all_chars.append(constants.NOISE_LABEL)  # Метка шума
            full_alphabet = prepdata.LabelsAlphabet(all_chars)

            ld=LSTMDecoder(num_units = 20, num_layers = 1, num_features = 2, num_classes = 67, learning_rate = 1e-2, batch_size = 1)
            train_word = prepdata.WordData()
            train_word.point_list.extend([(1, 1), (0, 0), (-1, 1), (0, 0.1)])

            train_word.labels_list.extend(['а', 'и', 'а', 'и'])

            train_word.labels_list = full_alphabet.encode_char_labels(train_word.labels_list)
            model_path="Models\\small_test2\\"+datetime.now().strftime('%d-%m-%Y-%I_%M_%S')
            ld.train([train_word], 50, True, model_name="small_test",model_dir_path=model_path)
            test_word = prepdata.WordData()
            test_word.point_list.extend([(0, 0), (-1, 0.99)])
            test_word.labels_list.extend(['и', 'а'])
            test_word.labels_list = full_alphabet.encode_char_labels(test_word.labels_list)

            true_probs=full_alphabet.one_hot(test_word.labels_list)

            probs=ld.get_probabilities([test_word.point_list],model_name='small_test',model_dir=model_path)
            #probs[0][0].shape == np.asarray(true_probs).shape
            true_probs_arr=np.asarray(true_probs)
            diff=probs[0][0]-true_probs_arr

            err=np.linalg.norm(diff)#Разница между вероятностями по норме Фробениуса
            max_norm=np.sqrt(diff.shape[0]*diff.shape[1])
            err/=max_norm
            print(f"Error:{err}")
        except Exception as exc:
            print(exc)
            self.fail()
        pass
        #labels = ld.label([test_word.point_list], model_name='small_test',model_dir="Models\\small_test\\")
        #char_labels = full_alphabet.decode_numeric_labels(labels)
        #assert list(labels[0])==test_word.labels_list

