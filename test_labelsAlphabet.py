from unittest import TestCase
from prepare_data import LettersAlphabet

class TestLabelsAlphabet(TestCase):

    def test_encoding_decoding(self):
        all_chars = [chr(x + 1040) for x in range(65)]  # Все символы русского алфавита
        alphabet=LettersAlphabet(all_chars)
        numeric_labels=alphabet.encode_char_labels(all_chars)
        decoded_char_labels=alphabet.decode_numeric_labels(numeric_labels)
        assert decoded_char_labels==all_chars

