from unittest import TestCase
from utils import decode_framewise
import numpy as np

class TestDecode_framewise(TestCase):
    def test_decode_framewise(self):
        test_sequence=np.zeros((2,5))
        test_sequence[0]=[0.8,0.9,0.1,0.2,0.9]
        test_sequence[1]=[0.2,0.1,0.9,0.8,0.1]
        decoded_classes=decode_framewise(test_sequence)
        target_classes=[1,0,1,1,0]
        assert (decoded_classes==target_classes)

    def test_one_element(self):
        test_sequence=np.zeros((2,1))
        test_sequence[0]=0.9
        test_sequence[1]=0.1
        decoded_classes=decode_framewise(test_sequence)
        target_classes=[0]
        assert (decoded_classes == target_classes)