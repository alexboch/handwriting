from lstm import *
from abc import ABCMeta

class DecoderFactory(metaclass=ABCMeta):

    def __init__(self,num_units,num_layers,num_classes,num_features,learning_rate,batch_size):
        self.num_units=num_units
        self.num_layers=num_layers
        self.num_classes=num_classes
        self.num_features=num_features
        self.learning_rate=learning_rate
        self.batch_size=batch_size

    def CreateDecoder(self):
        decoder=LSTMDecoder(num_units=self.num_units,num_layers=self.num_layers,num_classes=self.num_classes,num_features=self.num_features,learning_rate=self.learning_rate,batch_size=self.batch_size)
        return decoder
    pass

class ConnectionsOnlyDecoderFactory(DecoderFactory):
    def __init__(self):
        super(ConnectionsOnlyDecoderFactory,self).__init__(num_units=75,num_layers=1,num_classes=3,num_features=2,learning_rate=0.01,batch_size=1)

    pass

class FullAlphabetDecoderFactory(DecoderFactory):
    def __init__(self):
        super(FullAlphabetDecoderFactory,self).__init__(num_units = 300, num_layers = 1, num_features = 2, num_classes = 69, learning_rate = 1e-5, batch_size = 1)
    pass