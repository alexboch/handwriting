from enum import Enum
import prepare_data as pd
import trainer as tr
import configurator as conf
from lstm import *

class TrainConfig(Enum):
    BORDERS=1#Границы букв, метка для каждого входного вектора
    LETTERS=2#Метка буквы для каждого входного вектора
    LETTERS_MERGED=3#Метки букв, одинаковые буквы сливаются(распознавание слов без сегментации)


class TrainerBuilder:

    @staticmethod
    def get_symbols(train_config):
        pass

    def create_alphabet_from_chars(self, characters):
        alphabet=pd.LabelsAlphabet(characters)
        self._trainer.alphabet=alphabet


    def __init__(self,init_config=None):
        if init_config!=None:
            alphabet=conf.get_alphabet(init_config)
            labels_mapper=conf.get_labels_mapper(init_config)
            net_config=conf.get_network_config(init_config)
            num_classes=alphabet.get_length()+1#Для всех символов + пустая метка
            num_epochs=conf.get_num_epochs(init_config)
            network=LSTMDecoder(net_config.num_units,net_config.num_layers,net_config.num_features,num_classes,
                                net_config.learning_rate,net_config.batch_size,alphabet)
            model_name=conf.get_model_name(init_config)
            data_dir=conf.get_data_directory(init_config)
            data_loader=pd.DataHelper(alphabet,labels_mapper)
            self._trainer = tr.Trainer(network, data_loader, num_epochs, model_name, data_dir, True)
        pass




    @property
    def trainer(self):
        return self._trainer

    @property
    def trainer(self,value):
        self._trainer=value


    @property
    def load_dir(self):
        return self._trainer.data_directory

    @load_dir.setter
    def load_dir(self,value):
        self._trainer.data_directory=value


    def get_trainer(self):
        return self._trainer