from enum import Enum
import prepare_data as pd
import trainer as tr
import configurator as conf

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


    def __init__(self,init_config):
        self._trainer=tr.Trainer()
        alphabet=conf.get_alphabet(init_config)
        labels_mapper=conf.get_labels_mapper(init_config)
        net_config=conf.get_network_config(init_config)

        pass

    def get_trainer(self):
        return self._trainer