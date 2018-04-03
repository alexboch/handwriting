from enum import Enum
import prepare_data as pd
import trainer as tr

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
        self.trainer.alphabet=alphabet


    def __init__(self,init_config):
        self.trainer=tr.Trainer()
        symbols=TrainerBuilder.get_symbols(init_config)
        alphabet=pd.LabelsAlphabet(symbols)
        if init_config==TrainConfig.BORDERS:
            #Задать алфавит из символов 2-x символов границы и не-границы

            pass

    def get_trainer(self):
        pass