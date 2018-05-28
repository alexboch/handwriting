from enum import Enum
import prepare_data as pd
import trainer as tr
import configurator as conf
from lstm import *




class TrainerBuilder:
    """
    Помогает создать объект для загрузки данных и тренировки, используя заданную конфигурацию
    """

    def create_alphabet_from_chars(self, characters):
        alphabet=pd.LabelsAlphabet(characters)
        self._trainer.alphabet=alphabet


    def __init__(self,init_config=None):
        if init_config!=None:
            self.alphabet=conf.get_alphabet(init_config)
            self.featurizer = conf.get_featurizer(init_config)
            self.labels_mapper=conf.get_labels_mapper(init_config)
            self.net_config=conf.get_network_config(init_config)
            self.num_classes=self.alphabet.get_length()#Для всех символов
            self.num_epochs=conf.get_num_epochs(init_config)
            self.num_features=self.featurizer.GetNumFeatures()
            self.network=LSTMDecoder(self.net_config.num_units,self.net_config.num_layers,self.num_features,self.num_classes,
                                self.net_config.learning_rate,self.net_config.batch_size)
            self.model_name=conf.get_model_name(init_config)
            self.data_dir=conf.get_data_directory(init_config)
            self.data_loader=pd.DataHelper(self.alphabet,self.featurizer,labels_map_function=self.labels_mapper)
            #self._trainer = tr.Trainer(self.network, self.data_loader, self.num_epochs, self.model_name, self.data_dir, True)
        pass


    #def set_featurizer(self,ft_set):


    def set_learning_rate(self,learning_rate):
        self.net_config.learning_rate=learning_rate

    def build_trainer(self):
        return tr.Trainer(self.network, self.data_loader, self.num_epochs, self.model_name, self.data_dir, True)

    @property
    def trainer(self):
        return self._trainer

    @trainer.setter
    def trainer(self,value):
        self._trainer=value



    @property
    def load_dir(self):
        return self.data_dir

    @load_dir.setter
    def load_dir(self,value):
        self.data_dir=value
