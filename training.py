import utils
import tensorflow as tf
from configurator import TrainConfig
from prepare_data import *
import argparse
import json
from lstm import LSTMDecoder

utils.enable_kb_interrupt()#Включаем прерывание по Ctrl+C
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Reads training data and launches LSTM network training')
    usage = 'Usage:training --input filepath [--num_layers] [--num_units] [--learning_rate] [--num_epochs] [--output_training] [--validate]' \
            '[--load_path]' \
            '  --output_dir output_dir. ' \
            "Example: py training.py --input 'train_file' --num_layers 2 --num_units 500 --learning_rate=0.01 --output_dir 'trained_model_dir'"
    parser.add_argument('--input', '-i',type=str, help='input file', default='')
    parser.add_argument('--num_layers','-l',type=int,required=False,default=None)
    parser.add_argument('--num_units', '-u', type=int, required=False, default=None)
    parser.add_argument('--learning_rate', '-r',type=float,required=False,default=None)
    parser.add_argument('--output_dir', '-o', type=str,  default='')
    parser.add_argument('--num_epochs','-e',type=int,required=False,default=10)
    parser.add_argument('--output_training','-t',type=bool,required=False,default=True)
    parser.add_argument('--validate', '-v',type=bool,required=False, default=True)
    parser.add_argument('--load_path','-p',type=str,required=False,default=None)#Путь для загрузки сохраненной ранее модели для тренировки
    parser.add_argument('--nc_file','-c',type=str,required=False,default=None)#Путь к файлу конфигурации нейронки
    args = parser.parse_args()
    print(f"Arguments:{args}")

    has_input=args.input is not None and args.input!=''
    has_output_dir= args.output_dir is not None and args.output_dir != ''
    if has_input and has_output_dir:
        (num_features,num_classes,train_data)=DataHelper.read_data(args.input)#Читаем из файла кол-во признаков, классов и данные
        print("Data loaded")
        conf_dict=None
        if args.nc_file is not None:#Загрузить конфигурацию из файла
            with open(args.nc_file, 'r') as conf_file:
                conf_dict = json.load(conf_file)
                num_units=conf_dict['num_units']
                num_layers=conf_dict['num_layers']
                learning_rate=conf_dict['learning_rate']
                #batch_size=conf_dict['batch_size']
        if args.num_units is not None:
            num_units=args.num_units
        else:
            if conf_dict is None:
                num_units=250

        if args.num_layers is not None:
            num_layers=args.num_layers
        else:
            if conf_dict is None:
                num_layers=1
        if args.learning_rate is not None:
            learning_rate=args.learning_rate
        else:
            if conf_dict is None:
                learning_rate=0.01
        #Если аргумент не задан в командной строке, то, если задан путь к файлу конф-и, то задать оттуда
        #Если не задан в командной строке и не задан путь к файлу, задать зн-е по умолчанию
        #Если задан в командной строке, то задать оттуда
        lstm:LSTMDecoder=LSTMDecoder(num_units=num_units,num_layers=num_layers,
                                     num_features=num_features,num_classes=num_classes,
                                     learning_rate=learning_rate)
        print("LSTM Network created")
        lstm.train(train_data,num_epochs=args.num_epochs,output_training=args.output_training,
                   model_dir_path=args.output_dir,validate=args.validate,model_load_path=args.load_path)
    pass