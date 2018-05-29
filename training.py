import utils
import tensorflow as tf
from configurator import TrainConfig
from prepare_data import *
import argparse
from lstm import LSTMDecoder

utils.enable_kb_interrupt()#Включаем прерывание по Ctrl+C
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Reads training data and launches LSTM network training')
    usage = 'Usage:training --input filepath [--num_layers n1] [--num_units n2] [--learning_rate n3] [--num_epochs n4] [--output_training True|False] [--validate True|False]' \
            '  --output_dir output_dir. ' \
            "Example: py training.py --input 'train_file' --num_layers 2 --num_units 500 --learning_rate=0.01 --output_dir 'trained_model_dir'"
    parser.add_argument('--input', '-i',type=str, help='input file', default='')
    parser.add_argument('--num_layers','-l',type=int,required=False,default=1)
    parser.add_argument('--num_units', '-u', type=int, required=False, default=250)
    parser.add_argument('--learning_rate', '-r',type=float,required=False,default=1e-2)
    parser.add_argument('--output_dir', '-o', type=str,  default='')
    parser.add_argument('--num_epochs','-e',type=int,required=False,default=10)
    parser.add_argument('--output_training','-t',type=bool,required=False,default=True)
    parser.add_argument('--validate', '-v',type=bool,required=False, default=True)
    args = parser.parse_args()
    print(f"Arguments:{args}")
    has_input=args.input is not None and args.input!=''
    has_output_dir= args.output_dir is not None and args.output_dir != ''
    if has_input and has_output_dir:
        (num_features,num_classes,train_data)=DataHelper.read_data(args.input)#Читаем из файла кол-во признаков, классов и данные
        print("Data loaded")
        lstm:LSTMDecoder=LSTMDecoder(args.num_units,args.num_layers,num_features,num_classes,args.learning_rate)
        print("LSTM Network created")
        lstm.train(train_data,num_epochs=args.num_epochs,output_training=args.output_training,model_dir_path=args.output_dir,validate=args.validate)
    pass