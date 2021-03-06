from enum import Enum
import tensorflow as tf
import csv
import time
import datetime
import json
from utils import *
from graph_helper import GraphHelper

map_fn = tf.map_fn

class Loss(Enum):
    Sequence='Sequence'
    CrossEntropy='CrossEntropy'

class LSTMDecoder:
    """
    Класс для создания, обучения и получения разметки от LSTM-нейросети TODO:Сделать другие варианты клеток, кроме LSTM
    """
    #def __init__(self, num_units, num_layers, num_features, num_classes, learning_rate, batch_size=1):
    def __init__(self,**kwargs):
        """
        Конструктор, в нем задаются размеры слоев и создается клетка сети
        """
        print(f"Network constructor parameters:\n{kwargs}")
        num_units=kwargs.get('num_units',250)
        num_layers=kwargs.get('num_layers',2)
        num_features=kwargs.pop('num_features')
        learning_rate=kwargs.get('learning_rate',1e-2)
        batch_size=kwargs.get('batch_size',1)
        num_classes=kwargs.pop('num_classes')
        loss=kwargs.pop('loss')
        self.create_network(num_units, num_layers, num_features,
                            num_classes, learning_rate, batch_size,loss)

    @staticmethod
    def from_file(file_path):
        with open(file_path,'r') as conf_file:
            conf_dict=json.load(conf_file)
            return LSTMDecoder(**conf_dict)

    def save_net_config(self,path):
        conf_dict = {'num_layers': self.num_layers, 'num_units': self.num_units,
                     'learning_rate': self.learning_rate }
        with open(path, 'w') as net_config_file:
            json.dump(conf_dict, net_config_file)

    def get_probabilities(self,points,model_name,model_dir):
        """
        Возвращает вероятностное распределение
        :param points:Список входных точек
        :param model_name:
        :param model_dir:
        :return:Вероятности каждого класса для каждой точки каждого списка
        """
        model_name = os.path.join(model_dir, model_name + ".meta")
        with tf.Session() as session:
            with tf.name_scope("restore_inference"):
                saver = tf.train.import_meta_graph(model_name)
                saver.restore(session, tf.train.latest_checkpoint(model_dir))  # Загрузить натренированную модель
            probs_list = []
            for points_list in points:
                input_arr = np.asarray(points_list)
                input_arr = np.expand_dims(input_arr, 0)
                with tf.name_scope("get_probs"):
                    probs = session.run([self.probs],
                                                           feed_dict={self.inputs: input_arr,
                                                                      self.seq_len: [len(points_list)]})
                probs_list.append(probs[0])
        return probs_list

    TINY = 1e-6  # to avoid NaNs in logs


    def get_batch_feed(self,words:list,batch_size:int,keep_prob:float):
        lt = len(words)
        points_num=0
        for j in np.arange(0, lt, batch_size):  # Цикл по всем тренировочным словам, берем по batch_size слов
            real_batch_size = batch_size if j + batch_size < lt else lt - j
            max_index = j + real_batch_size
            multiples = [real_batch_size, 1, 1]
            batch_words = words[j:max_index]  # слова для создания батча
            np.random.shuffle(batch_words)
            max_len = max([w.length for w in batch_words])
            batch_inputs = [batch_word.point_list for batch_word in batch_words]
            seq_lengths = [batch_word.length for batch_word in batch_words]  # Вектор длин каждой последовательности
            points_num += sum(batch_word.length for batch_word in batch_words)
            inputs_arr = np.zeros((real_batch_size, max_len, self.num_features))
            if self.loss_kind == Loss.Sequence:
                targets_array = np.zeros((real_batch_size, max_len))
            else:
                targets_array = np.zeros((real_batch_size, max_len, self.num_outputs))
            for batch_num in range(real_batch_size):
                w = batch_words[batch_num]
                batch = batch_inputs[batch_num]
                l = seq_lengths[batch_num]
                inputs_arr[batch_num][:l] = batch
                targets_array[batch_num][:l] = w.labels_list
            # Подаем батч на вход и получаем результат
            feed_dict = {self.inputs: inputs_arr, self.targets: targets_array,
                         self.seq_len: seq_lengths, self.keep_prob: keep_prob, self.multiples: multiples
                         }
            if self.loss_kind == Loss.Sequence:
                weights = np.zeros((real_batch_size, max_len))
                for k in range(real_batch_size):
                    wl = seq_lengths[k]
                    w = batch_words[k]
                    weights[k][:wl] = w.weights
                feed_dict[self.entropy_weights] = weights
            yield feed_dict,points_num#Вернуть словарь для передачи в модель и кол-во точек в батче
                # feed_dict[self.entropy_weights]=[batch_word.weights for batch_word in batch_words][0]

    def train(self, words, num_epochs=100, output_training=False, model_name="model",
              model_dir_path=f"Models{os.sep}model",validate=True,keep_prob=0.5,model_load_path=None,batch_size=5):
        """
        :param words: Список слов, содержащих точки и метки
        """


        os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
        LOG_DIR = "Summary"

        train_writer = tf.summary.FileWriter(os.path.join(model_dir_path, LOG_DIR))
        val_writer=tf.summary.FileWriter(os.path.join(model_dir_path,os.path.join(LOG_DIR,"validation")))
        print("starting training,epochs:",num_epochs,"learning rate:",self.learning_rate)
        last_epoch_num=0
        gpu_options = tf.GPUOptions(allow_growth=True)
        session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        epoch_errors = []
        merged=tf.summary.merge_all()
        if model_load_path is not None:#Если задан путь, то загрузить и дотренировать
            load_dir = os.path.dirname(model_load_path)
            print(f"Loading model from{load_dir}")
            model_name=os.path.splitext(os.path.basename(model_load_path))[0]
            print(f"Model name:{model_name}")
            with tf.name_scope("restore_train"):
                saver=tf.train.Saver()
                print(f"Restoring checkpoint from {load_dir}")
                latest_ckeckpoint_path=tf.train.latest_checkpoint(load_dir)
                print(f"full path to the latest checkpoint:{latest_ckeckpoint_path}")
                saver.restore(session,latest_ckeckpoint_path)
        else:
            session.run(tf.global_variables_initializer())
        num_batches = len(words) / self.batch_size
        num_words = len(words)
        total_points_num=0#Общее кол-во точек во всех образцах
        output_period=max(num_epochs/100.0*2,1)#Выводить каждый раз, когда прошло 2% обучения
        #output_period=1
        start_time=time.time()
        start_datetime=datetime.datetime.now()
        for word in words:
            word.length=len(word.point_list)
            word.weights=[]
        words=list(filter(lambda w:w.length>0,words))#TODO:Убрать и перенести фильтрацию в скрипт подготовки данных
        weighted_words=list(words)

        if self.loss_kind==Loss.Sequence:
            for word in weighted_words:
                labels_arr=np.asarray(word.labels_list)
                num_for_classes=np.zeros(self.num_outputs)
                for nc in range(self.num_outputs):#Посчитать, сколько объектов каждого класса входит
                    num_for_classes[nc]=np.count_nonzero(labels_arr==nc)#Количество точек данного класса
                #Задать веса
                word.weights = np.ones((self.batch_size, len(word.point_list)), dtype=np.float32)
                class_weights=np.ones(self.num_outputs)
                n0=num_for_classes[np.where(num_for_classes>0)][0]
                i=0
                for ni in num_for_classes:
                    x=n0/ni
                    class_weights[i]=x
                    i+=1
                for j in range(self.num_outputs):
                    class_mask=(labels_arr==j)
                    word.weights[0][class_mask]=class_weights[j]

        data_len=len(weighted_words)
        train_len=data_len#Если нет валидации, берем весь массив слов
        validate=validate and data_len>1#Если слово только 1, то валидацию следать не сможем
        if data_len<=1:
            print("Dataset is too small to partition to train and validation sets!")
        if validate:
            np.random.shuffle(weighted_words)
            train_len=int(data_len*0.8)
            validation_data=weighted_words[train_len:]
        training_words = weighted_words[:train_len]
        train_points_num=0
        try:
            for i in np.arange(num_epochs):
                epoch_num=last_epoch_num+i
                epoch_errors_data=dict()
                np.random.shuffle(training_words)
                train_epoch_loss = 0  # Средняя ошибка по всем батчам в данной эпохе
                can_output=output_training and (i%output_period==0 or i==num_epochs-1)
                #training_words = [training_words[0]]  # TODO Убрать
                lt = len(training_words)

                batch_num=0
                for feed_dict,pt_num in self.get_batch_feed(training_words,batch_size,keep_prob):  # Цикл по всем тренировочным словам, берем по batch_size слов
                    train_points_num+=pt_num
                    step,summary,loss,probs,_ = session.run([self.global_step,merged,self.loss,self.probs,self.train_fn],
                                                                           feed_dict=feed_dict)
                    train_writer.add_summary(summary, step)
                    train_epoch_loss+=loss
                    if can_output:
                        print(f"Epoch:{epoch_num},batch number {batch_num} batch loss:{loss}")

                    batch_num+=1
                """Конец эпохи(Прошли весь тренировочный датасет)"""
                #Валидация
                if validate:
                    validation_epoch_norm=0#Среднее значение нормы на эпохе для датасета валидации
                    validation_epoch_nn=0
                    validation_epoch_cost=0
                    validation_loss_sum=0

                    #for val_word in validation_data:#Для каждого слова валидации
                    for val_feeds,_ in self.get_batch_feed(validation_data,batch_size,keep_prob):
                        step, summary,validation_loss=session.run([self.global_step,merged,self.loss], feed_dict=val_feeds)
                        print(f"loss{validation_loss} ")
                        val_writer.add_summary(summary,step)
                        validation_loss_sum+=validation_loss
                    num_val_batches=len(validation_data)/batch_size
                    validation_epoch_norm/=num_val_batches
                    validation_epoch_cost/=num_val_batches
                    validation_epoch_nn/=num_val_batches
                    epoch_validation_loss=validation_loss_sum/num_val_batches
                    epoch_errors_data["Epoch"] = epoch_num
                    epoch_errors_data["Validation loss"]=epoch_validation_loss
                    epoch_errors_data["Validation norm"]=validation_epoch_norm
                    epoch_errors_data["Validation cost"]=validation_epoch_cost
                    epoch_errors_data["Validation normalized distance"]=validation_epoch_nn
                    if i % output_period == 0 or i == num_epochs - 1:
                        if output_training:
                            print("On validation set:")
                            print("Epoch number:", epoch_num)
                            print("Epoch loss:",epoch_validation_loss)
                            print("Epoch normalized distance:",validation_epoch_nn)
                            
                train_epoch_loss /= num_batches
                elapsed_time=time.time()-start_time
                if i%output_period==0 or i==num_epochs-1:
                    if output_training:
                        print("On training set:")
                        print("Epoch number:",epoch_num)
                        print(f"Epoch loss:{train_epoch_loss}")
                        print(f"Time:{elapsed_time}")
                    epoch_errors_data["Time"]=elapsed_time
                    epoch_errors_data["Train loss"]=train_epoch_loss
                    epoch_errors.append(epoch_errors_data)
        except KeyboardInterrupt:#Ctrl-c
            print("Keyboard interrupt")
        except Exception as exc:
            PrintException()
        finally:
            end_datetime=datetime.datetime.now()
            print(f"Output directory:{model_dir_path}")
            print("Saving training config...")
            config_file_path=os.path.join(model_dir_path,'config.txt')
            if not os.path.exists(model_dir_path):
                os.makedirs(model_dir_path)#Создать папку модели, если она не существует

            with open(config_file_path,'w+') as config_file:
                config_file.write(f"Training started at {start_datetime.strftime('%H:%M:%S on %B %d, %Y')}\n")
                config_file.write(f"Training finished at {end_datetime.strftime('%H:%M:%S on %B %d, %Y')}\n")
                config_file.write(f"Number of training samples:{num_words}\n")
                config_file.write(f"Total number of points:{total_points_num}\n")
                config_file.write("--Neural net configuration--\n")
                config_file.write(f"Number of layers:{self.num_layers}\n")
                config_file.write(f"Number of units:{self.num_units}\n")
                config_file.write(f"Learning rate{self.learning_rate}\n")
                print("Training config saved")
            print("Saving network config...")
            net_config_file_path = os.path.join(model_dir_path, 'net_config.json')
            self.save_net_config(net_config_file_path)
            print("network config saved...")
            print("Saving training results...")
            for train_epoch_cost in epoch_errors:
                for k,v in train_epoch_cost.items():
                    train_epoch_cost[k]=str(v).replace('.',',')#Заменить на запятую, чтобы excel понимал
            #Вывод в csv-файл
            self._csv_path=os.path.join(model_dir_path,f"{model_name}.csv")
            with open(self._csv_path, 'w+',newline='') as csvfile:
                sep=getListSeparator()#Получить разделитель для текущих настроек локали
                headers=[]
                if len(epoch_errors)>0:
                    headers=list(epoch_errors[0].keys())
                csvwriter=csv.DictWriter(csvfile,fieldnames=headers,delimiter=sep)
                csvwriter.writeheader()
                for i in np.arange(len(epoch_errors)):
                    data=epoch_errors[i]
                    csvwriter.writerow(data)#Записать строку в csv
            self._checkpoint_path=os.path.join(model_dir_path, model_name+'.ckpt')#Путь к сохраненному графу
            saver = tf.train.Saver(save_relative_paths=True)
            saver.save(session,self._checkpoint_path)
            print("Freezing graph...")
            GraphHelper.freeze_graph(model_dir_path,'probs')
            print("Graph freezing finished")


            train_writer.add_graph(session.graph)
            train_writer.flush()
            train_writer.close()
            print("Training result saved")
            session.close()
        return epoch_errors

    def lstm_cell(self):
        cell = tf.contrib.rnn.LSTMCell(self.num_units, state_is_tuple=True)
        return cell


    def create_multi_bilstm(self,num_layers,inputs):
        """
        Создает двунаправленную LSTM-сеть
        :param num_layers:
        :param inputs:
        :return:
        """
        cells_fw=[]
        cells_bw=[]
        for n in range(num_layers):
            cell_fw=self.lstm_cell()
            cell_bw=self.lstm_cell()
            cells_fw.append(cell_fw)
            cells_bw.append(cell_bw)
        return tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=cells_fw,cells_bw=cells_bw,inputs=inputs, dtype=tf.float32)


    def create_network(self, num_units, num_layers, num_features, num_outputs, learning_rate, batch_size=10,loss_kind=Loss.Sequence):
        self.num_units = num_units
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.inputs = tf.placeholder(tf.float32, [None, None, num_features], name='inputs')#[batch_size,max_time,num_features]
        self.keep_prob=tf.placeholder(tf.float32,name='keep_prob')#Вероятность, что выходной нейрон остается
        self.num_outputs = num_outputs
        self.num_features=num_features
        self.batch_size = batch_size
        self.loss_kind=loss_kind
        #self.batch_size=tf.placeholder(batch_size,trainable=False,name='batch_size')
        self.global_step=tf.Variable(0,trainable=False,name='global_step')
        #shape = tf.shape(self.inputs)
        if loss_kind==Loss.Sequence:
            self.targets=tf.placeholder(tf.int32,shape=[None,None],name='targets')#[num_batches,max_time]
        else:
            self.targets=tf.placeholder(tf.float32,shape=[None,None,self.num_outputs])#[num_batches,max_time,num_outputs]
        self.cells_stack = tf.contrib.rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.num_layers)],
                                                       state_is_tuple=True)

        self.W = tf.Variable(
            tf.truncated_normal([self.num_units * 2, self.num_outputs], stddev=0.1),name='W')  # Начальная матрица весов, домножается на 2, т.к. сеть двунаправленная
        #Нужно добавить одно измерение скопировать веса, чтобы умножать элементы из каждого батча на матрицу весов
        self.W=tf.expand_dims(self.W,axis=0)#Добавить измерение
        self.multiples=tf.placeholder(dtype=tf.int32)#Список для копирования весов, его элемент--это количественный множитель
        self.W=tf.tile(self.W,self.multiples)#Теперь должны быть копии весов для каждого батча
        self.weights_hist=tf.summary.histogram("weights",self.W)
        self.b = tf.Variable(tf.constant(0., shape=[self.num_outputs]), name='b')
        self.bias_hist=tf.summary.histogram("biases",self.b)
        # 1d array of size [batch_size]
        self.seq_len = tf.placeholder(tf.int32, [None], name='seq_len')
        # rnn_outputs--Тензор размерности [batch_size,max_time,cell.output_size],max_time--кол-во точек слова,cell.output_size=2(x,y координаты)
        # rnn_state--последнее состояние
        cells_fw = []
        cells_bw = []
        with tf.name_scope("nn_struct"):
            for n in range(num_layers):
                cell_fw = self.lstm_cell()
                cell_bw = self.lstm_cell()
                cells_fw.append(cell_fw)
                cells_bw.append(cell_bw)
            self.rnn_outputs, _,_=tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=cells_fw, cells_bw=cells_bw, inputs=self.inputs,
                                                                                                                 sequence_length=self.seq_len,
                                                                  dtype=tf.float32)
            #rnn_outputs[batch_size,max_time,layers_output]
            # layers_output are depth-concatenated forward and backward outputs
            #Значит, layers output будет равно num_units*2
            # Reshaping to apply the same weights over the timesteps
            #self.rnn_outputs=self.rnn_outputs[-1]#Берем вывод последнего слоя
            #rnn_outputs:[batch_size, max_time, layers_output]
            #self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.num_units*2])

        self.logits = tf.matmul(self.rnn_outputs, self.W) + self.b

        # self.logits=tf.cond(tf.equal(self.dropout_enabled,tf.constant(True)),lambda: tf.nn.dropout(self.logits,self.keep_prob),
        #                     lambda :self.logits)#Проверяем, задан ли дропаут
        self.logits=tf.nn.dropout(self.logits, self.keep_prob)
        # Reshaping back to the original shape
        #self.logits = tf.reshape(self.logits, [None, -1, self.num_outputs])
        # Time major

        self.probs=tf.nn.softmax(self.logits,name='probs')#вероятность для [batch_num,t,class_num]
        self.probs_hist=tf.summary.histogram('probs',self.probs)
        self.entropy_weights=tf.placeholder(tf.float32, [None, None], 'entropy_weights')#[batch_size, sequence_length]
        with tf.name_scope("loss"):
            if loss_kind==Loss.Sequence:
                self.loss=tf.contrib.seq2seq.sequence_loss(self.logits, self.targets, self.entropy_weights)
            else:
                self.xent=tf.nn.sigmoid_cross_entropy_with_logits(labels=self.targets, logits=self.logits)
                self.loss=tf.reduce_mean(self.xent)
            if self.loss_kind is not Loss.Sequence:
                tf.summary.histogram('xent',self.xent)
            tf.summary.scalar('loss',self.loss)
        with tf.name_scope("train"):
            self.train_fn = tf.train.MomentumOptimizer(self.learning_rate,
                                                   0.99).minimize(self.loss,global_step=self.global_step)
        self.ler=-1
        pass
