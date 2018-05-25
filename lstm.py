import tensorflow as tf
import csv
import time
from utils import *

map_fn = tf.map_fn


class ValFeed:
    def __init__self(self,points,labels,length,weights):

        pass

class LSTMDecoder:
    """
    Класс для создания, обучения и получения разметки от LSTM-нейросети TODO:Сделать другие варианты клеток, кроме LSTM
    """

    def __init__(self, num_units, num_layers, num_features, num_classes, learning_rate, batch_size,alphabet):
        """
        Конструктор, в нем задаются размеры слоев и создается клетка сети
        """
        self.create_network(num_units, num_layers, num_features,
                            num_classes, learning_rate, batch_size)
        self.alphabet=alphabet
        pass

    def get_probabilities(self,points,model_name,model_dir):
        """
        :param points:Список входных точек
        :param model_name:
        :param model_dir:
        :return:Вероятности каждого класса для каждой точки каждого списка
        """
        model_name = os.path.join(model_dir, model_name + ".meta")
        with tf.Session() as session:
            saver = tf.train.import_meta_graph(model_name)
            saver.restore(session, tf.train.latest_checkpoint(model_dir))  # Загрузить натренированную модель
            probs_list = []
            for points_list in points:
                input_arr = np.asarray(points_list)
                input_arr = np.expand_dims(input_arr, 0)
                probs = session.run([self.probs],
                                                       feed_dict={self.inputs: input_arr,
                                                                  self.seq_len: [len(points_list)]})
                probs_list.append(probs[0])
        return probs_list

    TINY = 1e-6  # to avoid NaNs in logs


    def train(self, words, num_epochs=100, output_training=False, model_name="model",model_dir_path=f"Models{os.sep}model",validate=True):
        """
        words--Список слов, содержащих точки и метки
        """
        os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
        words=words[0:2]
        print("starting training,epochs:",num_epochs,"learning rate:",self.learning_rate)
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        num_batches = len(words) / self.batch_size
        num_words = len(words)
        #output_period=max(num_epochs/100.0*2,1)#Выводить каждый раз, когда прошло 2% обучения
        output_period=1
        epoch_errors=[]
        start_time=time.time()

        weighted_words=list(words)
        for word in weighted_words:
            word.length = len(word.point_list)
            labels_arr=np.asarray(word.labels_list)
            num_for_classes=np.zeros(self.num_classes)
            for nc in range(self.num_classes):#Посчитать, сколько объектов каждого класса входит
                num_for_classes[nc]=np.count_nonzero(labels_arr==nc)#Количество точек данного класса
            #Задать веса
            word.weights = np.ones((self.batch_size, len(word.point_list)), dtype=np.float32)
            class_weights=np.ones(self.num_classes)
            n0=num_for_classes[np.where(num_for_classes>0)][0]
            i=0
            for ni in num_for_classes:
                x=n0/ni
                class_weights[i]=x
                i+=1
            for j in range(self.num_classes):
                class_mask=(labels_arr==j)
                word.weights[0][class_mask]=class_weights[j]

        data_len=len(weighted_words)
        train_len=data_len#Если нет валидации, берем весь массив слов
        valid_len=0
        validate=validate and data_len>1#Если слово только 1, то валидацию следать не сможем
        if data_len<=1:
            print("Dataset is too small to partition to test and validation sets!")
        if validate:
            np.random.shuffle(weighted_words)
            train_len=int(data_len*0.8)
            validation_data=weighted_words[train_len:]
        training_words = weighted_words[:train_len]

        try:
            for i in np.arange(num_epochs):
                epoch_errors_data=dict()
                np.random.shuffle(words)#
                train_epoch_loss = 0  # Средняя ошибка по всем батчам в данной эпохе
                can_output=output_training and (i%output_period==0 or i==num_epochs-1)
                for j in np.arange(0, len(training_words)):  # Цикл по всем тренировочным словам, берем по 1 слову
                    batch_word = training_words[j]  # слова для создания батча
                    batch_inputs = [batch_word.point_list]
                    batch_labels = [batch_word.labels_list]
                    seq_lengths = [batch_word.length]  # Вектор длин каждой последовательности
                    inputs_arr = np.asarray(batch_inputs)
                    targets_array = np.asarray(batch_labels)
                    #TODO: переделать под переменную длину либо убрать лишнее и оставить размер батча 1
                    #Подаем батч на вход и получаем результат
                    loss,probs,_ = session.run([self.loss,self.probs,self.train_fn],
                                                                           feed_dict={self.inputs: inputs_arr, self.targets: targets_array,self.entropy_weights:batch_word.weights,
                                                                                      self.seq_len: seq_lengths})
                    train_epoch_loss+=loss
                    if can_output:
                        print("word loss:", loss, " Epoch:", i,"Word:",j)

                """Конец эпохи(Прошли весь тренировочный датасет)"""
                #Валидация
                if validate:
                    validation_epoch_norm=0#Среднее значение нормы на эпохе для датасета валидации
                    validation_epoch_nn=0
                    validation_epoch_cost=0
                    validation_loss_sum=0

                    for val_word in validation_data:#Для каждого слова валидации
                        val_inputs_arr = np.asarray([val_word.point_list])
                        val_targets_arr=np.asarray([val_word.labels_list])

                        validation_loss=session.run(self.loss,
                                                                                  feed_dict={self.inputs:val_inputs_arr,
                                                                                             self.targets:val_targets_arr,
                                                                                             self.seq_len:[val_word.length],
                                                                                             self.entropy_weights:val_word.weights})
                        validation_loss_sum+=validation_loss
                    val_feeds_len=len(validation_data)
                    validation_epoch_norm/=val_feeds_len
                    validation_epoch_cost/=val_feeds_len
                    validation_epoch_nn/=val_feeds_len
                    epoch_validation_loss=validation_loss_sum/val_feeds_len
                    epoch_errors_data["Epoch"] = i
                    epoch_errors_data["Validation loss"]=epoch_validation_loss
                    epoch_errors_data["Validation norm"]=validation_epoch_norm
                    epoch_errors_data["Validation cost"]=validation_epoch_cost
                    epoch_errors_data["Validation normalized distance"]=validation_epoch_nn
                    if i % output_period == 0 or i == num_epochs - 1:
                        if output_training:
                            print("On validation set:")
                            print("Epoch number:", i)
                            print("Epoch loss:",epoch_validation_loss)
                            print("Epoch normalized distance:",validation_epoch_nn)
                train_epoch_loss /= len(training_words)
                elapsed_time=time.time()-start_time
                if i%output_period==0 or i==num_epochs-1:
                    if output_training:
                        print("On training set:")
                        print("Epoch number:",i)
                        print(f"Epoch loss:{train_epoch_loss}")
                        print(f"Time:{elapsed_time}")
                    epoch_errors_data["Time"]=elapsed_time

                    epoch_errors_data["Train loss"]=train_epoch_loss
                    epoch_errors.append(epoch_errors_data)
        except KeyboardInterrupt:#Ctrl-c
            print("Keyboard interrupt")
        finally:
            print("Saving training results...")
            for train_epoch_cost in epoch_errors:
                for k,v in train_epoch_cost.items():
                    train_epoch_cost[k]=str(v).replace('.',',')#Заменить на запятую, чтобы excel понимал
            #Вывод в csv-файл
            self._csv_path=os.path.join(model_dir_path,f"{model_name}.csv")
            if not os.path.exists(model_dir_path):
                os.makedirs(model_dir_path)#Создать папку модели, если она не существует

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
            self._checkpoint_path=os.path.join(model_dir_path, model_name)#Путь к сохраненному графу
            saver = tf.train.Saver()
            saver.save(session,self._checkpoint_path)
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


    def create_network(self, num_units, num_layers, num_features, num_classes, learning_rate, batch_size=10):
        self.num_units = num_units
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.inputs = tf.placeholder(tf.float32, [None, None, num_features], name='inputs')
        self.num_classes = num_classes
        self.num_features=num_features
        self.batch_size = batch_size
        shape = tf.shape(self.inputs)
        self.targets=tf.placeholder(tf.int32,shape=[self.batch_size,None],name='targets')
        self.cells_stack = tf.contrib.rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.num_layers)],
                                                       state_is_tuple=True)

        self.W = tf.Variable(
            tf.truncated_normal([self.num_units*2, self.num_classes], stddev=0.1),name='W')  # Начальная матрица весов, домножается на 2, т.к. сеть двунаправленная
        self.b = tf.Variable(tf.constant(0., shape=[self.num_classes]),name='b')


        # 1d array of size [batch_size]
        self.seq_len = tf.placeholder(tf.int32, [None], name='seq_len')
        # rnn_outputs--Тензор размерности [batch_size,max_time,cell.output_size],max_time--кол-во точек слова,cell.output_size=2(x,y координаты)
        # rnn_state--последнее состояние
        cells_fw = []
        cells_bw = []
        for n in range(num_layers):
            cell_fw = self.lstm_cell()
            cell_bw = self.lstm_cell()
            cells_fw.append(cell_fw)
            cells_bw.append(cell_bw)
        self.rnn_outputs, self.rnn_state_fw,self.rnn_state_bw=tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=cells_fw, cells_bw=cells_bw, inputs=self.inputs,
                                                                                                             sequence_length=self.seq_len,
                                                              dtype=tf.float32)
        # Reshaping to apply the same weights over the timesteps
        self.rnn_outputs=self.rnn_outputs[-1]#Берем вывод последнего слоя
        self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, self.num_units*2])
        self.logits = tf.matmul(self.rnn_outputs, self.W) + self.b
        # Reshaping back to the original shape
        self.logits = tf.reshape(self.logits, [self.batch_size, -1, self.num_classes])
        # Time major
        self.probs=tf.nn.softmax(self.logits,name='probs')#вероятность для [batch_num,t,class_num]
        self.entropy_weights=tf.placeholder(tf.float32, [self.batch_size, None], 'entropy_weights')
        self.loss=tf.contrib.seq2seq.sequence_loss(self.logits, self.targets, self.entropy_weights)
        self.train_fn = tf.train.MomentumOptimizer(self.learning_rate,
                                                   0.99).minimize(self.loss)
        self.ler=-1
        pass
