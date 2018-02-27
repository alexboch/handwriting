import tensorflow as tf
import pandas as pd
import numpy as np
import os
    
def load_lds(filename):
    raw_data=pd.read_json(filename,encoding='utf8')
    transposed_data=raw_data.T
    labeled_words=[]
    
    for text in transposed_data.iterrows():
        pl=text[1][['PointLists','Labels']]
    #for text_id in transposed_data.Id:
    #    mask=transposed_data.Id==text_id#Получить текст
    
    #for text in transposed_data.iterrows():# проход по текстам
      #  pl=text[['PointLists','Labels']]
        
    
        #lists_pairs=zip(pl.PointLists,pl.Labels)
        #Удалить те пары [точка,метка], где метка пустая
        
        #for i in np.range(len(text.TextWords)):#проход по индексам слов
            #Выбрать метки и точки с таким id
            #text
    return transposed_data

#Загрузить размеченные тексты из папки
#def load_labeled_texts(directory):
#    jsd_files=[]#Список имен файлов
#    for file in os.listdir(directory):
#        if file.endswith(".lds"):
#            jsd_files.append(file)

data=load_lds('Data//labeledTexts.lds')           