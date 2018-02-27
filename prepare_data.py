import tensorflow as tf
import pandas as pd
import numpy as np
import os


def label_to_int(char_label):
    return ord(char_label)-1072#Код буквы А=1072

def load_lds(filename):
    raw_data=pd.read_json(filename,encoding='utf8')
    transposed_data=raw_data.T
    labeled_words=[]
    #Создать ключи словаря из слов всех текстов
    words_dict=dict.fromkeys(np.hstack(transposed_data['TextWords'].ravel()))
    for text in transposed_data.iterrows():
        pl=text[1][['PointLists','Labels']]
        #words_dict=dict.fromkeys(text_words)
        grouped_words_data=[]
        #flattened_points=[]
        #flattened_labels=[]
        for i in np.arange(len(pl.PointLists)):
            points_list=pl.PointLists[i]
            labels_list=pl.Labels[i]
            for j in np.arange(len(points_list)):
                point=points_list[j]
                label=labels_list[j]
                if label is not None:#Если не нулевая метка
                    integer_label=label_to_int(label['Item1'])#Букву в число
                
            #flattened_points.append(points_list)
            #flattened_labels.append(labels_list)
            
        
            #for j in np.arange(len(points_list)):
             #   flattened_points.append(poin)
        
        
    #lists_pairs=zip(pl.PointLists,pl.Labels)#пары вида [([список точек],[список меток])]
    #print(list(lists_pairs))
    #Преобразовать пары в пары "Точка-метка"
    
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