#!/usr/bin/python
#coding:utf-8

"""
@author: wuxikun
@contact: wuxikun@bjgoodwill.com
@software: PyCharm Community Edition
@file: Main.py
@time: 12/12/18 6:28 PM
"""

from keras.layers import LSTM, Dense, Dropout, Activation, Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from keras.models import load_model
from keras.models import model_from_json
import pickle
import os

max_len = 20

def loadData(path):
    words = set()
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            words |= (set(str(line.strip())))
    return words


def print_weight(model):
    for layer in model.layers:
        print(layer.get_weights())


def vector_sentence(sentence, wordIndices):
    return [word_indices[v] for v in list(sentence.strip()) if v in wordIndices]


def custom_test():
    cur_model = load_model(model_path)
    run_model_on_custom_test_data(cur_model)


def run_model_on_custom_test_data(model):

    pred_list = ['聚乙二醇电解质散剂',
                '重组人碱性成纤维细胞生长因子凝胶',
                '硝酸甘油贴片',
                '硝酸甘油贴片',
                '硝酸甘油贴片',
                 '养血饮口服液',
                 '人参健脾丸',
                 '秋泻灵合剂',
                 '全鹿丸',
                 '归脾合剂',
                 '归芍调经胶囊'
                 ]

    pred_data = [vector_sentence(x, word_indices) for x in pred_list]
    pred_data = pad_sequences(pred_data, maxlen=max_len)
    y = model.predict(pred_data)
    print([0 if x <= 0.5 else 1 for x in y])

if __name__ == '__main__':

    model_path = 'data/medicine_category.h5'
    weight_path = 'data/medicine_category_weight.h5'
    west_path = 'data/west.txt'
    east_path = 'data/east.txt'
    w2id_path = 'data/word2id.pkl'

    if not os.path.exists(w2id_path):
        west = loadData(west_path)
        east = loadData(east_path)
        words = list(west | east)
        word_indices = {v: i for i, v in enumerate(words)}
        pickle.dump(word_indices, open(w2id_path, 'wb+'))
    else:
        word_indices = pickle.load(open(w2id_path, 'rb'))

    if True:
        custom_test()
    else:
        # load data
        west_src = open(west_path).readlines()
        east_src = open(east_path).readlines()
        all_data = [vector_sentence(line, word_indices) for line in west_src] + [vector_sentence(line, word_indices) for line in east_src]
        # 西医 0 中医　1
        all_pred = [0 for i in range(len(west_src))] + [1 for i in range(len(east_src))]

        # 手动打乱数据
        all_data = np.array(all_data)
        all_pred = np.array(all_pred)
        state = np.random.get_state()
        np.random.shuffle(all_data)
        np.random.set_state(state)
        np.random.shuffle(all_pred)

        all_data = pad_sequences(all_data, maxlen=max_len)
        train_size = int(len(all_data) * 4/5)
        train_data = all_data[: train_size]
        train_pred = all_pred[: train_size]
        test_data = all_data[train_size:]
        test_pred = all_pred[train_size:]

        model = Sequential()
        model.add(Embedding(len(train_data), 128, name='embedding'))
        model.add(LSTM(128, name='lstm'))
        model.add(Dense(1, activation='sigmoid', name='dense'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.fit(train_data, train_pred, batch_size=128, epochs=2)
        score = model.evaluate(test_data, test_pred, batch_size=128)
        print(score)
        model.save(model_path)
        with open('trian_model.json', 'w') as f:
            f.write(model.to_json())
        run_model_on_custom_test_data(model)
        test_mdel = load_model(model_path)
        run_model_on_custom_test_data(test_mdel)









