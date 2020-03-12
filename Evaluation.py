import os
import sys
import random
import hickle as h
import pickle as p
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json
from Utils.datasets import read_csv, create_one_shot_data, split_train_test
from Utils.features import create_features

def evaluation(testdata, testy, train_files_df, batchsize, num_epochs, save_path):
    print("Running tests....")
    
    #Loading Siamese Model
    json_file = open(save_path + 'Smodel'+str(num_epochs - 1)+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    SiameseModel = model_from_json(loaded_model_json)
    # load weights into new model
    SiameseModel.load_weights(save_path + 'Smodel'+str(num_epochs - 1)+'.h5')
    
    test_batches = test.shape[0]//batchsize
    #test_batches = 2
    test_a = 0
    preds = []
    for x in range(test_batches):
        start = x*batchsize
        end = min(start + batchsize, test.shape[0])
        testbatch_x = test[start:end]
        testbatchy = testy[start:end]
        testbatch_data1, testbatch_data2 = data_transform(testbatch_x, train_files_df, batchsize)
        pred = np.argmax(SiameseModel.predict([testbatch_data1, testbatch_data2]), axis = 1)
        print([(i, j) for i, j in zip(pred, testy[start:end])])
        preds.extend(pred) 
    acc = sum([1 if i == j else 0 for i, j in zip(preds, testy[:end])])/end
    print("TEST accuracy:", acc)

if __name__ = '__main__':
    train_path = sys.argv[1]
    train_file = pd.read_csv(train_path, sep = '\t')
    data = np.load('Utils/SiameseDataset.npy')
    _,_,_,_,test, testy = split_train_test(dataset)
    evaluation(test, testy, train_file, batchsize = 64, num_epochs, 'features/data/models/')
