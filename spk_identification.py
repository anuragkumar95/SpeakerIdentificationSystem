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


def make_feats(filename):
        filename = filename.split('.')[0] + ".npy"
        features_path = r'features/data/'
        mfcc = np.load(features_path + "MFCC/" + filename)
        delta = np.load(features_path + "DMFCC/" + filename)
        ddelta = np.load(features_path + "DDMFCC/" + filename)
        return np.row_stack((mfcc, delta, ddelta))

def data_transform(pairs, train, batchsize):    
    files = []
    for p in pairs:
        files.append([train.iloc[p[0]]['path'], train.iloc[p[1]]['path']])
    inp1 = np.asarray([make_feats(i[0]) for i in files])
    inp2 = np.asarray([make_feats(i[1]) for i in files])
    return (inp1, inp2)

def Encoder(input_shape, embedding_dimension, drop_out=0.05):
    #Input placeholders
    inp_placeholder = tf.keras.Input(shape=input_shape)

    #Convolution layer 1
    layer1 = layers.Conv2D(40, 25, activation='relu', name="L11", padding='same')(inp_placeholder)
    norm1 = layers.BatchNormalization()(layer1)
    drops1 = layers.SpatialDropout2D(drop_out)(norm1)
    out1 = layers.MaxPooling2D(5)(drops1)
    
    #Convolution layer 2
    layer2 = layers.Conv2D(30, 15, activation='relu', name="L21", padding='same')(out1)
    norm2 = layers.BatchNormalization()(layer2)
    drops2 = layers.SpatialDropout2D(drop_out)(norm2)
    out2 = layers.MaxPooling2D(4)(drops2)
    
    #Convolution layer 3
    layer3 = layers.Conv2D(10, 5, activation='relu', name="L31", padding='same')(out2)
    norm3 = layers.BatchNormalization()(layer3)
    drops3 = layers.SpatialDropout2D(drop_out)(norm3)
    out3 = layers.MaxPooling2D((5,1))(drops3)
    
    #Convolution layer 4
    layer4 = layers.Conv2D(15, 5, actop_out)(norm5), name="L41", padding='same')(out3)
    norm4 = layers.BatchNormalizationop_out)(norm5)
    drops4 = layers.SpatialDropout2D(op_out)(norm5)4)
    out4 = layers.MaxPooling2D(1)(droop_out)(norm5)

    #Convolution layer 5
    layer5 = layers.Conv2D(10, 5, actop_out)(norm5), name="L51", padding='same')(out4)
    norm5 = layers.BatchNormalizationop_out)(norm5)
    out5 = layers.SpatialDropout2D(drop_out)(norm5)

    #Flattened layer
    flatten = layers.Flatten()(out5)
   
    #Dense Layer
    embeds = layers.Dense(embedding_dimension, activation = "sigmoid", name="D11")(flatten)
    
    #Definining our model
    encoder = tf.keras.Model(inputs=inp_placeholder, outputs = embeds)
    return encoder

def SiameseNetwork(input_shape):
    inp_1 = tf.keras.Input(shape=input_shape)
    inp_2 = tf.keras.Input(shape=input_shape)
    
    encoder1 = Encoder(input_shape = (5388, 20, 3), embedding_dimension = 128)
    encoder2 = Encoder(input_shape = (5388, 20, 3), embedding_dimension = 128)

    #Encode each branch
    embeds1 = encoder1(inp_1)
    embeds2 = encoder2(inp_2)

    #Siamese network
    embedded_distance = layers.Subtract(name='subtract_embeddings')([embeds1, embeds2])
    embedded_distance1 = layers.Lambda(lambda x: tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)), name='euclidean_distance')(embedded_distance)
    siamese_out = layers.Dense(2, activation='sigmoid', name="OutputLayer")(embedded_distance1)

    #Model
    siamesemodel = tf.keras.Model(inputs=[inp_1,inp_2], outputs = siamese_out)
    return (siamesemodel, encoder1, encoder2)


def train_network(model, encoder1, encoder2, train, ytrain, val, yval, batchsize, num_epochs, train_files_df, model_save_path, lr=0.001):
    X1 = tf.keras.Input(shape=(5388, 20, 3))
    X2 = tf.keras.Input(shape=(5388, 20, 3))
    y = tf.placeholder('int32',[None],name='t1')
    
    siamese_out = model([X1, X2])

    #Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y,logits = siamese_out)
    loss = tf.reduce_mean(loss)
    
    #Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    soft_out = tf.nn.softmax(siamese_out)
    
    y_onehot = tf.one_hot(y, 2)
    
    #Prediction and accuracy
    pred = tf.equal(tf.argmax(soft_out,1), tf.argmax(y_onehot,1))
    accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))
    
    #Saving history
    hist = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        steps = train.shape[0]//batchsize
        #steps = 1000
        print("Steps:", steps)
        for epoch in range(num_epochs):
            epoch_loss1 = 0
            epoch_acc1 = 0
            epoch_loss2 = 0
            epoch_acc2 = 0
            for step in range(steps):
                start = step*batchsize
                end = min(start + batchsize, train.shape[0])
                batch_x = train[start:end]
                batchy = ytrain[start:end]
                batch_data1, batch_data2 = data_transform(batch_x, train_files_df, batchsize)
                _, cost = sess.run([optimizer, loss], feed_dict={X1:batch_data1, X2:batch_data2, y:batchy})
                train_acc = sess.run(accuracy, feed_dict = {X1:batch_data1, X2:batch_data2, y:batchy})
                epoch_loss1 += cost
                epoch_acc1 += train_acc             
                #Reverse
                _, cost = sess.run([optimizer, loss], feed_dict={X1:batch_data2, X2:batch_data1, y:batchy})
                train_acc = sess.run(accuracy, feed_dict = {X1:batch_data2, X2:batch_data1, y:batchy})
                epoch_loss2 += cost
                epoch_acc2 += train_acc
                if step%500 == 0:
                    print("Epoch:",epoch,"Step:",step+1,"TrainLoss:",(epoch_loss1/(step+1) + epoch_loss2/(step+1))/2, "accuracy:", (epoch_acc2/(step+1) + epoch_acc2/(step+1))/2)
                hist['train_loss'].append((epoch_loss1/(step+1) + epoch_loss2/(step+1))/2)
                hist['train_acc'].append((epoch_acc1/(step+1) + epoch_acc2/(step+1))/2)
                
            spkModel_json = model.to_json()
            ebdModel1_json = encoder1.to_json()
            ebdModel2_json = encoder2.to_json()
            with open(model_save_path + "Smodel" + str(epoch) + ".json", "w") as json_file:
                json_file.write(spkModel_json)
            
            with open(model_save_path + "Emodel1" + str(epoch) + ".json", "w") as json_file:
                json_file.write(ebdModel1_json)
            
            with open(model_save_path + "Emodel2" + str(epoch) + ".json", "w") as json_file:
                json_file.write(ebdModel2_json)
            
            model.save_weights(model_save_path + "Smodel"+str(epoch)+".h5")
            encoder1.save_weights(model_save_path + "Emodel1"+str(epoch)+".h5")
            encoder2.save_weights(model_save_path + "Emodel2"+str(epoch)+".h5")
            
            print("Saved model to disk")
            print("Validating results")
            #print("Epoch:",epoch,"Loss:", epoch_loss/steps, "accuracy:", epoch_acc/steps)
            val_batches = val.shape[0]//batchsize
            #val_batches = 2
            val_l = 0
            val_a = 0
            for x in range(val_batches):
                start = x*batchsize
                end = min(start + batchsize, val.shape[0])
                valbatch_x = val[start:end]
                valbatchy = yval[start:end]
                valbatch_data1, valbatch_data2 = data_transform(valbatch_x, train_files_df, batchsize)
                val_loss, val_acc = sess.run([loss,accuracy], feed_dict = {X1:valbatch_data1,
                                                                            X2:valbatch_data2,
                                                                            y:valbatchy})
                val_l += val_loss
                val_a += val_acc
                hist['val_loss'].append(val_loss)
                hist['val_acc'].append(val_acc)
            print("Validation:","Epoch:",epoch,"Loss:", val_l/val_batches, "accuracy:", val_a/val_batches)
          
    return (hist, model, encoder1, encoder2)

if __name__ == "__main__":
    #Path to audio files
    path_to_audio = sys.argv[1]

    #train tsv file
    train_file_path = sys.argv[2]

    #Create dataset for one-shot implementation
    create_one_shot_data(train_file_path)

    #loading features and splitting them into train, validation and test
    train, trainy, val, valy, test, testy = split_train_test()
    
    #Creating the model structure
    siamese, encoder1, encoder2 = SiameseNetwork(input_shape = (5388, 20, 3))

    ori_train_file = pd.read_csv(train_path, sep = '\t')

    #Training model
    model_hist, siamese, encoder1, encoder2 = train_network(model = siamese, encoder1 = encoder1, encoder2 = encoder2, train = train, trainy = trainy, val = val, valy = valy, train_files_df = ori_train_file, batchsize = 64, num_epochs = 1, model_save_path = "features/models/")
    

 