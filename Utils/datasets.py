import os
import sys
import numpy as np
import pandas as pd

def read_csv(path):
    return pd.read_csv(path, sep = '\t') 

def create_one_shot_data(path):
    
    train = read_csv(path)
    print("Reading training file....creating one_shot datasets..")
    #All speakers in my train data
    num_speakers = len(set(train['client_id']))
    spk = list(set(train['client_id']))
    mapped = {spk[i]:i for i in range(num_speakers)}
    train = train.replace(mapped)

    #The following dictionary stores a mapping of speakerID to indexes of examples
    clientID2Index = {}
    for i, r in train.iterrows():
        spk = r['client_id']
        val = r['path']
        if spk not in clientID2Index:
            clientID2Index[spk] = [val]
        else:  
            clientID2Index[spk].append(val)
    
    #Creating list of similar and different speaker pairs
    similar_dataset = []
    for i in clientID2Index:
        for j in clientID2Index[i]:
            for k in clientID2Index[i]:
                inp1, inp2 = j,k
                similar_dataset.append([inp1, inp2, 1])
    
    dissimilar_dataset = []
    for i in clientID2Index:
        other_speakers = [x for x in clientID2Index.keys() if i!=x]
        for j in other_speakers:
            for x in clientID2Index[i]:
                for y in clientID2Index[j]:
                    inp1, inp2 = x,y
                    dissimilar_dataset.append([inp1, inp2, -1])

    one_shot_data = pd.DataFrame(dissimilar_dataset, columns = ['inp1', 'inp2', 'label'])
    similar = pd.DataFrame(similar_dataset, columns = ['inp1', 'inp2', 'label'])
    one_shot_data = pd.concat([one_shot_data, similar])
    #shuffling the data
    one_shot_data = one_shot_data.sample(frac = 1)
    one_shot_data.to_csv('C:/Users/Anurag Kumar/Documents/GitHub/Projects/Speaker Identification/data/features/train.csv')
    print("Successfully saved one_shot dataset")
    '''
    test_data = [['common_voice_es_18736428','common_voice_es_18736428',1],['common_voice_es_18736428','common_voice_es_18736428',1],['common_voice_es_18736428','common_voice_es_18736428',1]]  
    test_df = pd.DataFrame(test_data, columns = ['inp1', 'inp2', 'label'])
    test_df.to_csv('C:/Users/Anurag Kumar/Documents/GitHub/Projects/Speaker Identification/data/features/train.csv')
    '''