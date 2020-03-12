import os
import sys
import numpy as np
import pandas as pd

def read_csv(path):
    return pd.read_csv(path, sep = '\t') 


def split_train_test(dataset):
    train, trainy = dataset[:700000,:2], dataset[:700000,2] 
    val, valy = dataset[700000:850000,:2],dataset[700000:850000,2]
    test, testy = dataset[850000:,:2], dataset[850000:,2]
    print(train.shape, trainy.shape, val.shape, valy.shape, test.shape, testy.shape)
    return train, train_y, val, val_y, test, test_y


#Creating similar and dissimilar datasets for Siamese Model
def createDataset(clientID2Index):
  similar_dataset = []
  for i in clientID2Index:
    for j in clientID2Index[i]:
      for k in clientID2Index[i]:
        inp1, inp2 = j,k
        similar_dataset.append((inp1, inp2, 1))
  
  dissimilar_dataset = []
  for i in clientID2Index:
    other_speakers = [x for x in clientID2Index.keys() if i!=x]
    for j in other_speakers:
      for x in clientID2Index[i]:
        for y in clientID2Index[j]:
          inp1, inp2 = x,y
          dissimilar_dataset.append((inp1, inp2, 0))
  #Making both the labels balanced
  ind = [i for i in range(len(dissimilar_dataset))]
  if len(similar_dataset) < len(dissimilar_dataset):
    indexes_to_have = random.sample(ind, len(similar_dataset))
    dissimilar_dataset = [dissimilar_dataset[i] for i in indexes_to_have]
  #Saving
  dataset = np.array(similar_dataset + dissimilar_dataset)
  #Shuffline dataset
  np.random.shuffle(dataset)
  print(dataset.shape)
  np.save('SiameseDataset.npy', dataset)


if __name__ == '__main__':
    #Path to dataset tsv file
    data_file_path = sys.argv[1]
    train = pd.read_csv('drive/My Drive/train.tsv', sep='\t')
    
    #All speakers in my data
    num_speakers = len(set(train['client_id']))
    spk = list(set(train['client_id']))
    mapped = {spk[i]:i for i in range(num_speakers)}
    train = train.replace(mapped)

    #The following dictionary stores a mapping of speakerID to indexes of examples
    clientID2Index = {i:[] for i in range(28)}

    for i, val in enumerate(train['client_id']):
        clientID2Index[val].append(i)

    createDataset(clientID2Index)
    print("Successfully created one_shot_dataset...")
    