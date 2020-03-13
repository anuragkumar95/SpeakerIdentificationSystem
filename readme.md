An end to end deep learning model for speaker identification. The model exploits the idea of One-shot learning.

# Requirements
This project assumes that you have the following python modules installed:
- Tensorflow V1.
- Pandas
- pysptk
- hickle

# Training
## Creating features
This model uses MFCCs, their deltas and double deltas as input features. To create these features, run the command
```
python Utils/features.py "path_to_audio_clips" "path_to_train_tsv_file"
```
The above command will create the features and store in path ```features/data/```. 

## Creating the One Shot Dataset
Run the command ```python Utils/datasets.py "path_to_train_tsv_file"```. This will create ```SiameseDataset.npy``` at path ```Utils/```.

## Training the model
Run the command ```python spk_identification.py "path_to_train_tsv_file" "SiameseDataset.npy"```.

## Evaluating the model
Run the command ```python Evaluation.py "path_to_train_tsv_file"```.





