# Language Modelling in Keras
This project involves language modelling in Keras, including training models to predict the next word in a sentence, computing sentence likelihood as well as similarity between words.

### Project Structure
This repository has the following structure

    .
    ├── models                            
    │  ├── __init__.py
    │  ├── ngram_nlm.py                           # language model algorithm
    ├── readers                 
    │  ├── _init_.py
    │  ├── ngram_dataset.py                       # Load dataset
    ├── _init_.py
    ├── lm.py                                     # Performs the language modelling
    ├── images
    └── README.md

The main python file is lm.py, where the bulk of the code is implemented. There, we can load the dataset, train language models as well as implement all 
the necessary functions to support the task. In ngram_dataset.py, we will implement functions that will load and preprocess the dataset, whereas in 
ngram_nlm.py, we will train the language model and implement functions to generate words (using a tri-gram neural language model, as shown below), compute
sentence likelihood as well as the consine similarity between 2 words using their word embeddings.

![model](https://github.com/Gianatmaja/Language-Modelling-Keras/blob/main/images/Architecture.png)

### Snapshots of Training Process
![losses](https://github.com/Gianatmaja/Language-Modelling-Keras/blob/main/images/Res1.png)
