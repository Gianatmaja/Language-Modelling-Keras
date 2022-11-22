from keras.models import Sequential
from keras import layers
from keras.utils import np_utils
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


class LanguageModel:
    def __init__(self, ngram_size, tokenizer, dataset, D, MLP_units):
        self.Total_words = (dataset.X.shape[0])*(dataset.X.shape[1])
        self.input_dim = len(set(np.reshape(dataset.X, self.Total_words))) + 2
        self.output_dim = D
        self.input_len = ngram_size - 1
        self.ngram_size = ngram_size
        self.tokenizer = tokenizer
        self.Output = len(set(np.reshape(dataset.X, self.Total_words)))
        
        self.model = Sequential()

        self.model.add(layers.Embedding(input_dim = self.input_dim, output_dim = self.output_dim, input_length = self.input_len))
        self.model.add(layers.GlobalMaxPooling1D())
        self.model.add(layers.Dense(MLP_units, activation='relu'))
        self.model.add(layers.Dense(MLP_units, activation='relu'))
        self.model.add(layers.Dense(self.Output, activation='softmax'))
        
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())


    def train(self, train_X, prod_dim, epochs=10, batch_size=8):

        X_train = train_X[:,:2]
        y_train = [vec[2] for vec in train_X]
        All_words = [word for word in set(np.reshape(train_X, prod_dim))]
        
        Enc = OneHotEncoder(handle_unknown = 'ignore')
        Enc.fit(np.reshape(All_words, (-1,1)))
        y_tr = Enc.transform(np.reshape(y_train, (-1,1))).toarray()
        
        history = self.model.fit(X_train, y_tr, epochs = epochs)
        
        return history


    def predict(self, context):

        logits = self.model.predict(context)
        pred_index = np.argmax(logits) + 2
        word = self.tokenizer.index_word[pred_index]

        return pred_index, logits, word
    

    def generate(self, context, max_num_words=20):
        output = []
        Word = context[0]

        output, _, w = self.predict(context)
        
        return w
    
    def sent_log_likelihood(self, ngrams):
        logprob = 0

        for context_vec in ngrams:
            a = context_vec[0]
            b = context_vec[1]
            c = context_vec[2]
            
            ind, probs, _ = self.predict([context_vec[:2]])
            sol = np.log(probs[0][c-2])
            logprob = logprob + sol
            
        return logprob


    def fill_in(self, prefix, suffix, get_ngrams_fn):

        logits = []
        pred_word_id = 0
        return pred_word_id, logits

    def get_word_embedding(self, word):
        return self.model.layers[0].get_weights()[0][self.tokenizer.word_index[word]]