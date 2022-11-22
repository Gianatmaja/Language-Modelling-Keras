
from readers.ngram_dataset import NGramDataset
from models.ngram_nlm import LanguageModel
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def cosine_similarity(vector_A, vector_B):
    num = np.dot(vector_A, vector_B)
    den = (np.sqrt(np.sum(np.square(vector_A))))*(np.sqrt(np.sum(np.square(vector_B))))
    sol = num/den
    
    return sol

def plot_history(history):
    loss = history.history['loss']
    x = range(1, len(loss) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(x, loss, 'b', label='Training loss')
    plt.title('Training loss')
    plt.legend()
    plt.show()


def plot_history2(history):
    acc = history.history['accuracy']
    loss = history.history['loss']
    x = range(1, len(loss) + 1)
    
    plt.figure(figsize = (14,5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, loss, 'r', label='Training loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(x, acc, 'b', label='Training accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    
    plt.show()


def exercises_sonnet():
    # Fitting a 3-gram Language Model on the Sonnet dataset and plotting the training loss histogram.

    # Init hyperparameters
    corpus = "sonnet"
    ngram_size = 3
    epochs = 300
    batch_size = 8

    # Create dataset of ngrams
    dataset = NGramDataset(corpus=corpus, ngram_size=ngram_size) 
    # Create Language Model (LM)
    lm = LanguageModel(ngram_size, dataset.tokenizer, dataset, D = 30, MLP_units = 15)
    # Train LM
    history = lm.train(dataset.X, epochs = epochs, batch_size = batch_size, prod_dim = 357)
    # Plot training curve
    plot_history(history)
    plot_history2(history)

    
    # Predicting the next word and probability score given the bigram 'all the'.
    
    context = ['all the']
    context_vec = dataset.vectorize(context)
    pred_index, logits, word = lm.predict(context_vec)
    print('Prediction probabilities: ', logits)
    print('Next word index predicted: ', pred_index)
    print('Word predicted: {} with probability {}'.format(word, logits[0][pred_index-2]))


    # Generating some random text (20 words), starting with '<sos> <sos>'. 

    
    context = ['<sos> <sos>']
    Words = ['<sos>', '<sos>']
    context_vec = dataset.vectorize(context)
    Ind = []
    Con = context_vec
    for i in range(20):
        w = lm.generate([Con[0][-2:]])
        Words.append(w)
        W_ = [" ".join(word for word in Words)]
        Con = dataset.vectorize(W_)
    print(Con)
    text = W_
    print(text)


def exercises_toy():
    # Fiting a 3-gram Language Model on the Toy Dataset and plotting the training loss histogram.
    corpus = "toy"
    ngram_size = 3

    dataset = NGramDataset(corpus=corpus, ngram_size=ngram_size) 
 
    epochs = 500
    batch_size = 8
    
    lm = LanguageModel(ngram_size, dataset.tokenizer, dataset, D = 50, MLP_units = 30)
    history = lm.train(dataset.X, epochs = epochs, batch_size = batch_size, prod_dim = 105)
    plot_history(history)
    plot_history2(history)
    
    # Predicting the next word given the bigram '<sos> the'.

    
    context = ['<sos> the']
    context_vec = dataset.vectorize(context)
    pred_index, logits, word = lm.predict(context_vec)
    print('Prediction probabilities: ', logits)
    print('Next word index predicted: ', pred_index)
    print('Word predicted: {} with probability {}'.format(word, logits[0][pred_index-2]))
    
    Crook = ['crook']
    print('Crook is item no.{} in probabilities vector.'.format(dataset.vectorize(Crook)[0][0] - 1))
    print('\n')
   
    
    # Trial: Which of the two sentences S1: 'The thief stole the suitcase.' and S2: 'The crook stole the suitcase.'
    # is more likely? 

    sentence_1 = 'The thief stole the suitcase.'
    sentence_2 = 'The crook stole the suitcase.'
    sent_1_ngrams = dataset.get_ngrams(dataset.vectorize([dataset.preprocess(sentence_1)])[0])
    sent_2_ngrams = dataset.get_ngrams(dataset.vectorize([dataset.preprocess(sentence_2)])[0])
    
    
    log_likelihood_1 = lm.sent_log_likelihood(sent_1_ngrams)
    log_likelihood_2 = lm.sent_log_likelihood(sent_2_ngrams)
    print("Log[P(S1)] = ", log_likelihood_1)
    print("Log[P(S2)] = ", log_likelihood_2)
    print('P(S1) = ', np.exp(log_likelihood_1))
    print('P(S2) = ', np.exp(log_likelihood_2))
    #print('Log-likelihood of first sentence: ', np.log(log_likelihood_1))
    #print('Log-likelihood of second sentence: ', np.log(log_likelihood_2))
    #print('Trigram probabilities of first sentence: ', Probas1)
    #print('Trigram probabilities of second sentence: ', Probas2)

    # Training Word Embeddings.
    WE_thief = lm.get_word_embedding('thief')
    WE_crook = lm.get_word_embedding('crook')
    WE_cop = lm.get_word_embedding('cop')
    
    
    # Implementing cosine similarity
    thief_crook_cs = cosine_similarity(WE_thief, WE_crook)
    thief_cop_cs = cosine_similarity(WE_thief, WE_cop)
    crook_cop_cs = cosine_similarity(WE_crook, WE_cop)
    
    print('\n')
    print('Word embedding for thief: ', WE_thief)
    print('Word embedding for crook: ', WE_crook)
    print('Word embedding for cop: ', WE_cop)
    print('\n')
    print('Cosine similarity for thief and crook: ', thief_crook_cs)
    print('Cosine similarity for thief and cop: ', thief_cop_cs)
    print('Cosine similarity for crook and cop: ', crook_cop_cs)



def main():

    exercises_sonnet() 

    exercises_toy()


if __name__ == "__main__":
    main()