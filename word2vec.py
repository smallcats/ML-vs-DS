import re
import numpy as np
import tensorflow as tf
from os import chdir as cd
from os import listdir as ls
from matplotlib import pyplot as plt
from random import shuffle
from math import exp
from sklearn import svm

cd('C:\\Users\\Geoffrey\\Documents\\Job_descriptions')

#--------------------------Preprocessing------------------------------
def getFileNames(path):
    """
    Gets the .txt files in path
    args: path: string representing the relative path to the directory containing file names
    returns: filenames: a list of the .txt file names
    """
    filenames = []
    for name in ls(path):
        if name[-3:] == 'txt': filenames.append(path+'\\'+name)
    return filenames

def importText(filename):
    """
    Imports the text of a file to a list of words (including "." as a word)
    """
    with open(filename, 'r') as file:
        rawtext = file.read()
    return re.sub("\.", " . ",re.sub("([-\r\n,\(\)!;:\?/]|\ufeff)", " ", rawtext).lower()).split()

def wordsTo5Grams(wordlist):
    """
    Takes a list of words and returns 5-grams in the form (word1, word2, word4, word5, word3)
    """
    grams = []
    for k in range(len(wordlist)-4):
        grams.append(wordlist[k:k+2]+wordlist[k+3:k+5]+wordlist[k+2:k+3])
    return grams

def getGrams(filenames):
    """
    Gets 5-grams from files in string form.
    
    args: filenames: a list of file names
    
    returns: grams: a list of length 5 lists of strings that are the 5-grams from the filenames, with the 3rd word in the last slot
    """
    grams = []
    for filename in filenames:
        text = importText(filename)
        grams.extend(wordsTo5Grams(text))
    return grams

def getVocab(filenames):
    """
    Gets the vocab dict from a list of text files.

    args: filenames: a list of file names (they should all be text files)

    returns: vocab: a dict whose keys are vocab words and values are nonnegative integers.
                       vocab[word] = k means that word is represented by the kth standard basis vector
    """
    vocab_set = set()
    for filename in filenames:
        text = importText(filename)
        vocab_set.update(text)
    vocab = dict()
    vnum = 0
    for word in vocab_set:
        vocab[word] = vnum
        vnum += 1
    return vocab


def getBatch(batchsize, grams, vocab, vocabdim = 0):
    """
    Prepares a batch.
    args: batchsize: the number of samples in the batch
          gramstart: the start of the grams from which the batch is drawn (0 is the first gram)
          gramstop: the end of the grams from which the batch is drawn
          grams: the list of 5-grams
          vocab: the vocab dict
          vocabdim: the dimension of the (unencoded) word vectors, if vocabdim == 0, this will default to len(vocab)
    returns: batch_array: an array to be fed to the TensorFlow graph as the sample
             batch_labels: an array of labels to be fed to TensorFlow
    """
    if vocabdim == 0: vocabdim = len(vocab)
    batch_array = np.zeros(shape=(batchsize, vocabdim*4))
    batch_labels = np.zeros(shape=(batchsize, vocabdim))
    batch_choice = np.random.randint(0, len(grams), batchsize)
    for k, gramnum in enumerate(batch_choice):
        g = grams[gramnum]
        for l in range(4):
            batch_array[k][l*vocabdim + vocab[g[l]]] = 1.0
        batch_labels[k][vocab[g[4]]] = 1.0
    return batch_array, batch_labels

#---------------------------Build graphs------------------------------
def buildGuess(vocabsize, repdim):
    """
    Builds the tensorflow graph for the neural net on which CBOW Word2Vec will be trained. Guesses the middle word of a 5-gram.
      The structure has 1 hidden layer connected to the input as (direct sum of) Word2Vec encodings of the outer 4 words of the 5-gram,
      and an output layer fully connected to the hidden layer.
    
    args: vocabsize: the number of words in the vocab. The input layer will have dimension 4*vocabsize,
                     and the output layer will have dimension equal to vocabsize.
          repdim: the number of dimensions in which words will be represented. The hidden layer will have dimension 4*repdim.

    returns: in_layer: the input layer
             guess_layer: the final layer that guesses the word.
    """
    #parameters:
    Wenc = tf.Variable(np.random.randn(vocabsize, repdim)/vocabsize, dtype = tf.float32)
    benc = tf.Variable(0.1*np.random.randn(repdim), dtype = tf.float32)

    Wguess = tf.Variable(np.random.randn(4*repdim, vocabsize)/(4*repdim), dtype = tf.float32)
    bguess = tf.Variable(0.1*np.random.randn(vocabsize), dtype=tf.float32)

    #layers
    in_layer = tf.placeholder(dtype=tf.float32, shape=(None, 4*vocabsize))
    rep_layer1 = tf.nn.sigmoid(tf.matmul(tf.slice(in_layer, begin=(0,0), size=(-1, vocabsize)), Wenc) + benc)
    rep_layer2 = tf.nn.sigmoid(tf.matmul(tf.slice(in_layer, begin=(0,vocabsize), size=(-1, vocabsize)), Wenc) + benc)
    rep_layer4 = tf.nn.sigmoid(tf.matmul(tf.slice(in_layer, begin=(0,2*vocabsize), size=(-1, vocabsize)), Wenc) + benc)
    rep_layer5 = tf.nn.sigmoid(tf.matmul(tf.slice(in_layer, begin=(0,3*vocabsize), size=(-1, vocabsize)), Wenc) + benc)

    guess_layer = tf.matmul(tf.concat([rep_layer1, rep_layer2, rep_layer4, rep_layer5], 1), Wguess) + bguess #softmax shows up with the losses in the training graph

    return in_layer, guess_layer

def buildGuess2(vocabsize, repdims):
    """
    2-layer verion of buildGuess. repdims is a tuple of 2 layers of weights.
    """
    #parameters:
    W1enc = tf.Variable(np.random.randn(vocabsize, repdims[0])/vocabsize, dtype = tf.float32)
    b1enc = tf.Variable(0.1*np.random.randn(repdims[0]), dtype = tf.float32)
    W2enc = tf.Variable(np.random.randn(repdims[0], repdims[1])/repdims[0], dtype = tf.float32)
    b2enc = tf.Variable(0.1*np.random.randn(repdims[1]), dtype = tf.float32)

    W1guess = tf.Variable(np.random.randn(4*repdims[1], 4*repdims[0])/(4*repdims[1]), dtype = tf.float32)
    b1guess = tf.Variable(0.1*np.random.randn(4*repdims[0]), dtype=tf.float32)
    W2guess = tf.Variable(np.random.randn(4*repdims[0], vocabsize)/(4*repdims[0]), dtype=tf.float32)
    b2guess = tf.Variable(0.1*np.random.randn(vocabsize), dtype=tf.float32)

    #layers
    in_layer = tf.placeholder(dtype=tf.float32, shape=(None, 4*vocabsize))
    
    rep1_layer1 = tf.nn.sigmoid(tf.matmul(tf.slice(in_layer, begin=(0,0), size=(-1, vocabsize)), W1enc) + b1enc)
    rep1_layer2 = tf.nn.sigmoid(tf.matmul(tf.slice(in_layer, begin=(0,vocabsize), size=(-1, vocabsize)), W1enc) + b1enc)
    rep1_layer4 = tf.nn.sigmoid(tf.matmul(tf.slice(in_layer, begin=(0,2*vocabsize), size=(-1, vocabsize)), W1enc) + b1enc)
    rep1_layer5 = tf.nn.sigmoid(tf.matmul(tf.slice(in_layer, begin=(0,3*vocabsize), size=(-1, vocabsize)), W1enc) + b1enc)

    rep2_layer1 = tf.nn.sigmoid(tf.matmul(rep1_layer1, W2enc) + b2enc)
    rep2_layer2 = tf.nn.sigmoid(tf.matmul(rep1_layer2, W2enc) + b2enc)
    rep2_layer4 = tf.nn.sigmoid(tf.matmul(rep1_layer4, W2enc) + b2enc)
    rep2_layer5 = tf.nn.sigmoid(tf.matmul(rep1_layer5, W2enc) + b2enc)
    
    hid_layer = tf.nn.sigmoid(tf.matmul(tf.concat([rep2_layer1, rep2_layer2, rep2_layer4, rep2_layer5], 1), W1guess) + b1guess)
    guess_layer = tf.matmul(hid_layer, W2guess) + b2guess #softmax shows up with the losses in the training graph

    return in_layer, guess_layer
    
def buildEncode(vocabsize, repdim):
    """
    Builds the tensorflow graph for encoding words after training on the guess graph.

    args: vocabsize: the number of words in the vocab. Size of the input vector.
          repdim: the number of encoding dimensions. Size of the output layer.

    returns: in_layer: the input layer
             rep_layer: the output layer
    """
    #parameters
    Wenc = tf.Variable(np.zeros((vocabsize,repdim)), dtype=tf.float32)
    benc = tf.Variable(np.zeros(repdim), dtype=tf.float32)

    #layers
    in_layer = tf.constant(np.eye(vocabsize), dtype=tf.float32)
    rep_layer = tf.matmul(in_layer, Wenc) + benc

    #initialization
    init = tf.global_variables_initializer()
    restorer = tf.train.Saver()

    return init, restorer, rep_layer

def buildEncode2(vocabsize, repdims):
    """
    2-layer version of buildEncode. repdims should be a tuple of length 2.
    """
    #parameters
    W1enc = tf.Variable(np.zeros((vocabsize,repdims[0])), dtype=tf.float32)
    b1enc = tf.Variable(np.zeros(repdims[0]), dtype=tf.float32)
    W2enc = tf.Variable(np.zeros((repdims[0],repdims[1])), dtype=tf.float32)    
    b2enc = tf.Variable(np.zeros(repdims[1]), dtype=tf.float32)

    #layers
    in_layer = tf.constant(np.eye(vocabsize), dtype=tf.float32)
    hid_layer = tf.nn.sigmoid(tf.matmul(in_layer, W1enc) + b1enc)
    rep_layer = tf.matmul(hid_layer, W2enc) +b2enc

    #initialization
    init = tf.global_variables_initializer()
    restorer = tf.train.Saver()

    return init, restorer, rep_layer


def buildTraining(vocabsize, repdims, guess=buildGuess):
    """
    Build the tensorflow graph for training Word2Vec on the guess graph.

    args: vocabsize: the number of words in the vocab.
          repdims: the number of encoding dimensions. If using guess=buildGuess2, should be a tuple of 2 dimension sizes.
          guess: the function that builds the guess graph.

    returns: in_layer: input layer placeholder
             labels: labels placeholder
             train_step: training step node (running trains for one batch)
             loss: the loss node
             init: initializer node
             saver: Saver object for saving weights
    """
    #build guess graph
    in_layer, guess_layer = guess(vocabsize, repdims)
    labels = tf.placeholder(dtype=tf.float32, shape=(None, vocabsize))

    #training
    loss = tf.losses.softmax_cross_entropy(labels, guess_layer)
    optimizer = tf.train.AdamOptimizer()
    train_step = optimizer.minimize(loss)

    #initialization and saving
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    return in_layer, labels, train_step, loss, init, saver

#-----------------------------Run graphs--------------------------------

def runTraining(in_layer, labels, train_step, loss, init, saver, grams, vocab, batchsize=10, ts_per_epoch=10, num_epochs=10, valid_frac=10, model_name = 'model'):
    """
    Trains and saves on the training graph.

    args: in_layer, labels, train_step, loss, init, saver (see buildTraining function for more details)
          grams: the list of 5-grams on which the network will be trained
          vocab: the vocab dict for grams
          batchsize: the size of batches on which to train
          num_epochs: the number of epochs (1 epoch = 10 train steps, validation run after each epoch)
          ts_per_epoch: the number of training steps per epoch
          valid_frac: 1/valid_frac is the fraction of the samples used as validation set (test set is the same size).
          model_name: the name of the model for saving .ckpt files
          
    returns: train_losses: losses on the training set after each train step (len(train_losses)=10*num_epochs)
             valid_losses: losses on the validation set after each epoch (len(valid_losses)=num_epochs)
             test_loss: average loss on the test set (a scalar)
             save_epoch: the epoch after which parameters were saved (due to early stopping)
    """
    #divide into train, valid, test
    shuffle(grams)
    vocabsize = len(vocab)
    valid_size = len(grams)//valid_frac #size of validation and test sets
    train_grams = grams[:len(grams)-2*valid_size]
    valid_grams = grams[len(grams)-2*valid_size:len(grams)-valid_size]
    test_grams = grams[len(grams)-valid_size:]

    #run the graph, accumulating losses
    train_losses = []
    valid_losses = []
    print('Start Session:',end='')
    with tf.Session() as sess:
        sess.run(init)
        save_epoch = -1
        for epoch in range(num_epochs):
            print('.',sep='',end='')
            for step in range(ts_per_epoch):
                batch_arr, labels_arr = getBatch(batchsize, train_grams, vocab, len(vocab))
                _, lossstep = sess.run((train_step, loss), {in_layer: batch_arr, labels: labels_arr})
                train_losses.append(lossstep)
            valid_arr, labels_arr = getBatch(valid_size, valid_grams, vocab, len(vocab))
            lossstep = sess.run(loss, {in_layer: valid_arr, labels: labels_arr})
            valid_losses.append(lossstep)
            if epoch > 0.2*num_epochs and (save_epoch == -1 or lossstep < valid_losses[save_epoch]):
                saver.save(sess, '.\\Models\\'+model_name+'.ckpt')
                save_epoch = epoch
        print('\nFinished Training! Restoring best parameters, and running test loss.')
        saver.restore(sess, '.\\Models\\'+model_name+'.ckpt')
        test_arr, labels_arr = getBatch(valid_size, test_grams, vocab, len(vocab))
        test_loss = sess.run(loss, {in_layer: test_arr, labels: labels_arr})
        print('Parameters are saved under the name: {0}'.format(model_name))
    tf.reset_default_graph()
    return train_losses, valid_losses, test_loss, save_epoch
        
def runEncode(init, restorer, rep_layer, vocab, model_name='model'):
    """
    Run the encode graph and convert to a dict, which is saved

    args: vocab: the vocab dict

    returns: encoding: a dict with keys the vocab words and values the encoding
    """
    with tf.Session() as sess:
        sess.run(init)
        restorer.restore(sess, '.\\Models\\'+model_name+'.ckpt')
        encode_arr = sess.run(rep_layer)
    encoding = dict()
    for word in vocab.keys():
        encoding[word] = encode_arr[vocab[word]][:]
    np.save('.\\Models\\'+model_name+'_encoding.npy', encoding)
    tf.reset_default_graph()
    return encoding

def main(repdims, guess = buildGuess, model_name = 'model'):
    """
    Preps the data, runs the training, saves the model, runs the encoding and saves that, and then graphs the training

    args: repdims: if guess = buildGuess, the number of representation dimensions. if guess = buildGuess2, a tuple of the two representation dimensions
    """
    if guess == buildGuess:
        encode = buildEncode
        rd = repdims
    else:
        encode = buildEncode2
        rd = repdims[1]
    filenames = getFileNames('.\\Control_group')
    filenames.extend(getFileNames('.\\Data_group'))
    filenames.extend(getFileNames('.\\ML_group'))
    vocab = getVocab(filenames)
    np.save('.\\Models\\'+model_name+'_vocab.npy', vocab)
    grams = getGrams(filenames)
    vocabsize = 5944
    nodes = buildTraining(vocabsize, repdims, guess)
    train_losses, valid_losses, test_loss, save_epoch = runTraining(*nodes, grams, vocab, batchsize=20, ts_per_epoch=100, num_epochs=100, model_name=model_name)
    nodes = encode(vocabsize, repdims)
    encoding = runEncode(*nodes, vocab, model_name=model_name)
    print('In epoch {0}, the model was saved. The correct word had was given a 1/{1} chance at that time.'.format(save_epoch, exp(-test_loss)))
    x1 = np.arange(0,10000)
    x2 = np.arange(0,10000, 100)
    train_ce, = plt.plot(x1, train_losses, 'r', label='Training Loss')
    valid_cd, = plt.plot(x2, valid_losses, 'y', label='Validation Loss')
    plt.legend()
    plt.xlabel('Training step')
    plt.ylabel('Cross-Entropy')
    plt.title('Training a {0} Dimensional Word2Vec Encoding'.format(rd))
    plt.show()
    return encoding

def docAvg(filename, encoding):
    """
    Finds the average of the word vectors in a document.

    args: filename: the name of the text file
          encoding: the encoding resulting from running W2V

    returns: doc_mean: the average of the documents' word vectors.
    """
    text_list = importText(filename)
    word_vecs = [encoding[word] for word in text_list]
    return np.mean(np.array(word_vecs), axis = 0)

def svmDocAvg(encoding):
    """
    Runs a linear SVM analysis on document averages of W2V encodings. ml label is 1, ds label is 0.

    args: encoding: the W2V encoding

    returns: classifier: the SVM classifier (a sklearn.svm.LinearSVC object).
             valid_acc: the validation accuracy
             test_acc: the test accuracy
    """
    #get data
    ml_filenames = getFileNames('.\\ML_group')
    ds_filenames = getFileNames('.\\Data_group')
    shuffle(ml_filenames)
    shuffle(ds_filenames)

    #apply docAvg
    ml_avgs = [docAvg(filename, encoding) for filename in ml_filenames]
    ds_avgs = [docAvg(filename, encoding) for filename in ds_filenames]

    #put data in form usable by sklearn.svm
    train_avgs = ml_avgs[:40]+ds_avgs[:40]
    train_labs = np.concatenate((np.ones(40), np.zeros(40)))

    #create and train classifier
    classifier = svm.LinearSVC()
    classifier.fit(train_avgs, train_labs)

    #test classifier on validation and test sets
    valid_acc = classifier.score(ml_avgs[40:45]+ds_avgs[40:45], np.concatenate((np.ones(5), np.zeros(5))))
    test_acc = classifier.score(ml_avgs[45:]+ds_avgs[45:], np.concatenate((np.ones(5), np.zeros(5))))

    return classifier, valid_acc, test_acc, ml_avgs, ds_avgs

def plotSVM(dataseries1, dataseries2, coef, intercept, title):
    x = np.arange(-10,10,0.01)
    y = -coef[0]/coef[1]*x - intercept/coef[1]
    plt.plot(x,y)
    ax = plt.subplot()
    d = np.zeros(len(y))
    e = -np.ones(len(y))
    ax.fill_between(x, y, where=y<d, color='b', alpha=0.3)
    ax.fill_between(x, y, e, color='#ffa500', alpha=0.3)
    ml_plt = plt.scatter([k[0] for k in dataseries1], [k[1] for k in dataseries1],c='#ffa500')
    ds_plt = plt.scatter([k[0] for k in dataseries2], [k[1] for k in dataseries2],c='b')
    plt.legend((ml_plt, ds_plt),('Machine Learning','Data Science'))
    plt.title(title)
    plt.show()
    return None
