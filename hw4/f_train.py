import pandas as pd
import numpy as np
import sys
import re
import jieba
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping , CSVLogger

def loadTrainData(trainXPath, trainYPath):

    with open(trainXPath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # Use regular expression to get rid of the index
        rawTrainX = [re.sub('^[0-9]+,', '', s) for s in lines[1:]]
        
    rawTrainY = pd.read_csv(trainYPath)['label']
    rawTrainY = np.array(rawTrainY)
    
    bid = '[bB][0-9]+'
    rawTrainX = [re.sub(bid, '', s) for s in rawTrainX]
    #rawTrainX = [re.sub('[a-zA-Z]+', 'a', s) for s in rawTrainX]
    #rawTrainX = [re.sub('[0-9]+', '0', s) for s in rawTrainX]
    
    rawTrainX = [list(jieba.cut(s, cut_all=False)) for s in rawTrainX]
    
    return rawTrainX, rawTrainY

def loadTestData(testXPath):
    
    with open(testXPath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # Use regular expression to get rid of the index
        rawTestX = [re.sub('^[0-9]+,', '', s) for s in lines[1:]]    

    bid = '[bB][0-9]+'
    rawTestX = [re.sub(bid, '', s) for s in rawTestX]
    #rawTestX = [re.sub('[a-zA-Z]+', 'a', s) for s in rawTestX]
    #rawTestX = [re.sub('[0-9]+', '0', s) for s in rawTestX]
    
    rawTestX = [list(jieba.cut(s, cut_all=False)) for s in rawTestX]
    
    return rawTestX

def get_valdata( x , y ):
    val_size = 20000
    val_x = x[ len(x)-val_size : ]
    val_y = y[ len(y)-val_size : ]
    x = x[ : len(x)-val_size ]
    y = y[ : len(y)-val_size ]
    return x,y,val_x,val_y

def wordEmbedding(rawTrainX):
    
    rawTrainX = [s for s in rawTrainX]
    
    # Train Word2Vec model
    emb_model = Word2Vec(rawTrainX, size=emb_dim)
    emb_model.save("f_word2vec_cut.model")
    num_words = len(emb_model.wv.vocab) + 1  # +1 for OOV words

    # Create embedding matrix (For Keras)
    emb_matrix = np.zeros((num_words, emb_dim), dtype=float)
    for i in range(num_words - 1):
        v = emb_model.wv[emb_model.wv.index2word[i]]
        emb_matrix[i+1] = v 

    # Convert words to index
    allTrainX = []
    for i, s in enumerate(rawTrainX):
        tmpList = []
        for word in s:
            try:
                tmpList.append(emb_model.wv.vocab[word].index + 1)
            except KeyError:
                continue
        allTrainX.append(tmpList)

    # Pad sequence to same length
    allTrainX = pad_sequences(allTrainX, maxlen=max_length)

    return allTrainX, emb_matrix, emb_model

if __name__ == '__main__':
    dictionary = sys.argv[4]
    
    jieba.set_dictionary( dictionary )
    trainXPath = sys.argv[1]
    trainYPath = sys.argv[2]
    testDataPath = sys.argv[3]

    global max_length, emb_dim, num_words
    max_length = 200
    emb_dim = 200

    rawTrainX, rawTrainY = loadTrainData(trainXPath, trainYPath)
    rawTestX = loadTestData(testDataPath)
    rawTestX = [s for s in rawTestX]

    allTrainX, emb_matrix, emb_model = wordEmbedding(rawTrainX)

    num_words, _ = emb_matrix.shape

    X_train, Y_train, X_val, Y_val = get_valdata(allTrainX, rawTrainY)

    # Setting optimizer and compile the model
    model = Sequential()
    model.add(Embedding(num_words,
                        emb_dim,
                        weights=[emb_matrix],
                        input_length=max_length,
                        trainable=False))
                        
    model.add(LSTM(256, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    ## version 3 : model same as version 1; change max_length = 100 (old = 50) ; 0.75467
    ## version 4 : model same as version 1; change max_length = 200 ; 0.75683
    ## version 5 : same as version 4; change emb_dim = 200 ; 0.75792

    model.summary()

    adam = Adam(lr=0.001, decay=1e-6, clipvalue=0.5)
    model.compile(loss='binary_crossentropy',
                optimizer=adam,
                metrics=['accuracy'])

    # Setting callback functions
    csv_logger = CSVLogger('training.log')
    check_save  = ModelCheckpoint("model_w2v-{epoch:03d}-{val_acc:.3f}.h5",monitor='val_acc',save_best_only=True)    

    earlystopping = EarlyStopping(monitor='val_acc', 
                                patience=6, 
                                verbose=1, 
                                mode='max')
    # Train the model
    batch_size = 512
    epochs = 100
    model.fit(X_train, Y_train, 
            validation_data=(X_val, Y_val),
            epochs=epochs, 
            batch_size=batch_size,
            callbacks=[csv_logger, check_save])
