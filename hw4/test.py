# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 03:17:33 2018

@author: USER
"""
import numpy as np
import pandas as pd
import csv
import sys 
import jieba
import re
from keras.models import load_model
from keras.models import Model, Input
from keras import layers
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec as w2v

def resultClassifier( result ) :
    new_result = []
    for r in result :
        max_p = max( r )
        for i in range( len(r) ) :
            if r[i] == max_p :
                new_result.append( i )
    return new_result 

def loadTestData(testXPath):
    
    with open(testXPath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # Use regular expression to get rid of the index
        rawTestX = [re.sub('^[0-9]+,', '', s) for s in lines[1:]]    

    bid = '[bB][0-9]+'
    rawTestX = [re.sub(bid, '', s) for s in rawTestX]
    rawTestX = [re.sub('[a-zA-Z]+', 'a', s) for s in rawTestX]
    rawTestX = [re.sub('[0-9]+', '0', s) for s in rawTestX]
    
    rawTestX = [list(jieba.cut(s, cut_all=False)) for s in rawTestX]
    
    return rawTestX

def ensembleModels(models, model_input):
    # collect outputs of models in a list
    yModels=[model(model_input) for model in models] 
    # averaging outputs
    yAvg=layers.average(yModels) 
    # build model from same input and avg output
    modelEns = Model(inputs=model_input, outputs=yAvg,    name='ensemble')  
   
    return modelEns

def getModel() :
    model_names = ['model_w2v_1.h5',
                   'model_w2v_2.h5',
                   'model_w2v_3.h5']
    models=[]
    for i in range( len(model_names) ):
    
        modelTemp=load_model( model_names[i] ) # load model
        modelTemp.name="model"+str(i) # change name to be unique
        models.append(modelTemp)
    
    model_input = Input(shape=models[0].input_shape[1:]) # c*h*w
    model = ensembleModels(models, model_input)
    return model

if __name__ == "__main__":
    test_file = sys.argv[1]
    dictionary = sys.argv[2]
    out_file = sys.argv[3]
    
    
    jieba.set_dictionary( dictionary )
    x = loadTestData( test_file )

    max_length = 200
    emb_model = w2v.load('f_word2vec_cut.model')

    testX = []
    for i, sentnce in enumerate(x):
        tmpList = []
        for word in sentnce:
            try:
                emb_model.wv.vocab[word]
            except KeyError:
                continue
            else:
                tmpList.append(emb_model.wv.vocab[word].index + 1)
        testX.append(tmpList)
        
    test_x = pad_sequences( testX , maxlen=max_length )
	
    #rnn_model = load_model( sys.argv[1] )
    rnn_model = getModel()
    ans = []
    result = rnn_model.predict(test_x)
    print( 'get result' )
    for i in range( len(result) ) :
        if result[i] < 0.5 :
            ans.append(0) 
        else :
            ans.append(1) 
    
    print( ans )
    #result = resultClassifier( result )
	
    #ans = result
    index = np.arange(0, len(ans), 1)
    #index = index.reshape(len(index),1)
    #index = [ "id_" + str(x) for x in index ]
    print( len(index) , len(ans) )
    y = np.vstack( (index,ans) )
    y = y.T
    #print( y.T )
    df_y = pd.DataFrame(y, columns=["id","label"])
    df_y.to_csv( out_file , index=False)#(sys.argv[2], index=False)
    
    del rnn_model


