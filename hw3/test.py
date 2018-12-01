# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 03:17:33 2018

@author: USER
"""
import numpy as np
import pandas as pd
import sys
from keras.models import load_model
from keras.utils import to_categorical
from keras.models import Model, Input
from keras import layers

def readData( file ) :
    df = pd.read_csv( file , dtype=str)  #讀取 CSV 檔案
    #df_x.drop( ['SEX','MARRIAGE'] , axis=1 , inplace=True)
    y = df['label'].astype(int).values
    x = df['feature']
    x = [ np.fromstring(i, dtype = int, sep = ' ') for i in x ]
    
    # delete the wrong data
    delete_index = []
    for i in range(len(x)) :
        if x[i].shape[0] != 2304 :
            delete_index.append( i )
    x = np.delete(x, delete_index)
    y = np.delete(y, delete_index)
    
    # normalization
    x = x / 255 
    # 由原本三維轉為四維矩陣以符合CNN的需求，這是因為RGB圖片的格式為為width, height, channels，加上ID數維度為4。圖片為灰階因此其channel為1，轉換後的shape為(ID, width, height, channel)
    x = [ i.reshape(48, 48, 1) for i in x ]
    
    # Onehot encoding
    y = to_categorical(y)
    
    # set validation set 
    val_size = int( len(y)*0.1 )
    val_x = x[ len(x)-val_size : ]
    val_y = y[ len(y)-val_size : ]
    # reset x
    x = list(x[ : len(x)-val_size ])
    y = list(y[ : len(y)-val_size ])
    # change x & y to ndarray 
    x = np.array( x )
    y = np.array( y ) 
    val_x = np.array( val_x )
    val_y = np.array( val_y ) 
    
    # Data augmentation
    x_f = np.fliplr( x ) 
    x = np.concatenate( (x,x_f) )
    y = np.concatenate( (y,y) )
    return x, y, val_x, val_y
    
def readTestData( file ) :
    df = pd.read_csv( file , dtype=str)  #讀取 CSV 檔案
    #y = df['label'].astype(int).values
    x = df['feature']
    x = [ np.fromstring(i, dtype = int, sep = ' ') for i in x ]
    
    # 由原本三維轉為四維矩陣以符合CNN的需求，這是因為RGB圖片的格式為為width, height, channels，加上ID數維度為4。圖片為灰階因此其channel為1，轉換後的shape為(ID, width, height, channel)
    x = [ i.reshape(48, 48, 1) for i in x ]
    
    # change x & y to ndarray 
    x = np.array( x )
    # normalization
    x = x / 255 
    return x

def resultClassifier( result ) :
    new_result = []
    for r in result :
        max_p = max( r )
        for i in range( len(r) ) :
            if r[i] == max_p :
                new_result.append( i )
    return new_result 

def ensembleModels(models, model_input):
    # collect outputs of models in a list
    yModels=[model(model_input) for model in models] 
    # averaging outputs
    yAvg=layers.average(yModels) 
    # build model from same input and avg output
    modelEns = Model(inputs=model_input, outputs=yAvg,    name='ensemble')  
   
    return modelEns

if __name__ == "__main__":
    #x,y,val_x,val_y = readData( 'train.csv' )
    test_x = readTestData( sys.argv[1] )
    #print( x.shape )
    #print( y.shape )

    model_names = ['model_1.h5','model_2.h5','model_3.h5','model_4.h5']
    models=[]
    for i in range( len(model_names) ):
    
        modelTemp=load_model( model_names[i] ) # load model
        modelTemp.name="model"+str(i) # change name to be unique
        models.append(modelTemp)
    
    model_input = Input(shape=models[0].input_shape[1:]) # c*h*w
    model = ensembleModels(models, model_input)
    
    #model = model.load_model('model.h5')
    #source = model.evaluate(val_x,val_y)
    #print( 'Total loss in Validation Set : ' , source[0] )
    #print( 'Accuracy of Validation Set : ' , source[1] )
    
    result = model.predict(test_x)
    result = resultClassifier( result )
    #print( 'result of Testing Set : ' , result )
    
    ans = result
    index = np.arange(0, len(ans), 1)
    #index = [ "id_" + str(x) for x in index ]
    y = np.vstack( (index,ans) )
    y = y.T
    #print( y.T )
    df_y = pd.DataFrame(y, columns=["id","label"])
    df_y.to_csv(sys.argv[2], index=False)
    
    del model
    
