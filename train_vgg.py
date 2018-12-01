# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 03:17:33 2018

@author: USER
"""
import numpy as np
import pandas as pd
import sys
import tensorflow as tf
import keras
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping ,ReduceLROnPlateau
from keras.callbacks import History ,ModelCheckpoint

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

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"  #设置需要使用的GPU的编号
    #gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.6 #设置使用GPU容量占GPU总容量的比例
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    x,y,val_x,val_y = readData( sys.argv[1] )
    #print( x.shape )
    #print( y.shape )

    datagen = ImageDataGenerator(rotation_range=30,
                                 width_shift_range=0.15,height_shift_range=0.15,
                                 zoom_range=[0.8, 1.2],
                                 shear_range=0.25)

    model = Sequential()
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=(48, 48, 1), activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))
    
    
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.35))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.35))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.45))
    
    model.add(Flatten())

    model.add(Dense(256, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(64, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(7, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy' , optimizer='adam' , metrics=['accuracy'] )
    
    hist = History()
    early_stop = EarlyStopping(monitor='val_acc', patience=7, verbose=1)
    check_save  = ModelCheckpoint("model_vgg_2-{epoch:03d}-{val_acc:.5f}.h5",monitor='val_acc',save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=5)

        #model.fit( x , y , batch_size=500 , epochs=35 )
    epoch = 300
    model.fit_generator( datagen.flow(x , y , batch_size=128) ,
                        validation_data=(val_x,val_y),
                        samples_per_epoch=(len(x)), epochs=epoch,
                        callbacks=[check_save,hist,reduce_lr] )
    
   
    model.save('model_vgg_'+str(epoch)+'.h5')   # HDF5 file, you have to pip3 install h5py if don't have it

    #model = model.load_model('model.h5')
    source = model.evaluate(val_x,val_y)
    print( 'Total loss in Validation Set : ' , source[0] )
    print( 'Accuracy of Validation Set : ' , source[1] )
    
    #print( 'result of Testing Set : ' , result )
    
    del model
