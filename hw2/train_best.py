# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 20:25:46 2018

@author: USER
"""
import pandas as pd
import numpy as np
import random

def feature_scaling(x , test_x):  
    #allx = np.vstack( (x,test_x) )
    mini = np.min(x, axis=0)
    maxi = np.max(x, axis=0)
    #np.save('model_max_x.npy',maxi)
    #np.save('model_min_x.npy',mini)
    return (x - mini) / (maxi - mini)

def cross_entropy(y,sigma): 
    return ( y * np.log( sigma ) + (1-y) * np.log( 1-sigma ) ) * ( -1 )

def readXFile( file ) :
    df_x = pd.read_csv( file , dtype=str)  #讀取 CSV 檔案
    #df_x.drop( ['SEX','MARRIAGE'] , axis=1 , inplace=True)
    # onehot_encoding
    #sex_onehot_encoding = pd.get_dummies( df_x['SEX'] , prefix='SEX' )
    #education_onehot_encoding = pd.get_dummies( df_x['EDUCATION'] , prefix='EDUCATION' )
    #marriage_onehot_encoding = pd.get_dummies( df_x['MARRIAGE'] , prefix='MARRIAGE' )
    df_x.drop( ['SEX','MARRIAGE' ] , axis=1 , inplace=True)
    #df_x = pd.concat( [df_x,sex_onehot_encoding] , axis = 1 )
    #df_x = pd.concat( [df_x,education_onehot_encoding] , axis = 1 )
    #df_x = pd.concat( [df_x,marriage_onehot_encoding] , axis = 1 )
    x = df_x.astype(float).values
    return x ;

def readYFile( file ) :
    df_y = pd.read_csv( file , dtype=str)  #讀取 CSV 檔案
    y = df_y['Y'].astype(int).values
    return y ;

x = readXFile( 'train_x.csv' )

y = readYFile( 'train_y.csv' )

test_x = readXFile( 'test_x.csv' )
#print( x[0].shape  )
x = feature_scaling(x,test_x)


# random shuffle the data set x&y
z = list( zip(x , y) )
random.Random(4).shuffle( z )
x , y = zip(*z)

# set validation set 
val_size = 1000
val_x = x[ len(x)-val_size :]
val_y = y[ len(y)-val_size :]
# reset x
x = list(x[: len(x)-val_size ])
y = list(y[: len(y)-val_size ])
# change x & y to ndarray 
x = np.array( x )
y = np.array( y ) 

# Logistic Regression
w = np.zeros( len(x[0]) )
b = 1 #bias
lamb = 0
iteration = 50000
learn_rate = 1
# learn_rate for w & b
lr_w = 1
lr_b = 1
#gama = 0.1


for i in range( iteration ) :
    z = np.dot(x,w) + b
#    print( z )
    sigma = 1 / (1 + np.exp(-1 * z))
    sigma = np.clip( sigma , 1 * (10**-10) , 1 - (1 * (10**-10)) ) #使用np.clip() 避免數值太小或太大而overflow
#    print( sigma )
    
    likelihood = np.sum( cross_entropy(y,sigma) )
    
#    print( likelihood )
    
    #grad_W = -np.dot((y - sigma),x) + ( lamb * w )
    #grad_b = -np.sum(y - sigma)
    grad_W = np.mean(-1 * x * (np.squeeze(y) - sigma).reshape((len(x),1)), axis=0)
    grad_b = np.mean(-1 * (np.squeeze(y) - sigma))
    #print( grad_W )
    # adagrad
    lr_w = lr_w + grad_W ** 2
    lr_b = lr_b + grad_b ** 2
    
    #update
    w = w - learn_rate / np.sqrt(lr_w) * grad_W
    b = b - learn_rate / np.sqrt(lr_b) * grad_b
#    print( w )
#    print( b )
    if i % 5000 == 0 :
        avgerr_ans = [ 1 if x >= 0.5 else 0 for x in sigma ]
        print ("iteration: %d | avgerr: %f " % ( i,np.count_nonzero([avgerr_ans[i]+y[i] for i in range(len(y))])/len(y) ))
    #print(   )
    
print( w )
print( b )

np.save('model_w.npy',w)
np.save('model_b.npy',b)

#validation set testing
print( "validation testing : " )

ans = []
val_loss = 0
z = np.dot(val_x,w) + b
sigma = 1 / (1 + np.exp(-1 * z))
sigma = np.clip( sigma , 1 * (10**-10) , 1 - (1 * (10**-10)) ) #使用np.clip() 避免數值太小或太大而overflow
ans = [ 1 if x >= 0.5 else 0 for x in sigma ]

for i in range(len(val_x)) :
    if val_y[i] != ans[i]:
        val_loss += 1
print( "validation avg loss = %f " % ( val_loss / len(val_y) ) )

