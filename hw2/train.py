# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 20:25:46 2018

@author: USER
"""
import pandas as pd
import numpy as np

def feature_scaling(x , test_x):  
    allx = np.vstack( (x,test_x) )
    mini = np.min(allx, axis=0)
    maxi = np.max(allx, axis=0)
    np.save('model_max_x.npy',maxi)
    np.save('model_min_x.npy',mini)
    return (x - mini) / (maxi - mini)

def cross_entropy(y,sigma): 
    return ( y * np.log( sigma ) + (1-y) * np.log( 1-sigma ) ) * ( -1 )

def class_counter( data , class_y ) :
    count = 0 
    for d in data :
        if d == class_y :
            count += 1
    return count

def class_mean_cal( x , y , class_y , class_len ) :
    count = 0
    for i in range( len(x) ) :
        if y[i] == class_y :
            count += x[i]
    return count / class_len

def covariance_martix_cal( x , y , mean , class_y , class_len ) :
    sigma = 0
    for i in range( len(x) ) :
        if y[i] == class_y :
            sigma += np.dot( (x[i]-mean).reshape((len(x[0]),1)) , x[i]-mean.reshape((1,len(x[0]))) ) # (x[i]-mean).reshape((len(x[0]),1)) = (x[i]-mean).T
    return sigma / class_len
        
def readXFile( file ) :
    df_x = pd.read_csv( file , dtype=str)  #讀取 CSV 檔案
    #df_x.drop( ['SEX','MARRIAGE'] , axis=1 , inplace=True)
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

# Probabilistic Generative Model
count_class1 = class_counter( y , 1 )
count_class0 = class_counter( y , 0 )
mean_class1 = class_mean_cal( x , y , 1 , count_class1 )
mean_class0 = class_mean_cal( x , y , 0 , count_class0 )
conMatrix_class1 = covariance_martix_cal( x , y , mean_class1 , 1 , count_class1 )
conMatrix_class0 = covariance_martix_cal( x , y , mean_class0 , 0 , count_class0 )
conMatrix_share = count_class1 / len(x) * conMatrix_class1 + count_class0 / len(x) * conMatrix_class0

print( 'done!' )

conMatrix_inverse = np.linalg.inv(conMatrix_share)

w = np.dot( mean_class1 - mean_class0 , conMatrix_inverse ) 
b = ( -0.5 * np.dot( np.dot( mean_class1.T , conMatrix_inverse ) , mean_class1 ) 
    + 0.5 * np.dot( np.dot( mean_class0.T , conMatrix_inverse ) , mean_class0 ) 
    + np.log( count_class1 / count_class0 ) )


print( w )
print( b )

np.save('model_generative_w.npy',w)
np.save('model_generative_b.npy',b)



