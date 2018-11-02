# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 22:28:07 2018

@author: USER
"""
import pandas as pd
import numpy as np
import sys

def feature_scaling(x):  
    min_x = np.min(x, axis=0)
    max_x = np.max(x, axis=0)
    return (x - min_x) / (max_x - min_x)
def readXFile( file ) :
    df_x = pd.read_csv( file , dtype=str)  #讀取 CSV 檔案
    #df_x.drop( ['SEX','MARRIAGE'] , axis=1 , inplace=True)
    # onehot_encoding
    #sex_onehot_encoding = pd.get_dummies( df_x['SEX'] , prefix='SEX' )
    #education_onehot_encoding = pd.get_dummies( df_x['EDUCATION'] , prefix='EDUCATION' )
    #marriage_onehot_encoding = pd.get_dummies( df_x['MARRIAGE'] , prefix='MARRIAGE' )
    df_x.drop( ['SEX','MARRIAGE'] , axis=1 , inplace=True)
    #df_x = pd.concat( [df_x,sex_onehot_encoding] , axis = 1 )
    #df_x = pd.concat( [df_x,education_onehot_encoding] , axis = 1 )
    #df_x = pd.concat( [df_x,marriage_onehot_encoding] , axis = 1 )
    x = df_x.astype(float).values
    return x ;


xfile = sys.argv[1] ;
yfile = sys.argv[2] ;
testxxfile = sys.argv[3] ;
outputfile = sys.argv[4] ;

x = readXFile( testxxfile )
y = []


#print( x[0].shape  )

#max_x = np.load('model_max_x.npy')
#min_x = np.load('model_min_x.npy')

x = feature_scaling(x)

# Logistic Regression
w = np.load('model_w.npy')
b = np.load('model_b.npy')

z = np.dot(x,w) + b
sigma = 1 / (1 + np.exp(-1 * z))
sigma = np.clip( sigma , 1 * (10**-10) , 1 - (1 * (10**-10)) ) #使用np.clip() 避免數值太小或太大而overflow
#print( sigma )
#
#ans = []
#for i in range(len(x)) :
#    z = np.dot(x[i],w) + b
#    sigma = 1 / (1 + np.exp(-1 * z))
#    ans[i].append(sigma)
#    

ans = [ 1 if x >= 0.5 else 0 for x in sigma ]

print( ans )

index = np.arange(0, len(ans), 1)
index = [ "id_" + str(x) for x in index ]
y = np.vstack( (index,ans) )
y = y.T
#print( y.T )
df_y = pd.DataFrame(y, columns=["id","Value"])
df_y.to_csv(outputfile, index=False)
