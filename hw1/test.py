# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 21:52:58 2018

@author: USER
"""

import pandas as pd # 引用Pandas 讀取資料套件並縮寫為 pd 
import numpy as np
import csv 
import sys

def predict(X,w,b): 
    return np.dot(X, w) + b

inputfile = sys.argv[1] ;
outputfile = sys.argv[2] ;

#let 
x = []
n_row = 0
text = open( inputfile ,"r")
row = csv.reader(text , delimiter= ",")

data_dim = 14

for r in row:
    if n_row % 18 != 15 and n_row % 18 != 16 :
        if n_row %18 == 0:
            x.append([])
            for i in range(2,11):
                x[n_row//18].append(float(r[i]) )
        else :
            for i in range(2,11):
                if r[i] !="NR":
                    x[n_row//18].append(float(r[i]))
                else:
                    x[n_row//18].append(0)
    n_row = n_row+1
text.close()

for i in range( len(x) ) :
    for j in range( len(x[i]) ) :
        #count_0 = 0
        if j >= 81 and j <= 89 and x[i][j] < 0. :
            # get the index of which data set inside pm2.5 value < 0
            x[i][j] = -1 * x[i][j] ;

print( len(x) )
# change x & y to ndarray 
x = np.array( x )
# read model
w = np.load('model_ta_alld_v_w.npy')
b = np.load('model_ta_alld_v_b.npy')

 #let 
ans = []

for i in range(len(x)) :
    ans.append(["id_"+str(i)])
    loss = predict(x[i],w,b)
    #print(loss)
    a = predict(x[i],w,b)
    ans[i].append(loss)
    

print(ans)

text = open(outputfile, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()