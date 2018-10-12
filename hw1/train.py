# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 20:53:30 2018

@author: USER
"""
import pandas as pd # 引用Pandas 讀取資料套件並縮寫為 pd 
import numpy as np
import random 
import csv

def feature_scaling(X, train=False):  
    mini = 0
    maxi = 0
    if train:
        mini = np.min(X, axis=0)
        maxi = np.max(X, axis=0)
    return (X - mini) / (maxi - mini)

def predict(X,w,b): 
    return np.dot(X, w) + b

def RMSELoss(X, Y,w,b):
    return np.sqrt(np.mean((Y - predict(X,w,b))** 2) )


data = []
# 每一個維度儲存一種污染物的資訊
for i in range(18):
	data.append([])

n_row = 0 
text = open('train.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0列沒有資訊
    if n_row != 0 and (n_row-1) % 18 != 15 and (n_row-1) % 18 != 16 :
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))	
    n_row = n_row+1
text.close()

data = [ d for d in data if d != [] ]

for i in range( len(data) ) :
   print( len(data[i]) )

x = []
y = []
# 每 12 個月
for i in range(12):
    # 一個月取連續10小時的data可以有471筆
    for j in range(471):
        x.append([])
        # 18種污染物
        for t in range(16):
            # 連續9小時
            for s in range(9):
                x[471*i+j].append(data[t][480*i+j+s] )
        y.append(data[9][480*i+j+9])

print( len(x) )
print( len(y) ) 

# random shuffle the data set x&y
z = list( zip(x , y) )
random.Random(4).shuffle( z )
x , y = zip(*z)

# set validation set 
val_size = 219
val_x = x[ len(x)-val_size :]
val_y = y[ len(y)-val_size :]
# reset x
x = list(x[: len(x)-val_size ])
y = list(y[: len(y)-val_size ])
#
#print( x )
#print( y ) 

# delete the data useless if pm2.5 value < 0         
delete_index = [] ;
for i in range( len(x) ) :
    count_pm25_0 = 0
    count_pm10_0 = 0
    if y[i] <= 0. or y[i] > 150 : 
        # get the index of which data set inside pm2.5 value < 0
        delete_index.append( i ) ;
        continue ;
    for j in range( len(x[i]) ) :
        #count_0 = 0
        if j >= 72 and j <= 89 and x[i][j] < 0. :
            # get the index of which data set inside pm2.5 value < 0
            delete_index.append( i ) ;
            break ;
        if j >= 81 and j <= 89 and x[i][j] > 150 :
            delete_index.append( i ) ;
            break ;
        #if y[i] < 0. or y[i] > 150  :
            # get the index of which data set inside pm2.5 value < 0
            #delete_index.append( i ) ;
        if j >= 81 and j <= 89 and x[i][j] == 0 :
            count_pm25_0 += 1
        if count_pm25_0 > 3 :
            delete_index.append( i ) ;
            break ;
        #pm10
        if j >= 72 and j <= 80 and x[i][j] == 0 :
            count_pm10_0 += 1
        if count_pm10_0 > 3 :
            delete_index.append( i ) ;
            break ;
       
# pop the useless data
delete_index = np.array( delete_index )
for index in delete_index :
    x.pop( index )
    y.pop( index )
    delete_index -= 1
    
# change x & y to ndarray 
x = np.array( x )
y = np.array( y ) 
print( len(x) )
print( len(y) ) 

# set train info.
loss_value = 0 
w = np.zeros(len(x[0]))
b = 1 ; # bias
learn_rate = 0.1
lamb = 1
batch_size = x.shape[0]
# learn_rate for w & b
#lr_w = 0
lr_w = 0
lr_b = 0

# train times
iteration = 50000

# train
print( "training : " )

for epoch in range(iteration):
        
    # mse loss
    grad_b = -np.sum(y - predict(x,w,b))/ batch_size
    grad_W = -np.dot(x.T, (y - predict(x,w,b))) / batch_size + ( lamb * w )
    
    # adagrad
    lr_b += grad_b ** 2
    lr_w += grad_W ** 2
    
    #update
    b = b - learn_rate / np.sqrt(lr_b) * grad_b
    w = w - learn_rate / np.sqrt(lr_w) * grad_W
    
    scalars_loss_w = np.sum(abs(grad_W)) / len(grad_W)
    print ("iteration: %d | loss_w: %f | RMSELoss: %f " % ( epoch,scalars_loss_w,abs(RMSELoss(x,y,w,b)) ))

#for i in range(iteration):
#    loss_w = 0
#    loss_b = 0
#    
#    for j in range(len(x)):
##        print( "x[j] : " + str(x[j]) )
##        print( "b - np.dot(w,x[j]) : " + str(b + np.dot(w,x[j])) )
#        loss_w += 2.0 * ( y[j] - b - np.dot(w,x[j]) ) * ( -1 * x[j] ) + ( 2 * lamb * w )
#        loss_b += 2.0 * ( y[j] - b - np.dot(w,x[j]) ) * ( -1 )
#        
#        
#    # update learning rate
#    lr_w = lr_w + loss_w ** 2
#    lr_b = lr_b + loss_b ** 2
#    # update w & b
#    w = w - learn_rate / np.sqrt(lr_w) * loss_w
#    b = b - learn_rate / np.sqrt(lr_b) * loss_b
#    
#    scalars_loss_w = np.sum(abs(loss_w)) / len(loss_w)
#    print ("iteration: %d | loss_w: %f | loss_b: %f " % ( i,scalars_loss_w,abs(loss_b) ))

print( w )
print( b )

np.save('model_ta_alld_v_w.npy',w)
np.save('model_ta_alld_v_b.npy',b)

#validation set testing
print( "validation testing : " )

ans = []
val_loss = []
for i in range(len(val_x)) :
    ans.append(["id_"+str(i)])
    a = b + np.dot(w,val_x[i])
    loss = val_y[i] - a
    val_loss.append( abs(loss) )
    ans[i].append(a)
    print("validation: %d | loss: %f " % ( i, loss ) )

print( "validation avg loss = %f " % ( sum(val_loss) / len(val_y) ) )
