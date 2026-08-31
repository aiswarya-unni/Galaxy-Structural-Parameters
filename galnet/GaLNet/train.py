#! -*- coding: utf-8 -*-
import time
import keras.optimizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint
import astropy.io.fits as pyfits
from keras import backend as K
import csv
import os
import cmodel
import keras
import warnings
from astropy.visualization import ZScaleInterval
warnings.filterwarnings("ignore")

#check for GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#-----------------Set Parameters-----------------

def set_params():
    global pix_num,epochs,batch_size,learning_rate,delta
    global train_num,val_num
    global GaLNet_dir
    global psf_pix_num
    
    pix_num=81
    psf_pix_num=35

    epochs=150
    batch_size=128
    learning_rate=1e-3
    delta=0.001
    
    train_num=100000
    val_num=20000

    GaLNet_dir='C:/lsst/GaLNet/' 
        
#-----------------------------Loss Functions----------------------

def r_square(y_true, y_pred):
    SSR = K.mean(K.square(y_pred-K.mean(y_true)),axis=-1)
    SST = K.mean(K.square(y_true-K.mean(y_true)),axis=-1)
    return SSR/SST
    
def huber_loss(y_true, y_pred):
    error = y_pred - y_true
    abs_error = K.abs(error)
    quadratic = K.minimum(abs_error, delta)   #errors smaller than delta, the Huber loss behaves like the mean squared error (quadratic)
    linear = abs_error - quadratic    # errors larger than delta, the Huber loss behaves like the mean absolute error (linear)
    return 0.5 * K.square(quadratic) + delta * linear
  
#-----------------------------Data Preprocessing----------------------

def read_catalog(train=True,num=2000):
    names,parameters=[],[]
    if train==True:
        catalog=GaLNet_dir+'data/train_data.csv'
        print("reading training catalog")  
    else:
        catalog=GaLNet_dir+'data/validation_data.csv'
        print("reading validation catalog")  
        
    with open (catalog) as f:
        reader=csv.DictReader(f)
        reader=list(reader)[:num]
        for row in reader:
            name=row['name']
            names.append(name)

            mag=(float(row['mag'])-16)/(25-16)
            xcen=(float(row['xcen'])-(-0.4))/(0.4-(-0.4))
            ycen=(float(row['ycen'])-(-0.4))/(0.4-(-0.4))
            R_eff=(float(row['R_eff'])-0.2)/(4-0.2)
            n =(float(row['n'])-0.1)/(8.0-0.1) 
            
            pa=float(row['pa'])
            q=float(row['q'])
            e1=(1-q)/(1+q)*(np.cos(2*pa/180*np.pi))  
            e2=(1-q)/(1+q)*(np.sin(2*pa/180*np.pi))    
            e1=(e1-(-0.67))/(0.67-(-0.67))  
            e2=(e2-(-0.67))/(0.67-(-0.67))  
            
            parameters.append([mag,xcen,ycen,R_eff,e1,e2,n]) 
    parameters = np.array(parameters)
    return names,parameters      

def change_PSF_size(psf):
    nx,ny=psf_pix_num,pix_num

    d1 = np.asarray(psf)
    nx_ , ny_ = np.shape (d1)
    PSF = np.zeros ( ( nx , ny ) ) 
    dx = ( nx - nx_ ) // 2 
    dy = ( ny - ny_ ) // 2 
    for ii in range ( nx_ ): 
        for jj in range ( ny_ ):
            PSF[ ii + dx ] [ jj + dy ] = d1 [ ii ] [ jj ]
    return PSF

def read_images(name,train_set = True):
    if train_set==True:
        image_dir=GaLNet_dir+'data/train/'
    else:
        image_dir=GaLNet_dir+'data/chandra_sim/validation/'
    
    name=image_dir+name
    with pyfits.open(name,memmap=False) as f:
        image=f[0].data*10**2
        psf=f[2].data*10**3
        psf=change_PSF_size(psf)
 
        
    x=np.concatenate((image,psf),axis=0)
    x=np.expand_dims(x,2)
    return x             #matrix
    
#---------------shuffle and batch the input data------------------
   
def func_data_arg(names,Y, batch_size,train_set = True):      
    
    idxs = list(range(len(names)))
    np.random.shuffle(idxs)                      #shuffle
    names = np.array(names)[idxs] #reorder names
    Y = np.array(Y)[idxs]     #reorder parameters
    
    batch_num = int(len(names) / batch_size)  
    max_len = batch_num * batch_size 
    names = np.array(names[:max_len])
    Y = np.array(Y[:max_len])
    #print(len(Y))
     
    names_batches = np.split(names, batch_num) 
    Y_batches = np.split(Y, batch_num)
    while True:
        for i in range(len(names_batches)):
            if train_set:
                x=list(map(read_images,names_batches[i],[True for _ in range(batch_size)]))
            else:
                x=list(map(read_images,names_batches[i],[False for _ in range(batch_size)]))
       
            x=np.array([a for a in x])
            y=np.array(Y_batches[i])
            yield x,y               

#-----------------------------Model----------------------

def models():
    inpt = Input(shape=(pix_num+psf_pix_num,pix_num,1))
    outpt=cmodel.CNN(inpt)
    model = Model(inputs=inpt, outputs=outpt)
    model.summary()
    return model    

#-----------------Training History-----------------

def plot_history(history_df):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    
    # Plot Loss
    axs[0, 0].plot(history_df['loss'], label='Training Loss')
    axs[0, 0].plot(history_df['val_loss'], label='Validation Loss')
    axs[0, 0].set_title('Loss')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # Plot MSE
    axs[0, 1].plot(history_df['mse'], label='Training MSE')
    axs[0, 1].plot(history_df['val_mse'], label='Validation MSE')
    axs[0, 1].set_title('MSE')
    axs[0, 1].set_ylabel('MSE')
    axs[0, 1].legend()

    # Plot MAE
    axs[1, 0].plot(history_df['mae'], label='Training MAE')
    axs[1, 0].plot(history_df['val_mae'], label='Validation MAE')
    axs[1, 0].set_title('MAE')
    axs[1, 0].set_ylabel('MAE')
    axs[1, 0].legend()

    # Plot R^2
    axs[1, 1].plot(history_df['r_square'], label='Training R^2')
    axs[1, 1].plot(history_df['val_r_square'], label='Validation R^2')
    axs[1, 1].set_title('R^2')
    axs[1, 1].set_ylabel('R^2')
    axs[1, 1].legend()
    
    # Set common labels
    for ax in axs.flat:
        ax.set_xlabel('Epoch')

    # Save and show plot
    plt.savefig('./GaLNet/fig/training_history.png')
    plt.close()

#---------------Main----------------- 
   
if __name__ == '__main__':
    
    set_params()
    names_train,parameters_train=read_catalog(num=train_num)  
    names_val,parameters_val=read_catalog(num=val_num,train=False)
    
    steps_train=int(len(names_train)/batch_size) 
    steps_val=int(len(names_val)/batch_size)   
 
    
    checkpoint = ModelCheckpoint('./model/weights.h5', monitor='val_loss',verbose=1, save_best_only=True)
    model=models()

    model.compile(loss=huber_loss, optimizer=Adam(learning_rate), metrics=['mse','mae',r_square])
    

    data_gen_train=func_data_arg(names_train,parameters_train,batch_size,train_set = True)
    data_gen_val=func_data_arg(names_val,parameters_val,batch_size,train_set = False)
 
    t1=time.time()
    history = model.fit(data_gen_train,
                        steps_per_epoch=steps_train, 
                        epochs=epochs,
                        validation_data=data_gen_val,
                        validation_steps=steps_val,
                        callbacks=[checkpoint],
                        verbose=1)
    t2=time.time()
    print("Training time (hours): ", (t2-t1)/3600)
              
    # Save metrics to CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('./GaLNet/result/training_history.csv', index=False)

    plot_history(history_df)








