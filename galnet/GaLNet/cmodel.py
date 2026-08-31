#! -*- coding: utf-8 -*-
import numpy as np
from keras.models import Model
from keras.layers import *
from keras import regularizers


##############################################################
################ Normal CNN model ############################ 
##############################################################    
def CNN(x):

    #x = ZeroPadding2D((1,1))(x)
    #x=BatchNormalization(axis=3)(x)
    
    x=Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x=Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x=AveragePooling2D(pool_size=(2,2))(x)
    
    x=Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x=Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x=AveragePooling2D(pool_size=(2,2))(x)
    
    x=Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x=Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x=AveragePooling2D(pool_size=(2,2))(x)
    
    x=Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x=Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x=AveragePooling2D(pool_size=(2,2))(x)
    
    x=Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x=Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x=AveragePooling2D(pool_size=(2,2))(x)
    
    x = Flatten()(x)
    
    x=Dense(512,activation='relu')(x)
    x=Dense(512,activation='relu')(x)
    
    outpt=Dense(7)(x)
    return outpt

   
    
    
#############################################################
################### ResNet model ############################
#############################################################
def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None,regular=False):
    x = Conv2D(nb_filter, kernel_size, padding=padding,strides=strides)(x)
    #x = BatchNormalization(axis=3, name=bn_name)(x)
    x = Activation('relu')(x)
    return x   
    

def Basic_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False, regular=False):
    x = Conv2d_BN(inpt,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same', regular=regular)
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,padding='same', regular=regular)
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt,nb_filter=nb_filter,strides=strides,kernel_size=kernel_size)
        x = add([x,shortcut])
        return x
    else:
        x = add([x,inpt])
        return x
  
def ResNet(inpt):
    
    x=inpt
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(3, 3), padding='same')
    x=AveragePooling2D(pool_size=(2,2))(x)
    
    x = Basic_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = Basic_Block(x, nb_filter=64, kernel_size=(3, 3))

    x =Basic_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x =Basic_Block(x, nb_filter=128, kernel_size=(3, 3))
    
    x =Basic_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x =Basic_Block(x, nb_filter=256, kernel_size=(3, 3))
    
    x =Basic_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x =Basic_Block(x, nb_filter=512, kernel_size=(3, 3))

    x = Flatten()(x)
    #x = GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x)
    x=Dense(1024,activation='relu')(x)
    x=Dense(1024,activation='relu')(x)
    outpt=Dense(7,kernel_regularizer=regularizers.l2(0.0001))(x)
    
    return outpt
 




    
    
    
    
    
    

