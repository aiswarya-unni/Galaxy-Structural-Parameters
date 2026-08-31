import time
import astropy.io.fits as pyfits
import csv
import numpy as np
from scipy.special import *
from keras.models import Model
from keras.layers import *
import warnings
warnings.filterwarnings('ignore')
import cmodel
import os
import openvino as ov

#-----------------Set Parameters-----------------

def set_parameters():
    global pix_num,num
    global GaLNet_dir
    global psf_pix_num
    
    num=10000
    pix_num=81
    psf_pix_num=35
    GaLNet_dir='C:/lsst/GaLNet/' 


def csv_header():
    fname = f'./GaLNet/result/pred_test_para.csv'
    headers = ['name','mag_true','x_true','y_true','re_true','e1_true','e2_true','n_true',
               'mag_cnn','x_cnn','y_cnn','re_cnn','e1_cnn','e2_cnn','n_cnn','SNR']
    with open(fname, 'w') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(headers)
    f.close()

def save_csv(paras):
    fname = f'./GaLNet/result/pred_test_para.csv'
    data=paras
    with open(fname, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data)
    f.close() 

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

#-----------------Read Data-----------------

def read_data():

    catalog =  f"{GaLNet_dir}/data/test_data.csv"
    image_dir = f"{GaLNet_dir}/data/test/"

    X,Y=[],[]
    name_save=[]
    SNR=[]
    a=0
    
    with open (catalog) as f:
        reader=csv.DictReader(f)
        reader=list(reader)[:num]
        for row in reader:
            a=a+1
            if a%5000==0:
                print ('Data reading finished:', 100.0*a/(num),"%")
            name=row['name']
            name_save.append(name)
            SNR.append(row['SNR'])

            name=image_dir+name
            with pyfits.open(name,memmap=False) as f:
                image=f[0].data*10**2
                psf=f[2].data*10**3
                psf=change_PSF_size(psf)
               
            x=np.concatenate((image,psf),axis=0)
            x=np.expand_dims(x,2)
           

            mag=(float(row['mag'])-16)/(25-16)
            xcen=(float(row['xcen'])-(-0.4))/(0.4-(-0.4))
            ycen=(float(row['ycen'])-(-0.4))/(0.4-(-0.4))

            #Choose the normalization method for R_eff based on the available data
            R_eff=(float(row['R_eff'])-0.2)/(4-0.2)     #for simulated data
            #re_circ = float(row['sersic_0_rad_arcsec']) * np.sqrt(float(row['sersic_0_ratio']))
            #R_eff = (re_circ - 0.2) / (4 - 0.2)   #for the rparameters from DP1 object catalog

            n =(float(row['n'])-0.1)/(8.0-0.1) 
            
            pa=float(row['pa'])
            q=float(row['q'])
            e1=(1-q)/(1+q)*(np.cos(2*pa/180*np.pi))
            e2=(1-q)/(1+q)*(np.sin(2*pa/180*np.pi))
            e1=(e1-(-0.67))/(0.67-(-0.67))
            e2=(e2-(-0.67))/(0.67-(-0.67))
            
            X.append(x)
            Y.append([mag,xcen,ycen,R_eff,e1,e2,n]) 
    X = np.array(X)
    Y = np.array(Y)
    return X,Y,name_save,SNR

#------------------Re-scale Parameters-----------------

def re_scale(y):
    y_new=[]
    for i in range(len(y)):
        para=y[i]
        para[0]=para[0]*(25-16)+16
        para[1]=para[1]*(0.4-(-0.4))+(-0.4)
        para[2]=para[2]*(0.4-(-0.4))+(-0.4)
        para[3]=para[3]*(4-0.2)+0.2
        para[4]=para[4]*(0.67-(-0.67))+(-0.67)
        para[5]=para[5]*(0.67-(-0.67))+(-0.67)
        para[6]=para[6]*(8-0.1)+0.1
        y_new.append(para)
    return y_new

#----------------Load Model-----------------
def models():
    inpt = Input(shape=(pix_num+psf_pix_num,pix_num,1))
    outpt=cmodel.CNN(inpt)
    model = Model(inputs=inpt, outputs=outpt)
    model.summary()
    return model

#----------------Prediction-----------------

def prediction(X):
    model=models()
    model.load_weights(f'./model/weights.h5')
    y_pred=model.predict(X,batch_size=128)
    return y_pred


#---------------Main----------------- 

if __name__ == '__main__':  
    set_parameters()  
    csv_header()
    
    X,Y,name,SNR=read_data()
    
    t1=time.time()
    y_pred=prediction(X)
    y_pred=re_scale(y_pred)
    
    y_true=Y
    y_true=re_scale(y_true)

    #print total time for prediction
    t2=time.time()
    print('Prediction time:', t2-t1, 'seconds')

    for i in range(len(y_pred)):
        paras=[name[i]]+list(np.concatenate([y_true[i],y_pred[i]]))+[SNR[i]]
        save_csv(paras)

    os.system('python GalNet/plot_test.py')

    
    
























    
    
