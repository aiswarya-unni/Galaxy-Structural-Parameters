import glob
import sys
import os
import shutil
import csv
import random
import astropy.io.fits as pyfits
import scipy.signal as signal
import scipy.special as sf
import scipy.ndimage as  ndimage
from scipy.special import *
from scipy.stats import expon
import numpy as np
from skimage.transform import resize
sys.setrecursionlimit(100000)


#-----------------Set Parameters-----------------

def set_parameters():
    global num_need,snr_crt,mode
    global pix_num,pix_size,lenth,HR_pix_num   
    global noise_dir,data_dir,train_img_dir,test_img_dir,galaxy_dir, psf_dir
    global noise_cata,train_cata,test_cata
    global csv_file
    
    mode="test"  #train,test,validation
    train_num=100000
    test_num=10000
    val_num=20000
    snr_crt=35  
    
    pix_num=81
    pix_size=0.2
    HR_factor=5


    HR_pix_num=pix_num*HR_factor
    HR_pix_size= pix_size/HR_factor
    lenth=(HR_pix_num-1)*HR_pix_size
    
    data_dir='./data/'

    # Noise and PSFs
    noise_cata=data_dir+'real/background.csv'
    noise_dir=data_dir+'real/image/'
    psf_dir=data_dir+'real/psf/'


    #Seting the directors for saving files
    train_cata=data_dir+'train_data.csv'
    test_cata=data_dir+'test_data.csv'
    val_cata=data_dir+'validation_data.csv'
    train_img_dir=data_dir+'train'
    test_img_dir=data_dir+'test'
    val_img_dir=data_dir+'validation'


    if mode=="train":
        num_need=train_num
        galaxy_dir=train_img_dir
        if os.path.exists(galaxy_dir):
            shutil.rmtree(galaxy_dir)
        os.mkdir(galaxy_dir)
        csv_file=train_cata
    elif mode=='test':
        num_need=test_num
        galaxy_dir=test_img_dir
        if os.path.exists(galaxy_dir):
            shutil.rmtree(galaxy_dir)
        os.mkdir(galaxy_dir)
        csv_file=test_cata
    else:
        num_need=val_num
        galaxy_dir=val_img_dir
        if os.path.exists(galaxy_dir):
            shutil.rmtree(galaxy_dir)
        os.mkdir(galaxy_dir)
        csv_file=val_cata
  
#------------------Data and PSF-----------------

def noise_catalog():
    csv_file=noise_cata
    noise=[]
    psf=[]
    with open (csv_file) as f:
        reader=csv.DictReader(f)
        for row in reader:
            noise.append(row['image_name'])
            psf.append(row['psf_name'])
    return noise,psf
       
def read_noise(noise_file,psf_file):
    with pyfits.open(noise_dir+noise_file,memmap=False) as f:
        noise=f[0].data
    with pyfits.open(psf_dir+psf_file,memmap=False) as f:
        PSF=f[0].data
        
    PSF=PSF/np.sum(PSF)
    
    if random.choice([0,1])==0:   
        noise = np.flip(noise,1) 
        PSF = np.flip(PSF,1)   
    if random.choice([0,1])==0:
        noise = np.flip(noise,0)   
        PSF = np.flip(PSF,0)
    return noise,PSF

    
def save_galaxy(image,noise,PSF,fname):
    fname=galaxy_dir+'/'+fname
    hdu=pyfits.PrimaryHDU(image,header=None)
    hdu.writeto(fname)
    pyfits.append(fname,noise)
    pyfits.append(fname,PSF)

def csv_header():
    fname=csv_file
    headers = ['name','mag','xcen','ycen','R_eff','pa','q','n','SNR']
    with open(fname, 'w') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(headers)
    f.close()    

def save_csv(paras):
    fname=csv_file
    data=paras
    with open(fname, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data)
    f.close() 
    

# ----------------light-------------------

def coordinate():
    nx = HR_pix_num
    ny = HR_pix_num
    xhilo = [-lenth/2.0, lenth/2.0]
    yhilo = [-lenth/2.0, lenth/2.0]
    x = (xhilo[1] - xhilo[0]) * np.outer(np.ones(ny), np.arange(nx)) / float(nx-1) + xhilo[0]
    y = (yhilo[1] - yhilo[0]) * np.outer(np.arange(ny), np.ones(nx)) / float(ny-1) + yhilo[0]
    return x,y
    
def xy_transform(x, y, x_cen, y_cen, phi):
    xnew=(x-x_cen)*np.cos(np.pi*phi/180.0)+(y-y_cen)*np.sin(np.pi*phi/180.0)
    ynew=-(x-x_cen)*np.sin(np.pi*phi/180.0)+(y-y_cen)*np.cos(np.pi*phi/180.0)
    return (xnew, ynew)


def sersic_phot(x, y, par):
    (xnew, ynew)=xy_transform(x, y, par[1], par[2], par[4])
    n=par[6]
    if n >= 0.36: # from Ciotti & Bertin 1999, truncated to n^-3
        k=2.0*n-1./3+4./(405.*n)+46./(25515.*n**2.)+131./(1148175.*n**3.)
    else: # from MacArthur et al. 2003
        k=0.01945-0.8902*n+10.95*n**2.-19.67*n**3.+13.43*n**4.
    r=np.sqrt(xnew**2./par[5]+par[5]*ynew**2.)
    return par[0]*k**(2.0*n)/(np.pi*par[3]**2.0*sf.gamma(2.0*n+1))*np.exp(-k*(r/par[3])**(1./n))

def func_mag():
    mag_range=[16,25]
    if random.choice([0,1,3])==0:  
        mag=random.uniform(mag_range[0], mag_range[1])
    else:
        mag=mag_range[0]+mag_range[1]-(np.random.exponential(scale=1)+mag_range[0])

    if mag_range[0]<mag <mag_range[1]:
       return mag
    else:
       return func_mag()
                 
def func_Reff():
    if random.choice([0,1])==0: 
        r=np.random.normal(0.7,1)+0.7
    else:
        r=random.uniform(0.2, 4)
        
    if 0.2< r <4:
       return r
    else:
       return func_Reff()
       
def func_q():
    q=np.random.normal(0.8,0.3)
    if 0.2< q <1:
       return q
    else:
       return func_q()

def func_nser():
    n=np.random.f(20,3)*2-0.2
    if 0.1< n <8:
       return n
    else:
       return func_nser()
       
def phot_parameters():
    mag=func_mag()
    m0 =31.4
    phot_flux=10**((mag-m0)/(-2.5))
    
    phot_xcen  = random.uniform(-0.4, 0.4)
    phot_ycen  = random.uniform(-0.4, 0.4)
    phot_R_eff = func_Reff()
    phot_pa    = random.uniform(0.0, 180.0)
    phot_axrat = func_q()
    phot_nser  = func_nser()
    phot_para = np.array([phot_flux, phot_xcen, phot_ycen,\
                             phot_R_eff, phot_pa, phot_axrat, phot_nser])
    return phot_para,mag
 
 
def cal_std(img): 
    img_pix=len(img)
    pix=5
    img1=img[0:pix,0:pix]
    img2=img[img_pix-pix:img_pix,0:pix]
    img3=img[img_pix-pix:img_pix,img_pix-pix:img_pix]
    img4=img[0:pix,img_pix-pix:img_pix]
    pixcen=int((img_pix-1)/2)
    img5=img[pixcen-3:pixcen+2,pixcen-3:pixcen+2]
    imgs=[img1,img2,img3,img4,img5]
    img_new=[]
    for i in range(len(imgs)):
        if np.mean(imgs[i])>10:
            continue
        img_new.append(imgs[i])
    img_new=np.array(img_new)
    img_new=img_new.flatten()
    mean=np.mean(img_new)
    std=np.std(img_new)
    return std
        
#-----------------SNR calculation-----------------

def SNfunc(data,sig,sig_raw,significancefloor=3):

    data=data*10**2 
    sig=sig*10**2 
    sig_raw=sig_raw*10**2  
    D=data.ravel()   
    S=sig.ravel()
    
    std=cal_std(sig_raw) 
    indices = np.where((sig< std) & (sig>-std)) 
    sig[indices] = std

    
    data[abs(data/sig)<significancefloor]=0   
    data=np.array(data, dtype="<f4")

    masks, multiplicity = ndimage.label(data)    
    labels=np.arange(1, multiplicity+1)
    SNs=np.zeros(multiplicity+1)  

    for i in range(multiplicity):
        D=data[masks==i+1].ravel()
        S=sig[masks==i+1].ravel()
        args=np.argsort(-D/S)  
        D=np.take(D,args) 
        S=np.take(S,args)
        Dsum=np.cumsum(D) 
        Ssum=np.cumsum(S**2)**0.5 
        SNi=(Dsum/Ssum).max() 
        SNs[i]=SNi

    SNs=-np.sort(-SNs)
    SNR=np.max(SNs)  
    return SNR

#---------------Galaxy Simulation----------------- 
    
def simulate(num,x,y,noise,PSF,a):
        
    phot_para,mag=phot_parameters()
    phot = sersic_phot(x, y, phot_para)
    phot=resize(phot,(pix_num,pix_num))*pix_size**2
    phot=signal.fftconvolve (phot, PSF, mode = 'same')


    pix1=int(40-(pix_num-1)/2)
    pix2=int(40+(pix_num-1)/2+1)
    noise_cut=noise[pix1:pix2,pix1:pix2]
    
    galaxy=noise_cut+phot
    
    snr=SNfunc(galaxy,noise_cut, noise)

    if snr>=snr_crt and a<300:          
        fname=str(num)+'.fits'
        save_galaxy(galaxy,noise,PSF,fname)
        
        for j in range(len(phot_para)-1):
            phot_para[j+1]=np.round(phot_para[j+1],4)
            
        paras=[fname,np.round(mag,4),phot_para[1],phot_para[2],phot_para[3],\
                 phot_para[4],phot_para[5],phot_para[6],np.round(snr,4)]     
        save_csv(paras)
    elif snr<snr_crt and a<300:
        a=a+1
        return simulate(num,x,y,noise,PSF,a)
    else:
        print("Can not get a galaxy with high SNR,noise number is:",num)


#---------------Dataset Generation----------------- 

def process(x,y):
    
    noise_file,psf_file=noise_catalog()

    num_noise=len(noise_file)
    for i in range(num_need):
        noise_index=np.random.randint(0, num_noise)
        noise,PSF=read_noise(noise_file[noise_index],psf_file[noise_index])
        if i%1000==0:
            print ('galaxy simulation finished:', np.round(100.0*i/num_need,5),"%")
        a=0
        simulate(i,x,y,noise,PSF,a)


#---------------Main----------------- 

if __name__ == '__main__':
    set_parameters()
    csv_header()  
    x,y=coordinate()
    process(x,y)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




