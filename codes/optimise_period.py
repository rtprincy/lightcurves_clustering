from astropy.timeseries import LombScargle
import numpy as np
import pandas as pd

import time
from astropy.io import fits, ascii
from astropy.table import Table, vstack

import os
import glob

import tqdm

from lk_stat_package import lk_stat

def freq_grid(times,oversample_factor=10,f0=None,fn=None):
    times=np.sort(times)
    df = 1.0 / (times.max() - times.min())
    if f0 is None:
        f0 = df
    if fn is None:
        fn = 0.5 / np.median(np.diff(times)) 
    return np.arange(f0, fn, df / oversample_factor)


data=pd.read_csv("ms_wd_targets_summary_stat_N_G_BP_RP_25.csv")

Nsample=1000

ids=data['source_id'].values


path=save_ls_to="project_obj_btw_MS_WD/gaia_periodograms/"

oversampling_factor=10

passbands=['G','BP','RP']
t_int=4.4/86400

for band in passbands:
    
    opt_period_col='opt_period_%s'%(band)

    data[opt_period_col]=[np.nan]*len(data)
    
    for source in tqdm.tqdm(new_ids):
        file_path='/idia/users/princy/project_obj_btw_MS_WD/gaia_lightcurves/%d.csv'%(source)
        lsp_path=path+"lsp/"+str(source)+'_%s.npy'%(band)
        theta_path=path+"theta/"+str(source)+'_%s.npy'%(band)
        df=pd.read_csv(file_path)
        df=df[~df['variability_flag_g_reject']] 

        

        if (df.shape[0]>=24)&(os.path.exists(lsp_path)):     
            mag=df['g_transit_mag'].dropna().values
            Time=df['g_transit_time'].dropna().values
   
            flux=df['g_transit_flux'].dropna().values
            flux_err=df['g_transit_flux_error'].dropna().values
            
            mag_err= (2.5/np.log(10))*(flux_err / flux)
    
            freq, lsp=np.load(lsp_path)
            theta=np.load(theta_path)
            
            psi=2*lsp/theta
            
            ################### Optimise dominant frequency ##################################################
            
            # This is to optimise the frequency search around the frequency peak by oversampling the frequency grid. For e.g., if the peak is at 2 c/d, we search for the best frequency from 1.999 to 2.001 using a small frequency step.
    
            idx_peak=np.argmax(psi)
            
            f_step=np.diff(freq)[0]
    
            peak_freq=freq[idx_peak]
            
            #Here we take 10 steps before and after the frequency peak as a new frequency search range. 
            
            lower_range=max(freq.min(),peak_freq - (oversampling_factor*f_step))
            upper_range=min(freq.max(),peak_freq + (oversampling_factor*f_step))
    
    
    
            fine_grid_freq=freq_grid(Time,oversample_factor=100,f0=lower_range,fn=upper_range)
         
        
            lsp_fg = LombScargle(t=Time, y=mag, dy=mag_err,nterms=1).power(frequency=fine_grid_freq, method="cython", normalization="psd")
            theta_fg= theta_f(1/fine_grid_freq, mag, mag_err, Time)
        
            psi_fine_grid=2*lsp_fg/theta_fg
        
            best_freq=fine_grid_freq[np.argmax(psi_fine_grid)]
            
            best_freq=fine_grid_freq[np.argmax(lsp_fg)]
    
           
        
                
            best_period=1/best_freq
      
            
            
            data[opt_period_col][data['source_id']==int(source)]=best_period

        

data.to_csv("ms_wd_targets_summary_stat_N_G_BP_RP_25_optperiod_GBPRP.csv",index=None)
