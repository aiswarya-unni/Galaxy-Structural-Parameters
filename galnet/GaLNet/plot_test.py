#! -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import csv

#-----------------Set Parameters-----------------

def set_parameters():
    global keywords,x_limit,display    
    keywords=["x","y","mag","R_eff","e1","e2","n"]
    display = ["x","y","mag", r"$R_e$", "e1", "e2", "n"]
    x_limit=[[-0.41,0.41],[-0.41,0.41],[15.8,25],[0.1,4.1],[-0.7,0.7],[-0.7,0.7],[0.4,8.1]]  #


#-----------------Read Data-----------------

def read_data():
    para_true=[]
    para_cnn=[]
    csv_file=f'./GaLNet/result/pred_test_para.csv'
    with open (csv_file) as f:
        reader=csv.DictReader(f)
        for row in reader:
            mag_t=float(row['mag_true'])
            re_t =float(row['re_true'])
            if mag_t<25 and re_t<4:  
                x=float(row['x_true'])
                y=float(row['y_true'])
                mag=float(row['mag_true'])
                re=float(row['re_true'])
                e1=float(row['e1_true'])
                e2=float(row['e2_true'])
                n=float(row['n_true'])
                para_true.append([x,y,mag,re,e1,e2,n])
            
                x=float(row['x_cnn'])
                y=float(row['y_cnn'])
                mag=float(row['mag_cnn'])
                re=float(row['re_cnn'])
                e1=float(row['e1_cnn'])
                e2=float(row['e2_cnn'])
                n=float(row['n_cnn'])
                para_cnn.append([x,y,mag,re,e1,e2,n])
    return para_true,para_cnn


#---------------Plotting-----------------

def plot_compare_fig(y_true, y_pred):
    y_limit = x_limit

    for i in range(len(keywords)):
        para_true = []
        para_pred = []
        for j in range(len(y_pred)):
            if abs((y_true[j])[0] - (y_pred[j])[0]) < 5:
                para_true.append((y_true[j])[i])
                para_pred.append((y_pred[j])[i])

        para_true = np.array(para_true)
        para_pred = np.array(para_pred)

        xlim = x_limit[i]
        keyword = keywords[i]
        keyword = display[i]
        a = (xlim[0] + xlim[1]) / 2 - (xlim[1] - xlim[0]) * 0.1
        b = xlim[1] - (xlim[1] - xlim[0]) * 0.07
        text_loc = [a, b]

        # --- Calculate Metrics ---
        # Bias
        bias = np.mean(np.abs(para_pred - para_true))
        bias = np.abs(bias)

        # R^2
        ss_res = np.sum((para_pred - para_true) ** 2)
        ss_tot = np.sum((para_true - np.mean(para_true)) ** 2)
        r_squared = np.abs(1 - (ss_res / ss_tot))

        # RMSE
        rmse = np.sqrt(np.mean((para_pred - para_true) ** 2))

        # MAE
        mae = np.mean(np.abs(para_pred - para_true))

        # NMAD
        delta_p = (para_pred - para_true) / (1 + para_true)
        nmad = 1.4826 * np.median(np.abs(delta_p - np.median(delta_p)))

        # Outlier Fraction
        threshold = 0.15
        discrepancies = np.abs(para_pred - para_true) / (1 + para_true)
        outliers_count = np.sum(discrepancies > threshold)
        outlier_fraction = outliers_count / len(para_true)

        # Bin Statistics
        errors = np.abs(para_true - para_pred)
        num_bins = 8
        bins = np.linspace(xlim[0], xlim[1], num_bins + 1)
        quantiles = np.linspace(0, 100, num_bins + 1)
        #bins = np.percentile(para_true, quantiles)


        bin_median_true = []
        bin_median_pred = []
        bin_errors = []
        bin_maes = []

        for k in range(num_bins):
            bin_indices = (para_true >= bins[k]) & (para_true < bins[k + 1])
            if np.any(bin_indices):
                bin_true_median = np.median(para_true[bin_indices])
                bin_pred_median = np.median(para_pred[bin_indices])
                bin_pred_std = np.std(para_true[bin_indices])

                bin_median_true.append(bin_true_median)
                bin_median_pred.append(bin_pred_median)
                bin_errors.append(bin_pred_std)
                bin_maes.append(np.median(np.abs(para_pred[bin_indices] - para_true[bin_indices])))

        bin_median_true = np.array(bin_median_true)
        bin_median_pred = np.array(bin_median_pred)
        bin_errors = np.array(bin_errors)

        fontsize=25
        fig = plt.figure(figsize=(10, 8))
        plt.plot(xlim, xlim, "--", lw=2, color='black', alpha=0.4)
        plt.plot(para_true, para_pred, '.', lw=1, markersize=2, alpha=0.4, zorder=1)
        plt.errorbar(bin_median_true, bin_median_pred, yerr=bin_maes, fmt='o', capsize=5, color='r', capthick=2, alpha=0.6, zorder=1)
        plt.text(text_loc[0], text_loc[1], keyword, color="black", fontsize=fontsize)

        for idx, mae in enumerate(bin_maes):
            plt.text(bin_median_true[idx], bin_median_pred[idx] + 0.05, f"{mae:.3f}", fontsize=20, color='black', ha='right')

        # ---------------- Add Stats to Plot -----------------
        stats_text = (
            f"$Bias = {bias:.2f}$\n"
            f"$R^2 = {r_squared:.2f}$\n"
            f"RMSE = {rmse:.2f}\n"
            f"NMAD = {nmad:.2f}"
        )
        plt.text(xlim[0] + 0.05 * (xlim[1] - xlim[0]), 
                 xlim[1] - 0.25 * (xlim[1] - xlim[0]), 
                 stats_text, fontsize=fontsize, color="dodgerblue",
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

        plt.xlabel('TRUE', fontsize=fontsize)
        plt.ylabel('GALNET', fontsize=fontsize)
        plt.tight_layout()

        plt.xlim(xlim[0], xlim[1])
        plt.ylim(xlim[0], xlim[1])

        plt.tick_params(labelsize=fontsize)
        fname = f'./GaLNet/fig/' + 'test_' + keywords[i] + '.png'
        plt.savefig(fname,bbox_inches='tight')
        plt.close() 


#---------------Main----------------- 

if __name__ == '__main__':  
    set_parameters()          
    y_true,y_pred= read_data()
    plot_compare_fig(y_true,y_pred)
    

