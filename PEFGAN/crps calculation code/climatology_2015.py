# 计算climatology for validation for nci

# from scipy.stats import norm
# ps.crps_ensemble(obs,ens).shape
from os import mkdir
import os
from datetime import timedelta, date, datetime
import properscoring as ps
import numpy as np
import data_processing_tool as dpt
import sys

sys.path.append('../')


def rmse(ens, hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.sqrt(((ens - hr) ** 2).sum(axis=(0)) / ens.shape[0])


def mae(ens, hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.abs((ens - hr)).sum(axis=0) / ens.shape[0]


def mae_mean(ens, hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.abs((ens.mean(axis=0) - hr))


def mae_median(ens, hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.abs((np.median(ens, axis=0) - hr))


def bias(ens, hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return (ens - hr).sum(axis=0) / ens.shape[0]


def bias_median(ens, hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.median(ens, axis=0) - hr


def bias_relative(ens, hr, constant=1):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    constant: relative constant
    '''
    return (np.mean(ens, axis=0) - hr) / (constant + hr)


def bias_relative_median(ens, hr, constant=1):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    constant: relative constant
    '''
    return (np.median(ens, axis=0) - hr) / (constant + hr)

def calAWAPprob(AWAP_data, percentile):
    ''' 
    input: AWAP_data is  413 * 267
            percentile size is 413 * 267
    return: A probability matrix which size is 413 * 267 indicating the probability of the values in ensemble forecast 
    is greater than the value in the same pixel in percentile matrix

    '''
    
    return (AWAP_data > percentile) * 1

def calforecastprob(forecast, percentile):
    ''' 
    input: forecast is  9 * 413 * 267
            percentile size is 413 * 267
    return: A probability matrix which size is 413 * 267 indicating the probability of the values in ensemble forecast 
    is greater than the value in the same pixel in percentile matrix

    '''
    
    prob_matrix = (forecast > percentile)
    return np.mean(prob_matrix, axis = 0)

def calAWAPdryprob(AWAP_data, percentile):

    return (AWAP_data >= percentile) * 1

def calforecastdryprob(forecast, percentile):

    prob_matrix = (forecast >= percentile)
    return np.mean(prob_matrix, axis = 0)   
    

def main(year, time_windows):
    Brier_startyear = 1981
    Brier_endyear = 2010
    percentile_95 = dpt.AWAPcalpercentile(Brier_startyear, Brier_endyear, 95)
    percentile_99 = dpt.AWAPcalpercentile(Brier_startyear, Brier_endyear, 99)
    dates_needs = dpt.date_range(date(year, 1, 1), date(year + 1, 7, 29))
    file_BARRA_dir = "/scratch/iu60/yl3101/AGCD_mask_data/"
    #     date_map=np.array(dates_needs)
    # np.where(date_map==date(2012, 1, 1))
    crps_ref = []
    log_crps_ref = []
    Brier_0 = []
    Brier95 = []
    Brier99 = []
    mae_score = []
    bias_ref = []
    bias_median_ref = []
    mae_median_ref = []
    bias_relative_ref = []
    mean_bias_relative_model_half = []
    mean_bias_relative_model_1 = []
    mean_bias_relative_model_2 = []
    mean_bias_relative_model_2d9 = []
    mean_bias_relative_model_3 = []
    mean_bias_relative_model_4 = []

    for target_date in dates_needs:
        hr, Awap_date = dpt.read_awap_data_fc(file_BARRA_dir, target_date)
        hr_log, Awap_date_log = dpt.read_awap_data_fc_log(file_BARRA_dir, target_date)
        print('hr: ', hr)
        print('log hr: ', hr_log)
        ensamble = []
        log_ensamble = []
        #         for y in range(1990,target_date.year):
        for y in range(1981, 2010):

            if target_date.year == y:
                continue

            for w in range(1, time_windows):  # for what time of windows

                filename = file_BARRA_dir + \
                           str(y) + (target_date - timedelta(w)).strftime("-%m-%d") + ".nc"
                if os.path.exists(filename):
                    t = date(y, (target_date - timedelta(w)).month,
                             (target_date - timedelta(w)).day)
                    sr = dpt.read_awap_data_fc(file_BARRA_dir, t)
                    sr_log = dpt.read_awap_data_fc_log(file_BARRA_dir, t)
                    ensamble.append(sr)
                    log_ensamble.append(sr_log)

                filename = file_BARRA_dir + \
                           str(y) + (target_date + timedelta(w)).strftime("-%m-%d") + ".nc"
                if os.path.exists(filename):
                    t = date(y, (target_date + timedelta(w)).month,
                             (target_date + timedelta(w)).day)

                    sr = dpt.read_awap_data_fc(file_BARRA_dir, t)
                    sr_log = dpt.read_awap_data_fc_log(file_BARRA_dir, t)
                    ensamble.append(sr)
                    log_ensamble.append(sr_log)

            filename = file_BARRA_dir + str(y) + target_date.strftime("-%m-%d") + ".nc"
            if os.path.exists(filename):
                t = date(y, target_date.month, target_date.day)
                print(t)
                sr, temp_date = dpt.read_awap_data_fc(file_BARRA_dir, t)
                sr_log, temp_date_log = dpt.read_awap_data_fc_log(file_BARRA_dir, t)
                print('sr: ', sr)
                print('sr log: ', sr_log)
                ensamble.append(sr)
                log_ensamble.append(sr_log)
        if ensamble:
            print("calculate ensemble")
            ensamble = np.array(ensamble)
            print("ensemble.shape:", ensamble.shape)
            print("hr.shape:", hr.shape)
            print("date:", Awap_date)
            a = ps.crps_ensemble(hr, ensamble.transpose(1, 2, 0))
            prob_awap0 = calAWAPdryprob(hr, 0.1)
            prob_awap95 = calAWAPprob(hr, percentile_95)
            prob_awap99 = calAWAPprob(hr, percentile_99)
            prob_ensamble0 = calforecastdryprob(ensamble, 0.1)
            prob_ensamble95 = calforecastprob(ensamble, percentile_95)
            prob_ensamble99 = calforecastprob(ensamble, percentile_99)
            climat_mae = mae(ensamble, hr)
            # mae_median_score = mae_median(ensamble, hr)
            bias_score = bias(ensamble, hr)
            # bias_median_score = bias_median(ensamble, hr)
            # bias_relative_half = bias_relative_median(ensamble, hr, constant=0.5)
            # bias_relative_1 = bias_relative_median(ensamble, hr, constant=1)
            # bias_relative_2 = bias_relative_median(ensamble, hr, constant=2)
            # bias_relative_2d9 = bias_relative_median(ensamble, hr, constant=2.9)
            bias_relative_3 = bias_relative(ensamble, hr, constant=3)
            # bias_relative_4 = bias_relative_median(ensamble, hr, constant=4)
            #
            Brier_0.append((prob_awap0 - prob_ensamble0) ** 2)
            Brier95.append((prob_awap95 - prob_ensamble95) ** 2)
            Brier99.append((prob_awap99 - prob_ensamble99) ** 2)
            crps_ref.append(a)
            mae_score.append(climat_mae)
            # # mae_median_ref.append(mae_median_score)
            bias_ref.append(bias_score)
            # # bias_median_ref.append(bias_median_score)
            # mean_bias_relative_model_half.append(bias_relative_half)
            # mean_bias_relative_model_1.append(bias_relative_1)
            # mean_bias_relative_model_2.append(bias_relative_2)
            # mean_bias_relative_model_2d9.append(bias_relative_2d9)
            mean_bias_relative_model_3.append(bias_relative_3)
            # mean_bias_relative_model_4.append(bias_relative_4)
        if log_ensamble:
            print("calculate ensemble")
            log_ensamble = np.array(log_ensamble)
            print("log ensemble.shape:", log_ensamble.shape)
            print("hr.shape:", hr_log.shape)
            print("date:", Awap_date_log)
            a_log = ps.crps_ensemble(hr_log, log_ensamble.transpose(1, 2, 0))
            log_crps_ref.append(a_log)
    # CRPS-SS
    # if not os.path.exists('./save/crps_ss/climatology/'+str(year)):
    #     mkdir('./save/crps_ss/climatology/'+str(year))

    np.save("/scratch/iu60/yl3101/PEFGAN/new_crps/save/climatology/climatology_" + str(year) +
            "_all_lead_time_windows_" + str((time_windows - 1) * 2 + 1), np.array(crps_ref))
    np.save("/scratch/iu60/yl3101/PEFGAN/new_crps/save/climatology/log_climatology_" + str(year) +
                "_all_lead_time_windows_" + str((time_windows - 1) * 2 + 1), np.array(log_crps_ref))
    np.save("/scratch/iu60/yl3101/PEFGAN/new_crps/save/climatology/prob0_climatology_" + str(year) +
                "_all_lead_time_windows_" + str((time_windows - 1) * 2 + 1), np.array(Brier_0))
    np.save("/scratch/iu60/yl3101/PEFGAN/new_crps/save/climatology/prob95_climatology_" + str(year) +
                "_all_lead_time_windows_" + str((time_windows - 1) * 2 + 1), np.array(Brier95))
    np.save("/scratch/iu60/yl3101/PEFGAN/new_crps/save/climatology/prob99_climatology_" + str(year) +
                "_all_lead_time_windows_" + str((time_windows - 1) * 2 + 1), np.array(Brier99))
    np.save("/scratch/iu60/yl3101/PEFGAN/new_crps/save/climatology/mae_climatology_" + str(year) +
                "_all_lead_time_windows_" + str((time_windows - 1) * 2 + 1), np.array(mae_score))
    np.save("/scratch/iu60/yl3101/PEFGAN/new_crps/save/climatology/relative_bias_climatology_" + str(year) +
                "_all_lead_time_windows_" + str((time_windows - 1) * 2 + 1), np.array(mean_bias_relative_model_3))
    np.save("/scratch/iu60/yl3101/PEFGAN/new_crps/save/climatology/bias_climatology_" + str(year) +
                "_all_lead_time_windows_" + str((time_windows - 1) * 2 + 1), np.array(bias_ref))
    # MAE
    # if not os.path.exists('./save/mae_median/climatology/'+str(year)):
    #     mkdir('./save/mae_median/climatology/'+str(year))

    # np.save("./save/mae_median/climatology/climatology_" + str(year) +
    #         "_all_lead_time_windows_" + str((time_windows - 1) * 2 + 1), np.array(mae_median_ref))

    # Bias
    # if not os.path.exists('./save/bias/climatology/'+str(year)):
    #     mkdir('./save/bias_median/climatology/'+str(year))

    # np.save("./save/bias/climatology/climatology_" + str(year) +
    #         "_all_lead_time_windows_" + str((time_windows - 1) * 2 + 1), np.array(bias_ref))

    # Bias Median
    # if not os.path.exists('./save/bias_median/climatology/'+str(year)):
    #     mkdir('./save/bias_median/climatology/'+str(year))

    # np.save("./save/bias_median/climatology/climatology_" + str(year) +
    #        "_all_lead_time_windows_" + str((time_windows - 1) * 2 + 1), np.array(bias_median_ref))

    # Bias relative

    # if not os.path.exists('./save/bias_relative/climatology/' + str(year)):
    #    mkdir('./save/bias_relative/climatology/' + str(year))

    # if not os.path.exists('./save/bias_relative/climatology/' + str(year) + '/window' + str(i)):
    #    mkdir('./save/bias_relative/climatology/' + str(year) + '/window' + str(i))

    # np.save("./save/bias_relative/climatology/climatology_" + str(year) +
    #        "_all_lead_time_windows_" + str((time_windows - 1) * 2 + 1), np.array(bias_relative_ref))

    # np.save("./save/bias_relative_median/0.5/climatology/climatology_" + str(year) +
    #         "_all_lead_time_windows_" + str((time_windows - 1) * 2 + 1), np.array(mean_bias_relative_model_half))
    #
    # np.save("./save/bias_relative_median/1/climatology/climatology_" + str(year) +
    #         "_all_lead_time_windows_" + str((time_windows - 1) * 2 + 1), np.array(mean_bias_relative_model_1))
    #
    # np.save("./save/bias_relative_median/2/climatology/climatology_" + str(year) +
    #         "_all_lead_time_windows_" + str((time_windows - 1) * 2 + 1), np.array(mean_bias_relative_model_2))
    #
    # np.save("./save/bias_relative_median/2.9/climatology/climatology_" + str(year) +
    #         "_all_lead_time_windows_" + str((time_windows - 1) * 2 + 1), np.array(mean_bias_relative_model_2d9))
    #
    # np.save("/scratch/iu60/yl3101/DESRGAN/new_crps/save/climatology/" + str(year) +
    #         "_all_lead_time_windows_" + str((time_windows - 1) * 2 + 1), np.array(mean_bias_relative_model_3))
    #
    # np.save("./save/bias_relative_median/4/climatology/climatology_" + str(year) +
    #         "_all_lead_time_windows_" + str((time_windows - 1) * 2 + 1), np.array(mean_bias_relative_model_4))


if __name__ == '__main__':
    year_list = [2015]
    timewind = [1]
    for i in timewind:
        for j in year_list:
            main(j, i)
            print(str(j) + ", " + str(i * 2 - 1))
