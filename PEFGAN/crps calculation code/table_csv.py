import csv
from datetime import timedelta, date, datetime
import os
import sys
import numpy as np

year = 2015
window = 1
# norm = "bias_relative"  # ["crps_ss", "mae_median", "bias", "bias_median"]
norm = "crps_ss"  # ["crps_ss", "mae_median", "bias", "bias_median"]
model_name = 'vdynamic_weights'
model_num = 'model_G_i000008'

class aaa(object):
    def __init__(self, lead):
        self.ensemble_access = ['e01', 'e02', 'e03', 'e04',
                                'e05', 'e06', 'e07', 'e08', 'e09']
        self.lead_time = lead 
        self.rootdir = '/scratch/iu60/yl3101/PEFGAN/' + model_name + '/' + str(year) + '/'+ model_num + '/'
        self.files = self.get_filename_with_time_order(self.rootdir)
        print("Including the files: ", self.files)
    def get_filename_with_time_order(self, rootdir):
        _files = []
        # Initial dates for ACCESS-S2
        date_dict = {1: [1, 14, 15, 16, 30, 31], 2: [1, 14, 15, 16, 27, 28], 3: [1, 14, 15, 16, 30, 31],
                     4: [1, 14, 15, 16, 29, 30], 5: [1, 14, 15, 16, 30, 31], 6: [1, 14, 15, 16, 29, 30],
                     7: [1, 14, 15, 16, 30, 31], 8: [1, 14, 15, 16, 30, 31], 9: [1, 14, 15, 16, 29, 30],
                     10: [1, 14, 15, 16, 30, 31], 11: [1, 14, 15, 16, 29, 30], 12: [1, 14, 15, 16, 30, 31]}
        for mm in range(1, 13):
            for dd in date_dict[mm]:
                date_time = date(year, mm, dd)
                access_path = rootdir + "e09" + "/" + \
                              date_time.strftime("%Y-%m-%d") + "_" + "e09" + ".nc"
                #                   print(access_path)
                if os.path.exists(access_path):
                    #                 for i in range(self.lead_time,self.lead_time+1):
                    #                 for en in ['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']:
                    path = []
                    awap_date = date_time + timedelta(self.lead_time)
                    path.append(date_time)
                    path.append(awap_date)
                    path.append(self.lead_time)
                    _files.append(path)
        return _files

    def __getitem__(self, idx):
        return self.files[idx]


def load_climatology_data(lead_time):
    data = aaa(lead_time)
    climtology_lead_time = []
    climatology_data = np.load(
        '/scratch/iu60/yl3101/DESRGAN/new_crps/save/climatology/climatology_' + str(
            year) + '_all_lead_time_windows_' + str(window) + '.npy')
    dates_needs = date_range(date(year, 1, 1), date(year + 1, 7, 29))
    date_map = np.array(dates_needs)
    print("time interval:", len(date_map))
    print("The including files:", data.files)
    for _, target_date, _ in data.files:
        # print(target_date)
        idx = np.where(date_map == target_date)[0]
        print('corresponding index', idx)
        # print("AWAP pr:", climatology_data[idx])
        # climtology_lead_time.append(climatology_data[idx][0])
        climtology_lead_time.append(climatology_data[idx])
    return np.array(climtology_lead_time, dtype=object)

def load_climatology_data_log(lead_time):
    data = aaa(lead_time)
    climtology_lead_time = []
    climatology_data = np.load(
        '/scratch/iu60/yl3101/DESRGAN/new_crps/save/climatology/log_climatology_' + str(
            year) + '_all_lead_time_windows_' + str(window) + '.npy')
    dates_needs = date_range(date(year, 1, 1), date(year + 1, 7, 29))
    date_map = np.array(dates_needs)
    print("time interval:", len(date_map))
    print("The including files:", data.files)
    for _, target_date, _ in data.files:
        # print(target_date)
        idx = np.where(date_map == target_date)[0]
        print('corresponding index', idx)
        # print("AWAP pr:", climatology_data[idx])
        # climtology_lead_time.append(climatology_data[idx][0])
        climtology_lead_time.append(climatology_data[idx])
    return np.array(climtology_lead_time, dtype=object)

def load_prob0_data(lead_time):
    data = aaa(lead_time)
    prob0_lead_time = []
    prob0_data = np.load(
        '/scratch/iu60/yl3101/DESRGAN/new_crps/save/climatology/prob0_climatology_' + str(
            year) + '_all_lead_time_windows_' + str(window) + '.npy')
    dates_needs = date_range(date(year, 1, 1), date(year + 1, 7, 29))
    date_map = np.array(dates_needs)
    print("time interval:", len(date_map))
    print("The including files:", data.files)
    for _, target_date, _ in data.files:
        # print(target_date)
        idx = np.where(date_map == target_date)[0]
        print('corresponding index', idx)
        # print("AWAP pr:", climatology_data[idx])
        # climtology_lead_time.append(climatology_data[idx][0])
        prob0_lead_time.append(prob0_data[idx])
    return np.array(prob0_lead_time, dtype=object)

def load_prob95_data(lead_time):
    data = aaa(lead_time)
    prob95_lead_time = []
    prob95_data = np.load(
        '/scratch/iu60/yl3101/DESRGAN/new_crps/save/climatology/prob95_climatology_' + str(
            year) + '_all_lead_time_windows_' + str(window) + '.npy')
    dates_needs = date_range(date(year, 1, 1), date(year + 1, 7, 29))
    date_map = np.array(dates_needs)
    print("time interval:", len(date_map))
    print("The including files:", data.files)
    for _, target_date, _ in data.files:
        # print(target_date)
        idx = np.where(date_map == target_date)[0]
        print('corresponding index', idx)
        # print("AWAP pr:", climatology_data[idx])
        # climtology_lead_time.append(climatology_data[idx][0])
        prob95_lead_time.append(prob95_data[idx])
    return np.array(prob95_lead_time, dtype=object)

def load_prob99_data(lead_time):
    data = aaa(lead_time)
    prob99_lead_time = []
    prob99_data = np.load(
        '/scratch/iu60/yl3101/DESRGAN/new_crps/save/climatology/prob99_climatology_' + str(
            year) + '_all_lead_time_windows_' + str(window) + '.npy')
    dates_needs = date_range(date(year, 1, 1), date(year + 1, 7, 29))
    date_map = np.array(dates_needs)
    print("time interval:", len(date_map))
    print("The including files:", data.files)
    for _, target_date, _ in data.files:
        # print(target_date)
        idx = np.where(date_map == target_date)[0]
        print('corresponding index', idx)
        # print("AWAP pr:", climatology_data[idx])
        # climtology_lead_time.append(climatology_data[idx][0])
        prob99_lead_time.append(prob99_data[idx])
    return np.array(prob99_lead_time, dtype=object)

def load_mae_data(lead_time):
    data = aaa(lead_time)
    mae_lead_time = []
    mae_data = np.load(
        '/scratch/iu60/yl3101/DESRGAN/new_crps/save/climatology/mae_climatology_' + str(
            year) + '_all_lead_time_windows_' + str(window) + '.npy')
    dates_needs = date_range(date(year, 1, 1), date(year + 1, 7, 29))
    date_map = np.array(dates_needs)
    print("time interval:", len(date_map))
    print("The including files:", data.files)
    for _, target_date, _ in data.files:
        # print(target_date)
        idx = np.where(date_map == target_date)[0]
        print('corresponding index', idx)
        # print("AWAP pr:", climatology_data[idx])
        # climtology_lead_time.append(climatology_data[idx][0])
        mae_lead_time.append(mae_data[idx])
    return np.array(mae_lead_time, dtype=object)

def load_bias_data(lead_time):
    data = aaa(lead_time)
    bias_lead_time = []
    bias_data = np.load(
        '/scratch/iu60/yl3101/DESRGAN/new_crps/save/climatology/bias_climatology_' + str(
            year) + '_all_lead_time_windows_' + str(window) + '.npy')
    dates_needs = date_range(date(year, 1, 1), date(year + 1, 7, 29))
    date_map = np.array(dates_needs)
    print("time interval:", len(date_map))
    print("The including files:", data.files)
    for _, target_date, _ in data.files:
        # print(target_date)
        idx = np.where(date_map == target_date)[0]
        print('corresponding index', idx)
        # print("AWAP pr:", climatology_data[idx])
        # climtology_lead_time.append(climatology_data[idx][0])
        bias_lead_time.append(bias_data[idx])
    return np.array(bias_lead_time, dtype=object)

def load_relative_bias_data(lead_time):
    data = aaa(lead_time)
    relative_bias_lead_time = []
    relative_bias_data = np.load(
        '/scratch/iu60/yl3101/DESRGAN/new_crps/save/climatology/relative_bias_climatology_' + str(
            year) + '_all_lead_time_windows_' + str(window) + '.npy')
    dates_needs = date_range(date(year, 1, 1), date(year + 1, 7, 29))
    date_map = np.array(dates_needs)
    print("time interval:", len(date_map))
    print("The including files:", data.files)
    for _, target_date, _ in data.files:
        # print(target_date)
        idx = np.where(date_map == target_date)[0]
        print('corresponding index', idx)
        # print("AWAP pr:", climatology_data[idx])
        # climtology_lead_time.append(climatology_data[idx][0])
        relative_bias_lead_time.append(relative_bias_data[idx])
    return np.array(relative_bias_lead_time, dtype=object)

def date_range(start_date, end_date):
    """This function takes a start date and an end date as datetime date objects.
    It returns a list of dates for each date in order starting at the first date and ending with the last date"""
    return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]


def calculate_csv_file():
    print(str(year) + ", " + str(window) + ", " + str(norm))
    for lead_time in range(217):
        climat = load_climatology_data(lead_time)
        climat_log = load_climatology_data_log(lead_time)
        prob_0 = load_prob0_data(lead_time)
        prob95 = load_prob95_data(lead_time)
        prob99 = load_prob99_data(lead_time)
        mae_score = load_mae_data(lead_time)
        bias_score = load_bias_data(lead_time)
        realtive_bias_score = load_relative_bias_data(lead_time)
        print('total num of date:', climat.shape)
        climat = climat.mean(axis=0)
        climat_log = climat_log.mean(axis=0)
        prob_0 = prob_0.mean(axis=0)
        prob95 = prob95.mean(axis=0)
        prob99 = prob99.mean(axis=0)
        mae_score = mae_score.mean(axis=0)
        bias_score = bias_score.mean(axis=0)
        realtive_bias_score = realtive_bias_score.mean(axis=0)

        filename = 'climat_lead_time_' + str(lead_time)
        filename_log = 'log_climat_lead_time_' + str(lead_time)
        filename_0 = 'prob0_climat_lead_time_'+ str(lead_time)
        filename_95 = 'prob95_climat_lead_time_' + str(lead_time)
        filename_99 = 'prob99_climat_lead_time_' + str(lead_time)
        file_mae = 'mae_climat_lead_time_' + str(lead_time)
        file_bias = 'bias_climat_lead_time_' + str(lead_time)
        file_relative_bias = 'relative_bias_climat_lead_time_' + str(lead_time)

        if not os.path.exists("/scratch/iu60/yl3101/DESRGAN/new_crps/save/" + norm + "/mean_climatology/" + str(year)):
            os.mkdir("/scratch/iu60/yl3101/DESRGAN/new_crps/save/" + norm + "/mean_climatology/" + str(year))
            
        if not os.path.exists("/scratch/iu60/yl3101/DESRGAN/new_crps/save/" + norm + "/mean_climatology/" + str(year) + "/window" + str(
            window)):
            os.mkdir("/scratch/iu60/yl3101/DESRGAN/new_crps/save/" + norm + "/mean_climatology/" + str(year) + "/window" + str(
            window))

        np.save("/scratch/iu60/yl3101/DESRGAN/new_crps/save/" + norm + "/mean_climatology/" + str(year) + "/window" + str(
            window) + "/" + filename, climat)
        np.save("/scratch/iu60/yl3101/DESRGAN/new_crps/save/" + norm + "/mean_climatology/" + str(year) + "/window" + str(
            window) + "/" + filename_log, climat_log)
        np.save("/scratch/iu60/yl3101/DESRGAN/new_crps/save/" + norm + "/mean_climatology/" + str(year) + "/window" + str(
            window) + "/" + filename_0, prob_0)
        np.save("/scratch/iu60/yl3101/DESRGAN/new_crps/save/" + norm + "/mean_climatology/" + str(year) + "/window" + str(
            window) + "/" + filename_95, prob95)
        np.save("/scratch/iu60/yl3101/DESRGAN/new_crps/save/" + norm + "/mean_climatology/" + str(year) + "/window" + str(
            window) + "/" + filename_99, prob99)
        np.save("/scratch/iu60/yl3101/DESRGAN/new_crps/save/" + norm + "/mean_climatology/" + str(year) + "/window" + str(
            window) + "/" + file_mae, mae_score)
        np.save("/scratch/iu60/yl3101/DESRGAN/new_crps/save/" + norm + "/mean_climatology/" + str(year) + "/window" + str(
            window) + "/" + file_bias, bias_score)
        np.save("/scratch/iu60/yl3101/DESRGAN/new_crps/save/" + norm + "/mean_climatology/" + str(year) + "/window" + str(
            window) + "/" + file_relative_bias, realtive_bias_score)

calculate_csv_file()
