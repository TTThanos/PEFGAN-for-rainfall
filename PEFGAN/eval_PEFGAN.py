import data_processing_tool as dpt
import random
from torch.utils.data import Dataset
import torch
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
import platform
from datetime import timedelta, date, datetime
import numpy as np
import os
import time
import properscoring as ps


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
    
# def relative_bias(ens, hr):

#     return (ens - hr).sum(axis=0) / ens.shape[0] / hr

def rmse(ens, hr):
    '''
    ens:(ensemble,H,W)
    hr: (H,W)
    '''
    return np.sqrt(((ens - hr) ** 2).sum(axis=(0)) / ens.shape[0])


# ===========================================================
# Training settings
# ===========================================================


class ACCESS_AWAP_cali(Dataset):
    '''
2.using my net to train one channel to one channel.
    '''

    def __init__(self, start_date=date(1990, 1, 1), end_date=date(1990, 12, 31), regin="AUS", lr_transform=None,
                 hr_transform=None, shuffle=True, args=None):
        #         print("=> BARRA_R & ACCESS_S1 loading")
        #         print("=> from "+start_date.strftime("%Y/%m/%d")+" to "+end_date.strftime("%Y/%m/%d")+"")
        # '/g/data/rr8/OBS/AWAP_ongoing/v0.6/grid_05/daily/precip/'
        self.file_AWAP_dir = args.file_AWAP_dir
        self.file_ACCESS_dir = args.file_ACCESS_dir
        self.args = args

        self.lr_transform = lr_transform
        self.hr_transform = hr_transform

        self.start_date = start_date
        self.end_date = end_date

        self.regin = regin
        self.leading_time_we_use = args.leading_time_we_use

        self.ensemble_access = ['e01', 'e02', 'e03', 'e04',
                                'e05', 'e06', 'e07', 'e08', 'e09']
        self.ensemble = []
        for i in range(len(self.ensemble_access)):
            self.ensemble.append(self.ensemble_access[i])

        self.dates = self.date_range(start_date, end_date)

        self.filename_list = self.get_filename_with_time_order(
            self.file_ACCESS_dir)
        if not os.path.exists(self.file_ACCESS_dir):
            print(self.file_ACCESS_dir)
            print("no file or no permission")
        print(self.filename_list)
        en, cali_date, date_for_AWAP, time_leading = self.filename_list[0]
        if shuffle:
            random.shuffle(self.filename_list)

        data_awap = dpt.read_awap_data_fc_get_lat_lon(
            self.file_AWAP_dir, date_for_AWAP)
        self.lat_awap = data_awap[1]
        self.lon_awap = data_awap[2]

    def __len__(self):
        return len(self.filename_list)

    def date_range(self, start_date, end_date):
        """This function takes a start date and an end date as datetime date objects.
        It returns a list of dates for each date in order starting at the first date and ending with the last date"""
        return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

    def get_filename_with_no_time_order(self, rootdir):
        '''get filename first and generate label '''
        _files = []
        list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
        for i in range(0, len(list)):
            path = os.path.join(rootdir, list[i])
            if os.path.isdir(path):
                _files.extend(self.get_filename_with_no_time_order(path))
            if os.path.isfile(path):
                if path[-3:] == ".nc":
                    _files.append(path)
        return _files

    def get_filename_with_time_order(self, rootdir):
        '''get filename first and generate label ,one different w'''
        _files = []
        for date in self.dates:
            for i in range(self.leading_time_we_use, self.leading_time_we_use + 1):

                for en in self.ensemble:
                    access_path = rootdir + "e09" + "/" + \
                                  date.strftime("%Y-%m-%d") + "_" + "e09" + ".nc"
                    #                   print(access_path)
                    if os.path.exists(access_path):

                        if date == self.end_date and i == 1:
                            break
                        path = []
                        path.append(en)
                        awap_date = date + timedelta(i)
                        path.append(date)
                        path.append(awap_date)
                        path.append(i)
                        _files.append(path)

        # 最后去掉第一行，然后shuffle
        return _files

    def mapping(self, X, min_val=0., max_val=255.):
        Xmin = np.min(X)
        Xmax = np.max(X)
        # 将数据映射到[-1,1]区间 即a=-1，b=1
        a = min_val
        b = max_val
        Y = a + (b - a) / (Xmax - Xmin) * (X - Xmin)
        return Y

    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
        t = time.time()

        # read_data filemame[idx]
        en, access_date, awap_date, time_leading = self.filename_list[idx]

        lr = dpt.read_access_data_calibrataion(
            self.file_ACCESS_dir, en, access_date, time_leading, "pr")
        lr_log = dpt.read_access_data_calibrataion_log(
            self.file_ACCESS_dir, en, access_date, time_leading, "pr")
        label, AWAP_date = dpt.read_awap_data_fc(self.file_AWAP_dir, awap_date)
        label_log, AWAP_date_log = dpt.read_awap_data_fc_log(self.file_AWAP_dir, awap_date)

        return np.array(lr), np.array(label), np.array(lr_log), np.array(label_log), torch.tensor(
            int(en[1:])), torch.tensor(int(access_date.strftime("%Y%m%d"))),  torch.tensor(int(AWAP_date.strftime("%Y%m%d"))), torch.tensor(time_leading)


def write_log(log, args):
    print(log)
    if not os.path.exists("./save/" + args.train_name + "/"):
        os.mkdir("./save/" + args.train_name + "/")
    my_log_file = open("./save/" + args.train_name + '/train.txt', 'a')
    my_log_file.write(log + '\n')
    my_log_file.close()
    return


def main(year, days):

    model_name = 'vdynamic_weights'
    model_num = 'model_G_i000010'
    Brier_startyear = 1981
    Brier_endyear = 2010
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    # Hardware specifications
    parser.add_argument('--n_threads', type=int, default=0,
                        help='number of threads for data loading')

    parser.add_argument('--cpu', action='store_true', help='cpu only?')

    # hyper-parameters
    parser.add_argument('--train_name', type=str,
                        default="cali_crps", help='training name')

    parser.add_argument('--batch_size', type=int,
                        default=9, help='training batch size')
    parser.add_argument('--testBatchSize', type=int,
                        default=4, help='testing batch size')
    parser.add_argument('--nEpochs', type=int, default=200,
                        help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning Rate. Default=0.01')
    parser.add_argument('--seed', type=int, default=123,
                        help='random seed to use. Default=123')

    # model configuration
    parser.add_argument('--upscale_factor', '-uf', type=int,
                        default=4, help="super resolution upscale factor")
    parser.add_argument('--model', '-m', type=str, default='vdsr',
                        help='choose which model is going to use')

    # data
    parser.add_argument('--pr', type=bool, default=True, help='add-on pr?')

    parser.add_argument('--train_start_time', type=type(datetime(1990,
                                                                 1, 25)), default=datetime(1990, 1, 2), help='r?')
    parser.add_argument('--train_end_time', type=type(datetime(1990,
                                                               1, 25)), default=datetime(1990, 2, 9), help='?')
    parser.add_argument('--test_start_time', type=type(datetime(2012,
                                                                1, 1)), default=datetime(2012, 1, 1), help='a?')
    parser.add_argument('--test_end_time', type=type(datetime(2012,
                                                              12, 31)), default=datetime(2012, 12, 31), help='')

    parser.add_argument('--dem', action='store_true', help='add-on dem?')
    parser.add_argument('--psl', action='store_true', help='add-on psl?')
    parser.add_argument('--zg', action='store_true', help='add-on zg?')
    parser.add_argument('--tasmax', action='store_true', help='add-on tasmax?')
    parser.add_argument('--tasmin', action='store_true', help='add-on tasmin?')
    parser.add_argument('--leading_time_we_use', type=int,
                        default=1, help='add-on tasmin?')
    parser.add_argument('--ensemble', type=int, default=9,
                        help='total ensambles is 9')
    parser.add_argument('--channels', type=float, default=0,
                        help='channel of data_input must')
    # [111.85, 155.875, -44.35, -9.975]
    parser.add_argument('--domain', type=list,
                        default=[140.6, 153.9, -39.2, -18.6], help='dataset directory')
    
    parser.add_argument('--file_ACCESS_dir', type=str,
                        default="/scratch/iu60/yl3101/PEFGAN/"+ model_name + '/' + str(year) + "/" + model_num + "/",
                        help='dataset directory')
    parser.add_argument('--file_AWAP_dir', type=str, default="/scratch/iu60/yl3101/AGCD_mask_data/",
                        help='dataset directory')

    parser.add_argument('--precision', type=str, default='single', choices=('single', 'half', 'double'),
                        help='FP precision for test (single | half)')

    args = parser.parse_args()

    sys = platform.system()
    args.dem = False
    args.train_name = "pr_DESRGAN"
    args.channels = 0
    if args.pr:
        args.channels += 1
    if args.zg:
        args.channels += 1
    if args.psl:
        args.channels += 1
    if args.tasmax:
        args.channels += 1
    if args.tasmin:
        args.channels += 1
    if args.dem:
        args.channels += 1
    print("training statistics:")
    print("  ------------------------------")
    print("  trainning name  |  %s" % args.train_name)
    print("  ------------------------------")
    print("  evaluating model | %s" % args.file_ACCESS_dir[-17:])
    print("  ------------------------------")
    print("  num of channels | %5d" % args.channels)
    print("  ------------------------------")
    print("  num of threads  | %5d" % args.n_threads)
    print("  ------------------------------")
    print("  batch_size     | %5d" % args.batch_size)
    print("  ------------------------------")
    print("  using cpu only | %5d" % args.cpu)
    print("  ------------------------------")
    print("  The percentile year begin from | %5d" % int(Brier_startyear))
    print("  ------------------------------")
    print("  The percentile year end at | %5d" % int(Brier_endyear))

    lr_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    hr_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    args.test_start_time = datetime(year, 1, 1)
    args.test_end_time = datetime(year, 12, 31)

    write_log("start", args)
    percentile_95 = dpt.AWAPcalpercentile(Brier_startyear, Brier_endyear, 95)
    print("percentile 95 is : ", percentile_95)
    # print("type of percentile95", type(percentile_95))
    # print("The size of percentile95 ", len(percentile_95))
    # print("The size of percentile95[0] ", len(percentile_95[0]))
    # print('Maximum  value of percentile 95 ', percentile_95.max())
    percentile_99 = dpt.AWAPcalpercentile(Brier_startyear, Brier_endyear, 99)
    for lead in range(0, days):
        args.leading_time_we_use = lead

        data_set = ACCESS_AWAP_cali(args.test_start_time, args.test_end_time, lr_transform=lr_transforms,
                                    hr_transform=hr_transforms, shuffle=False, args=args)
        print("data_set length:", data_set.__len__())
        test_data = DataLoader(data_set,
                               batch_size=args.batch_size,
                               shuffle=False,
                               num_workers=args.n_threads, drop_last=False)
        
        Brier_0_model = []
        Brier_95_model = []
        Brier_99_model = []
        mean_mae_model = []
        mean_mae_mean_model = []
        mean_mae_mediam_model = []
        mean_bias_model = []
        mean_relative_bias_model = []
        mean_rmse_model = []
        mean_crps_model = []
        mean_crps_model_log = []
        mean_bias_median_model = []
        print("test_data length:", len(test_data))
        for batch, (pr, hr, pr_log, hr_log, _, access_date, awap_date, _) in enumerate(test_data):

            with torch.set_grad_enabled(False):

                sr_np = pr.cpu().numpy()
                hr_np = hr.cpu().numpy()
                sr_log_np = pr_log.cpu().numpy()
                hr_log_np = hr_log.cpu().numpy()
                print("sr:", sr_np.shape)
                print("hr:", hr_np.shape)
                # print("sr:", sr_log_np.shape)
                # print("hr:", hr_log_np.shape)
                print("ACCESS_date", access_date)
                print("AWAP_date", awap_date)
                # print("num of i", args.batch_size // args.ensemble)
                for i in range(args.batch_size // args.ensemble):
                    a = np.squeeze(
                        sr_np[i * args.ensemble:(i + 1) * args.ensemble])
                    print("shape of a ", a.shape)
                    b = np.squeeze(hr_np[i * args.ensemble])
                    print("shape of b ", b.shape)
                    a_log = np.squeeze(
                        sr_log_np[i * args.ensemble:(i + 1) * args.ensemble])
                    b_log = np.squeeze(hr_log_np[i * args.ensemble])
                    mae_DESRGAN = mae(a, b)
                    mae_mean_DESRGAN = mae_mean(a, b)
                    mae_median_DESRGAN = mae_median(a, b)
                    bias_DESRGAN = bias(a, b)
                    bias_median_DESRGAN = bias_median(a, b)
                    rmse_DESRGAN = rmse(a, b)
                    skil_DESRGAN = ps.crps_ensemble(b, np.transpose(a, (1, 2, 0)))
                    print("The shape of CRPS: ", skil_DESRGAN.shape)
                    skil_DESRGAN_log = ps.crps_ensemble(b_log, np.transpose(a_log, (1, 2, 0)))
                    relative_bias_DESRGAN = bias_relative(a, b, constant=3)
                    #calculating Brier score
                    prob_AWAP_0 = calAWAPdryprob(b, 0.1)
                    prob_forecast_0 = calforecastdryprob(a, 0.1)
                    Brier_0_model.append((prob_AWAP_0 - prob_forecast_0) ** 2)
                    prob_AWAP_95 = calAWAPprob(b, percentile_95)
                    prob_forecast_95 = calforecastprob(a, percentile_95)
                    print("prob of AWAP: ", prob_AWAP_95)
                    print("prob of Forecast: ", prob_forecast_95)
                    Brier_95_model.append((prob_AWAP_95 - prob_forecast_95) ** 2)
                    prob_AWAP_99 = calAWAPprob(b, percentile_99)
                    prob_forecast_99 = calforecastprob(a, percentile_99)
                    Brier_99_model.append((prob_AWAP_99 - prob_forecast_99) ** 2)


                    mean_mae_model.append(mae_DESRGAN)
                    mean_mae_mean_model.append(mae_mean_DESRGAN)
                    mean_mae_mediam_model.append(mae_median_DESRGAN)
                    mean_bias_model.append(bias_DESRGAN)
                    mean_bias_median_model.append(bias_median_DESRGAN)
                    mean_relative_bias_model.append(relative_bias_DESRGAN)
                    mean_rmse_model.append(rmse_DESRGAN)
                    mean_crps_model.append(skil_DESRGAN)
                    mean_crps_model_log.append(skil_DESRGAN_log)
        # print("lead time = ", lead)
        
        print("The length of mean_crps_model: ", len(mean_crps_model))
        print("The shape of the mean of mean_crps_model: ", np.mean(mean_crps_model, axis=0).shape)
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/Brier0/" + model_name + "/"):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/Brier0/"+ model_name)
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/Brier0/" + model_name + "/" + str(year)):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/Brier0/"+ model_name + "/" + str(year))
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/Brier0/" + model_name + "/" + str(year) + "/" + model_num):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/Brier0/"+ model_name + "/" + str(year)+ "/" + model_num)    
        np.save("/scratch/iu60/yl3101/PEFGAN/new_crps/save/Brier0/" + model_name + "/" + str(year) + "/" + model_num +  "/lead_time" + str(lead) + '_whole',
                np.mean(Brier_0_model, axis=0))

        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/Brier95/" + model_name + "/"):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/Brier95/"+ model_name)
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/Brier95/" + model_name + "/" + str(year)):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/Brier95/"+ model_name + "/" + str(year))
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/Brier95/" + model_name + "/" + str(year) + "/" + model_num):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/Brier95/"+ model_name + "/" + str(year)+ "/" + model_num)    
        np.save("/scratch/iu60/yl3101/PEFGAN/new_crps/save/Brier95/" + model_name + "/" + str(year) + "/" + model_num +  "/lead_time" + str(lead) + '_whole',
                np.mean(Brier_95_model, axis=0))
        
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/Brier99/" + model_name + "/"):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/Brier99/"+ model_name)
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/Brier99/" + model_name + "/" + str(year)):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/Brier99/"+ model_name + "/" + str(year))
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/Brier99/" + model_name + "/" + str(year) + "/" + model_num):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/Brier99/"+ model_name + "/" + str(year)+ "/" + model_num)    
        np.save("/scratch/iu60/yl3101/PEFGAN/new_crps/save/Brier99/" + model_name + "/" + str(year) + "/" + model_num +  "/lead_time" + str(lead) + '_whole',
                np.mean(Brier_99_model, axis=0))

        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/mae/" + model_name + "/"):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/mae/"+ model_name)
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/mae/" + model_name + "/" + str(year)):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/mae/"+ model_name + "/" + str(year))
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/mae/" + model_name + "/" + str(year) + "/" + model_num):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/mae/"+ model_name + "/" + str(year)+ "/" + model_num)    
        np.save("/scratch/iu60/yl3101/PEFGAN/new_crps/save/mae/" + model_name + "/" + str(year) + "/" + model_num +  "/lead_time" + str(lead) + '_whole',
                np.mean(mean_mae_model, axis=0))

        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/mae_mean/" + model_name + "/"):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/mae_mean/"+ model_name)
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/mae_mean/" + model_name + "/" + str(year)):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/mae_mean/"+ model_name + "/" + str(year))
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/mae_mean/" + model_name + "/" + str(year) + "/" + model_num):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/mae_mean/" + model_name + "/" + str(year) + "/" + model_num)
        np.save(
            "/scratch/iu60/yl3101/PEFGAN/new_crps/save/mae_mean/" + model_name + "/" + str(year) + "/" + model_num + "/lead_time" + str(lead) + '_whole',
            np.mean(mean_mae_mean_model, axis=0))


        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/mae_median/" + model_name + "/"):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/mae_median/"+ model_name)
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/mae_median/" + model_name + "/" + str(year)):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/mae_median/"+ model_name + "/" + str(year))
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/mae_median/" + model_name + "/" + str(year) + "/" + model_num):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/mae_median/" + model_name + "/" + str(year) + "/" + model_num)
        np.save(
            "/scratch/iu60/yl3101/PEFGAN/new_crps/save/mae_median/" + model_name + "/" + str(year) + "/" + model_num + "/lead_time" + str(lead) + '_whole',
            np.mean(mean_mae_mediam_model, axis=0))

        if not os.path.exists("/scratch/iu60/yl3101/v/new_crps/save/bias/" + model_name + "/"):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/bias/"+ model_name)
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/bias/" + model_name + "/" + str(year)):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/bias/"+ model_name + "/" + str(year))
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/bias/" + model_name + "/" + str(year) + "/" + model_num):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/bias/" + model_name + "/" + str(year)+ "/" + model_num)
        np.save(
            "/scratch/iu60/yl3101/PEFGAN/new_crps/save/bias/" + model_name + "/" + str(year) + "/" + model_num + "/lead_time" + str(lead) + '_whole',
            np.mean(mean_bias_model, axis=0))

        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/bias_median/" + model_name + "/"):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/bias_median/"+ model_name)
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/bias_median/" + model_name + "/" + str(year)):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/bias_median/"+ model_name + "/" + str(year))
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/bias_median/" + model_name + "/" + str(year) + "/" + model_num):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/bias_median/" + model_name + "/" + str(year)+ "/" + model_num)
        np.save(
            "/scratch/iu60/yl3101/PEFGAN/new_crps/save/bias_median/" + model_name + "/" + str(year) + "/" + model_num + "/lead_time" + str(lead) + '_whole',
            np.mean(mean_bias_median_model, axis=0))
        
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/relative_bias/" + model_name + "/"):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/relative_bias/"+ model_name)
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/relative_bias/" + model_name + "/" + str(year)):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/relative_bias/"+ model_name + "/" + str(year))
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/relative_bias/" + model_name + "/" + str(year) + "/" + model_num):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/relative_bias/" + model_name + "/" + str(year)+ "/" + model_num)
        np.save(
            "/scratch/iu60/yl3101/PEFGAN/new_crps/save/relative_bias/" + model_name + "/" + str(year) + "/" + model_num + "/lead_time" + str(lead) + '_whole',
            np.mean(mean_relative_bias_model, axis=0))

        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/rmse/" + model_name + "/"):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/rmse/"+ model_name)
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/rmse/" + model_name + "/" + str(year)):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/rmse/"+ model_name + "/" + str(year))
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/rmse/" + model_name + "/" + str(year) + "/" + model_num):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/rmse/" + model_name + "/" + str(year)+ "/" + model_num)
        np.save(
            "/scratch/iu60/yl3101/PEFGAN/new_crps/save/rmse/" + model_name + "/" + str(year) + "/" + model_num + "/lead_time" + str(lead) + '_whole',
            np.mean(mean_rmse_model, axis=0))

        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/crps_ss/" + model_name + "/"):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/crps_ss/"+ model_name)
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/crps_ss/" + model_name + "/" + str(year)):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/crps_ss/"+ model_name + "/" + str(year))
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/crps_ss/" + model_name + "/" + str(year) + "/" + model_num):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/crps_ss/" + model_name + "/" + str(year)+ "/" + model_num)
        np.save(
            "/scratch/iu60/yl3101/PEFGAN/new_crps/save/crps_ss/" + model_name + "/" + str(year) + "/" + model_num + "/lead_time" + str(lead) + '_whole',
            np.mean(mean_crps_model, axis=0))
        
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/crps_ss_log/" + model_name + "/"):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/crps_ss_log/"+ model_name)
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/crps_ss_log/" + model_name + "/" + str(year)):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/crps_ss_log/"+ model_name + "/" + str(year))
        if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/save/crps_ss_log/" + model_name + "/" + str(year) + "/" + model_num):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/save/crps_ss_log/" + model_name + "/" + str(year)+ "/" + model_num)
        np.save(
            "/scratch/iu60/yl3101/PEFGAN/new_crps/save/crps_ss_log/" + model_name + "/" + str(year) + "/" + model_num + "/lead_time" + str(lead) + '_whole',
            np.mean(mean_crps_model_log, axis=0))
        


if __name__ == '__main__':
    main(year=2009, days=42)
    print('2009 done')

    # main(year=2012, days=42)
    # print('2012 done')
    
    # main(year=2010, days=42)
    # print('2010 done')

    # main(year=2006, days=42)
    # print('2006 done')
    # main(year=2007, days=42)
    # print('2007 done')
    # main(year=2015, days=42)
    # print('2015 done')
    # main(year=2018, days=42)
    # print('2018 done')