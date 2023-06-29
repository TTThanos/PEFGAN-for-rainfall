import xarray as xr
import matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm

import csv

levels = {}
# levels["crps"]   = [0,0.2,0.4,0.6,0.8,1.0]
levels["crpsss"] = [ -0.4, -0.2, 0, 0.2, 0.4]
levels["new"] = [0, 0.1, 1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 100, 150]
levels["mae"] = [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1, 2, 4]
levels["bias"] = [-2, -1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5, 2]
levels["hour"] = [0., 0.2, 1, 5, 10, 20, 30, 40, 60, 80, 100, 150]
levels["day"] = [0., 0.2, 5, 10, 20, 30, 40, 60, 100, 150, 200, 300]
levels["week"] = [0., 0.2, 10, 20, 30, 50, 100, 150, 200, 300, 500, 1000]
levels["month"] = [0., 10, 20, 30, 40, 50, 100, 200, 300, 500, 1000, 1500]
levels["year"] = [0., 50, 100, 200, 300, 400, 600, 1000, 1500, 2000, 3000, 5000]
enum = {0: "0600", 1: "1200", 2: "1800", 3: "0000", 4: "0600"}

prcp_colours = [
    "#FFFFFF",
    '#edf8b1',
    '#c7e9b4',
    '#7fcdbb',
    '#41b6c4',
    '#1d91c0',
    '#225ea8',
    '#253494',
    '#4B0082',
    "#800080",
    '#8B0000']
prcp_colours_new = [
    "#FFFFFF",
    '#edf8b1',
    '#c7e9b4',
    '#7fcdbb',
    '#225ea8',
    '#253494', 
    '#8B0000']
prcp_colormap = matplotlib.colors.ListedColormap(prcp_colours)


def draw_aus(var, lat, lon, domain=[140.6, 153.9, -39.2, -18.6], mode="pr", titles_on=True,
             title="CRPS of precipation in 2012", colormap=prcp_colormap, cmap_label="CRPS-Skill Score", save=False,
             path="", color_bar = False):
    """ basema_ploting .py
This function takes a 2D data set of a variable from AWAP and maps the data on miller projection.
The map default span is longitude between 111.975E and 156.275E, and the span for latitudes is -44.525 to -9.975.
The colour scale is YlGnBu at 11 levels.
The levels specifed are suitable for annual rainfall totals for Australia.
"""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from mpl_toolkits.basemap import Basemap, maskoceans

    prcp_colormap_new = matplotlib.colors.ListedColormap(prcp_colours_new)

    if mode == "pr":
        level = 'new'

    # crps-ss
    if mode == "crps-ss":
        level = "crpsss"
        colormap = prcp_colormap_new

    if mode == "mae" or mode == "rmse":
        level = "mae"

    if mode == "bias":
        level = "bias"

    fig = plt.figure()
    level = levels[level]
    map = Basemap(projection="mill", llcrnrlon=domain[0], llcrnrlat=domain[2], urcrnrlon=domain[1], urcrnrlat=domain[3],
                  resolution='l')

    map.drawcoastlines()

    # map.drawmapboundary()
    map.drawparallels(np.arange(-90., 120., 5.), dashes=[1, 5], labels=[1,0,0,0])
    map.drawmeridians(np.arange(-180., 180., 5.), dashes=[1, 5], labels=[0, 0, 0, 1], rotation=45)
    llons, llats = np.meshgrid(lon, lat)  # 将维度按照 x,y 横向竖向
    # print(lon.shape,llons.shape)
    x, y = map(llons, llats)
    # print(x.shape,y.shape)

    norm = BoundaryNorm(level, len(level) - 1)

    # red square
    # var[255:260,205:510]= 1000
    # var[495:500,210:510]= 1000
    # var[260:500,205:210]= 1000
    # var[260:500,505:510]= 1000

    data = xr.DataArray(var, coords=[lat, lon], dims=["lat", "lon"])

    # pr
    if mode == "pr":
        cs = map.pcolormesh(x, y, data, norm=norm, cmap=colormap)

        # crps-ss
    if mode == "crps-ss":
        cs = map.pcolormesh(x, y, data, cmap="RdBu", vmin=-0.4, vmax=0.4)

    if mode == "mae" or mode == "rmse":
        cs = map.pcolormesh(x, y, data, norm=norm, cmap=colormap)

    if mode == "bias":
        cs = map.pcolormesh(x, y, data, cmap="RdBu", vmin=-2, vmax=2)

    if titles_on:
        # label with title, latitude, longitude, and colormap

        plt.title(title)
        plt.xlabel("\n\n\nLongitude")
        plt.ylabel("Latitude\n\n")

        # color bar
    if color_bar:
        cbar = plt.colorbar(ticks=level[:-1], shrink=0.8, extend="max")  # shrink = 0.8
        cbar.ax.set_ylabel(cmap_label)

        # cbar.ax.set_xticklabels(level) #报错

    # plt.plot([-1000,1000],[900,1000], c="b", linewidth=2, linestyle=':')

    if save:
        plt.savefig(path, bbox_inches = 'tight', dpi=300)
    else:
        plt.show()
    plt.cla()
    plt.close("all")
    return


def evaluate(year, lead, draw=False, window=1):
    '''
    evaluate CRPS-SS
    '''
    model_name = 'vdynamic_weights'
    model_num = 'model_G_i000008'
    print(year + " ," + str(lead))
    print('Outputing the csv file....')
    print('Model name is: ', model_name)
    print('model epoch is:', model_num)

    total = np.zeros((413, 267))
    total_crps_list = []
    total_qm = np.zeros((413, 267))
    qm_crps_list = []
    
    # total_bi = np.zeros((691, 886))
    # total_vdsr = np.zeros((691, 886))
    qm_mask = np.load("/scratch/iu60/yl3101/QM(AWAP)_mask/awap_binary_mask.npy")
    if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/csv_files/" + model_name + "/"):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/csv_files/"+ model_name)
    if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/csv_files/" + model_name + "/" + str(year)):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/csv_files/"+ model_name + "/" + str(year))
    if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/csv_files/" + model_name + "/" + str(year) + "/" + model_num):
            os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/csv_files/" + model_name + "/" + str(year)+ "/" + model_num)
    model_file_csv = "/scratch/iu60/yl3101/PEFGAN/new_crps/csv_files/"+ model_name + "/" + str(year)+ "/" + model_num + "/crps_ss_" + year + "_window" + str(window) + ".csv"
    climat_file_csv =  "/scratch/iu60/yl3101/PEFGAN/new_crps/csv_files/" + model_name + "/" + str(year)+ "/" + model_num + "/climatology_" + year + "_window" + str(window) + ".csv"
    QM_file_csv =  "/scratch/iu60/yl3101/PEFGAN/new_crps/csv_files/" + model_name + "/" + str(year)+ "/" + model_num + "/QM_" + year + "_window" + str(window) + ".csv"

    # write CSV file for climatology
    with open(climat_file_csv, "w", newline='') as file:
        csv_file = csv.writer(file)
        head = ["lead time", "climat_crps", "climat_crpsss", "climat mae", "climat bias", "climat relative_bias","climat Brierscore0", "climat Brierscore95", "climat Brierscore99"]
        csv_file.writerow(head)

        for time in (range(0, lead)):
            line = [time, ]
            leading0_clima = np.load(
                "/scratch/iu60/yl3101/PEFGAN/new_crps/save/crps_ss/mean_climatology/" + year + "/window" + str(
                    window) + "/climat_lead_time_" + str(
                    time) + ".npy", allow_pickle=True)
            leading0_clima = leading0_clima.astype("float32")
            leading0_clima = np.ma.masked_array(leading0_clima, mask=qm_mask)

            leading0_clima_0 = np.load(
                "/scratch/iu60/yl3101/PEFGAN/new_crps/save/crps_ss/mean_climatology/" + year + "/window" + str(
                    window) + "/prob0_climat_lead_time_" + str(
                    time) + ".npy", allow_pickle=True)
            leading0_clima_0 = leading0_clima_0.astype("float32")
            leading0_clima_0 = np.ma.masked_array(leading0_clima_0, mask=qm_mask)


            leading0_clima_95 = np.load(
                "/scratch/iu60/yl3101/PEFGAN/new_crps/save/crps_ss/mean_climatology/" + year + "/window" + str(
                    window) + "/prob95_climat_lead_time_" + str(
                    time) + ".npy", allow_pickle=True)
            leading0_clima_95 = leading0_clima_95.astype("float32")
            leading0_clima_95 = np.ma.masked_array(leading0_clima_95, mask=qm_mask)

            leading0_clima_99 = np.load(
                "/scratch/iu60/yl3101/PEFGAN/new_crps/save/crps_ss/mean_climatology/" + year + "/window" + str(
                    window) + "/prob99_climat_lead_time_" + str(
                    time) + ".npy", allow_pickle=True)
            leading0_clima_99 = leading0_clima_99.astype("float32")
            leading0_clima_99 = np.ma.masked_array(leading0_clima_99, mask=qm_mask)

            climat_mae = np.load(
                "/scratch/iu60/yl3101/PEFGAN/new_crps/save/crps_ss/mean_climatology/" + year + "/window" + str(
                    window) + "/mae_climat_lead_time_" + str(
                    time) + ".npy", allow_pickle=True)
            climat_mae = climat_mae.astype("float32")
            climat_mae = np.ma.masked_array(climat_mae, mask=qm_mask)

            climat_bias = np.load(
                "/scratch/iu60/yl3101/PEFGAN/new_crps/save/crps_ss/mean_climatology/" + year + "/window" + str(
                    window) + "/bias_climat_lead_time_" + str(
                    time) + ".npy", allow_pickle=True)
            climat_bias = climat_bias.astype("float32")
            climat_bias = np.ma.masked_array(climat_bias, mask=qm_mask)

            climat_realative_bias = np.load(
                "/scratch/iu60/yl3101/PEFGAN/new_crps/save/crps_ss/mean_climatology/" + year + "/window" + str(
                    window) + "/relative_bias_climat_lead_time_" + str(
                    time) + ".npy", allow_pickle=True)
            climat_realative_bias = climat_realative_bias.astype("float32")
            climat_realative_bias = np.ma.masked_array(climat_realative_bias, mask=qm_mask)

            line.append(np.nanmean(leading0_clima))
            line.append(np.nanmean(0))
            line.append(np.nanmean(climat_mae))
            line.append(np.nanmean(climat_bias))
            line.append(np.nanmean(climat_realative_bias))   
            line.append(np.nanmean(leading0_clima_0)) 
            line.append(np.nanmean(leading0_clima_95))
            line.append(np.nanmean(leading0_clima_99))
            csv_file.writerow(line)

    # write CSV file for Quantile Mapping
    with open(QM_file_csv, "w", newline='') as file:
        csv_file = csv.writer(file)
        head = ["lead time", "QM_crps", "QM_crpsss", "QM mae", "QM bias", "QM relative_bias","QM Brierscore0", "QM Brierscore95", "QM Brierscore99"]
        csv_file.writerow(head)

        for time in range(0, lead):
            line = [time, ]

            leading0_clima = np.load(
                "/scratch/iu60/yl3101/PEFGAN/new_crps/save/crps_ss/mean_climatology/" + year + "/window" + str(
                    window) + "/climat_lead_time_" + str(
                    time) + ".npy", allow_pickle=True)
            leading0_clima = leading0_clima.astype("float32")
            leading0_clima = np.ma.masked_array(leading0_clima, mask=qm_mask)

            leading0_qm = np.load(
                "/scratch/iu60/yl3101/PEFGAN/new_crps/save/crps_ss/QM/" + year + "/lead_time" + str(time) + "_whole.npy")
            leading0_qm = np.ma.masked_array(leading0_qm, mask=qm_mask)

            leading0_QM_0 = np.load(
                "/scratch/iu60/yl3101/PEFGAN/new_crps/save/Brier0/QM/"  + str(year) + 
                "/lead_time" + str(time) + '_whole.npy', allow_pickle=True)
            leading0_QM_0 = leading0_QM_0.astype("float32")
            leading0_QM_0 = np.ma.masked_array(leading0_QM_0, mask=qm_mask)


            leading0_QM_95 = np.load(
                "/scratch/iu60/yl3101/PEFGAN/new_crps/save/Brier95/QM/" + str(year) +
                  "/lead_time" + str(time) + '_whole.npy', 
                allow_pickle=True)
            leading0_QM_95 = leading0_QM_95.astype("float32")
            leading0_QM_95 = np.ma.masked_array(leading0_QM_95, mask=qm_mask)

            leading0_QM_99 = np.load(
                "/scratch/iu60/yl3101/PEFGAN/new_crps/save/Brier99/QM/" + str(year) +
                  "/lead_time" + str(time) + '_whole.npy', 
                allow_pickle=True)
            leading0_QM_99 = leading0_QM_99.astype("float32")
            leading0_QM_99 = np.ma.masked_array(leading0_QM_99, mask=qm_mask)

            QM_mae = np.load(
                "/scratch/iu60/yl3101/PEFGAN/new_crps/save/mae/QM/" + str(year) +
                  "/lead_time" + str(time) + '_whole.npy', allow_pickle=True)
            QM_mae = QM_mae.astype("float32")
            QM_mae = np.ma.masked_array(QM_mae, mask=qm_mask)

            QM_bias = np.load(
                "/scratch/iu60/yl3101/PEFGAN/new_crps/save/bias/QM/" + str(year) +
                  "/lead_time" + str(time) + '_whole.npy', allow_pickle=True)
            QM_bias = QM_bias.astype("float32")
            QM_bias = np.ma.masked_array(QM_bias, mask=qm_mask)

            QM_realative_bias = np.load(
                "/scratch/iu60/yl3101/PEFGAN/new_crps/save/relative_bias/QM/" + str(year) +
                  "/lead_time" + str(time) + '_whole.npy', allow_pickle=True)
            QM_realative_bias = QM_realative_bias.astype("float32")
            QM_realative_bias = np.ma.masked_array(QM_realative_bias, mask=qm_mask)

            crpsss_qm = 1 - (leading0_qm / leading0_clima)
            qm_crps_list.append(crpsss_qm)
            total_qm = total_qm + crpsss_qm
            lat = [-39.2, -39.15, -39.100002, -39.05, -39.,
                       -38.95, -38.9, -38.850002, -38.8, -38.75,
                       -38.7, -38.65, -38.600002, -38.55, -38.5,
                       -38.45, -38.4, -38.350002, -38.3, -38.25,
                       -38.2, -38.15, -38.100002, -38.05, -38.,
                       -37.95, -37.9, -37.850002, -37.8, -37.75,
                       -37.7, -37.65, -37.600002, -37.55, -37.5,
                       -37.45, -37.4, -37.350002, -37.3, -37.25,
                       -37.2, -37.15, -37.100002, -37.05, -37.,
                       -36.95, -36.9, -36.850002, -36.8, -36.75,
                       -36.7, -36.65, -36.600002, -36.55, -36.5,
                       -36.45, -36.4, -36.350002, -36.3, -36.25,
                       -36.2, -36.15, -36.100002, -36.05, -36.,
                       -35.95, -35.9, -35.850002, -35.8, -35.75,
                       -35.7, -35.65, -35.600002, -35.55, -35.5,
                       -35.45, -35.4, -35.350002, -35.3, -35.25,
                       -35.2, -35.15, -35.100002, -35.05, -35.,
                       -34.95, -34.9, -34.850002, -34.8, -34.75,
                       -34.7, -34.65, -34.600002, -34.55, -34.5,
                       -34.45, -34.4, -34.350002, -34.3, -34.25,
                       -34.2, -34.15, -34.100002, -34.05, -34.,
                       -33.95, -33.9, -33.850002, -33.8, -33.75,
                       -33.7, -33.65, -33.600002, -33.55, -33.5,
                       -33.45, -33.4, -33.350002, -33.3, -33.25,
                       -33.2, -33.15, -33.100002, -33.05, -33.,
                       -32.95, -32.9, -32.850002, -32.8, -32.75,
                       -32.7, -32.65, -32.600002, -32.55, -32.5,
                       -32.45, -32.4, -32.350002, -32.3, -32.25,
                       -32.2, -32.15, -32.100002, -32.05, -32.,
                       -31.95, -31.900002, -31.85, -31.800001, -31.75,
                       -31.7, -31.650002, -31.6, -31.550001, -31.5,
                       -31.45, -31.400002, -31.35, -31.300001, -31.25,
                       -31.2, -31.150002, -31.1, -31.050001, -31.,
                       -30.95, -30.900002, -30.85, -30.800001, -30.75,
                       -30.7, -30.650002, -30.6, -30.550001, -30.5,
                       -30.45, -30.400002, -30.35, -30.300001, -30.25,
                       -30.2, -30.150002, -30.1, -30.050001, -30.,
                       -29.95, -29.900002, -29.85, -29.800001, -29.75,
                       -29.7, -29.650002, -29.6, -29.550001, -29.5,
                       -29.45, -29.400002, -29.35, -29.300001, -29.25,
                       -29.2, -29.150002, -29.1, -29.050001, -29.,
                       -28.95, -28.900002, -28.85, -28.800001, -28.75,
                       -28.7, -28.65, -28.6, -28.550001, -28.5,
                       -28.45, -28.4, -28.35, -28.300001, -28.25,
                       -28.2, -28.15, -28.1, -28.050001, -28.,
                       -27.95, -27.9, -27.85, -27.800001, -27.75,
                       -27.7, -27.65, -27.6, -27.550001, -27.5,
                       -27.45, -27.4, -27.35, -27.300001, -27.25,
                       -27.2, -27.15, -27.1, -27.050001, -27.,
                       -26.95, -26.9, -26.85, -26.800001, -26.75,
                       -26.7, -26.65, -26.6, -26.550001, -26.5,
                       -26.45, -26.4, -26.35, -26.300001, -26.25,
                       -26.2, -26.15, -26.1, -26.050001, -26.,
                       -25.95, -25.9, -25.85, -25.800001, -25.75,
                       -25.7, -25.65, -25.6, -25.550001, -25.5,
                       -25.45, -25.4, -25.35, -25.300001, -25.25,
                       -25.2, -25.15, -25.1, -25.050001, -25.,
                       -24.95, -24.9, -24.85, -24.800001, -24.75,
                       -24.7, -24.65, -24.6, -24.550001, -24.5,
                       -24.45, -24.4, -24.35, -24.300001, -24.25,
                       -24.2, -24.15, -24.1, -24.050001, -24.,
                       -23.95, -23.9, -23.85, -23.800001, -23.75,
                       -23.7, -23.65, -23.6, -23.550001, -23.5,
                       -23.45, -23.4, -23.35, -23.300001, -23.25,
                       -23.2, -23.15, -23.1, -23.050001, -23.,
                       -22.95, -22.9, -22.85, -22.800001, -22.75,
                       -22.7, -22.65, -22.6, -22.550001, -22.5,
                       -22.45, -22.4, -22.35, -22.300001, -22.25,
                       -22.2, -22.15, -22.1, -22.050001, -22.,
                       -21.95, -21.9, -21.85, -21.800001, -21.75,
                       -21.7, -21.65, -21.6, -21.550001, -21.5,
                       -21.45, -21.4, -21.35, -21.300001, -21.25,
                       -21.2, -21.15, -21.1, -21.050001, -21.,
                       -20.95, -20.9, -20.85, -20.800001, -20.75,
                       -20.7, -20.65, -20.6, -20.550001, -20.5,
                       -20.45, -20.4, -20.35, -20.300001, -20.25,
                       -20.2, -20.15, -20.1, -20.050001, -20.,
                       -19.95, -19.9, -19.85, -19.800001, -19.75,
                       -19.7, -19.65, -19.6, -19.550001, -19.5,
                       -19.45, -19.4, -19.35, -19.300001, -19.25,
                       -19.2, -19.15, -19.1, -19.050001, -19.,
                       -18.95, -18.9, -18.85, -18.800001, -18.75,
                       -18.7, -18.65, -18.6]
            lon = [140.6, 140.65001, 140.70001, 140.75, 140.8, 140.85,
                       140.90001, 140.95001, 141., 141.05, 141.1, 141.15001,
                       141.20001, 141.25, 141.3, 141.35, 141.40001, 141.45001,
                       141.5, 141.55, 141.6, 141.65001, 141.70001, 141.75,
                       141.8, 141.85, 141.90001, 141.95001, 142., 142.05,
                       142.1, 142.15001, 142.20001, 142.25, 142.3, 142.35,
                       142.40001, 142.45, 142.5, 142.55, 142.6, 142.65001,
                       142.7, 142.75, 142.8, 142.85, 142.90001, 142.95,
                       143., 143.05, 143.1, 143.15001, 143.2, 143.25,
                       143.3, 143.35, 143.40001, 143.45, 143.5, 143.55,
                       143.6, 143.65001, 143.7, 143.75, 143.8, 143.85,
                       143.90001, 143.95, 144., 144.05, 144.1, 144.15001,
                       144.2, 144.25, 144.3, 144.35, 144.40001, 144.45,
                       144.5, 144.55, 144.6, 144.65001, 144.7, 144.75,
                       144.8, 144.85, 144.90001, 144.95, 145., 145.05,
                       145.1, 145.15001, 145.2, 145.25, 145.3, 145.35,
                       145.40001, 145.45, 145.5, 145.55, 145.6, 145.65,
                       145.7, 145.75, 145.8, 145.85, 145.9, 145.95,
                       146., 146.05, 146.1, 146.15, 146.2, 146.25,
                       146.3, 146.35, 146.4, 146.45, 146.5, 146.55,
                       146.6, 146.65, 146.7, 146.75, 146.8, 146.85,
                       146.9, 146.95, 147., 147.05, 147.1, 147.15,
                       147.2, 147.25, 147.3, 147.35, 147.4, 147.45,
                       147.5, 147.55, 147.6, 147.65, 147.7, 147.75,
                       147.8, 147.85, 147.9, 147.95, 148., 148.05,
                       148.1, 148.15, 148.2, 148.25, 148.3, 148.35,
                       148.4, 148.45, 148.5, 148.55, 148.6, 148.65,
                       148.7, 148.75, 148.8, 148.85, 148.9, 148.95,
                       149., 149.05, 149.09999, 149.15, 149.2, 149.25,
                       149.3, 149.34999, 149.4, 149.45, 149.5, 149.55,
                       149.59999, 149.65, 149.7, 149.75, 149.8, 149.84999,
                       149.9, 149.95, 150., 150.05, 150.09999, 150.15,
                       150.2, 150.25, 150.3, 150.34999, 150.4, 150.45,
                       150.5, 150.55, 150.59999, 150.65, 150.7, 150.75,
                       150.8, 150.84999, 150.9, 150.95, 151., 151.05,
                       151.09999, 151.15, 151.2, 151.25, 151.3, 151.34999,
                       151.4, 151.45, 151.5, 151.55, 151.59999, 151.65,
                       151.7, 151.75, 151.8, 151.84999, 151.9, 151.95,
                       152., 152.05, 152.09999, 152.15, 152.2, 152.25,
                       152.29999, 152.34999, 152.4, 152.45, 152.5, 152.54999,
                       152.59999, 152.65, 152.7, 152.75, 152.79999, 152.84999,
                       152.9, 152.95, 153., 153.04999, 153.09999, 153.15,
                       153.2, 153.25, 153.29999, 153.34999, 153.4, 153.45,
                       153.5, 153.54999, 153.59999, 153.65, 153.7, 153.75,
                       153.79999, 153.84999, 153.9]
            if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/image/QM/"):
                        os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/image/QM")
            if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/image/QM/" + str(year)):
                        os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/image/QM/"+ str(year))
            if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/image/QM/" + str(year) + "/window_" + str(
                                 window)):
                        os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/image/QM/" + str(year) + "/window_" + str(
                                 window))
            draw_aus(crpsss_qm[0], mode='crps-ss', lat=lat, lon=lon, title="CRPS_SS_QM_" + str(time),
                             titles_on=True,
                             save=True,
                             path="/scratch/iu60/yl3101/PEFGAN/new_crps/image/QM/" + str(year) + "/window_" + str(
                                 window) + "/QM_" + str(time) + ".jpeg")
            # if time== 6 or time == 13 or time == 29 or time == 41:
            #       total_qm_save = total_qm / (time + 1)
            #       draw_aus(total_qm_save.squeeze(), mode='crps-ss', lat=lat, lon=lon, title="CRPS_SS_QM_" + str(time + 1),
            #                 titles_on=True, save=True,path="/scratch/iu60/yl3101/PEFGAN/new_crps/image/window_" + str(
            #                     window) + "/crps_ss/" + year + "/QM_" + str(time + 1) + ".jpeg")
            if time == 27:
                fortnight_QMcrps = qm_crps_list[14:]
                print("shape1: ", len(fortnight_QMcrps))
                print("shape1 mean: ", np.mean(fortnight_QMcrps, axis = 0).shape)
                draw_aus(np.mean(fortnight_QMcrps, axis = 0).squeeze(), mode='crps-ss', lat=lat, lon=lon, title="CRPS_SS_14-27_QM",
                        titles_on=True,
                        save=True,
                        path="/scratch/iu60/yl3101/PEFGAN/new_crps/image/window_" + str(
                            window) + "/crps_ss/" + year + "/QM_14-27.jpeg")
            if time == 41:
                fortnight_QMcrps = qm_crps_list[28:]
                print("shape1: ", len(fortnight_QMcrps))
                print("shape1 mean: ", np.mean(fortnight_QMcrps, axis = 0).shape)
                draw_aus(np.mean(fortnight_QMcrps, axis = 0).squeeze(), mode='crps-ss', lat=lat, lon=lon, title="CRPS_SS_28-41_QM",
                         
                        titles_on=True,
                        save=True,
                        path="/scratch/iu60/yl3101/PEFGAN/new_crps/image/window_" + str(
                            window) + "/crps_ss/" + year + "/QM_28-41.jpeg")
                


            line.append(np.nanmean(leading0_qm))
            line.append(np.nanmean(crpsss_qm))
            line.append(np.nanmean(QM_mae))
            line.append(np.nanmean(QM_bias))
            line.append(np.nanmean(QM_realative_bias))   
            line.append(np.nanmean(leading0_QM_0)) 
            line.append(np.nanmean(leading0_QM_95))
            line.append(np.nanmean(leading0_QM_99))
            csv_file.writerow(line)

    # write CSV file for model result
    with open(model_file_csv, "w", newline='') as file:
        csv_file = csv.writer(file)
        head = ["lead time", "PUGAN_crps", "PUGAN_crpsss", "mae", "bias", "relative_bias", "Brierscore_0", "Brierscore95", "Brierscore99"]
        csv_file.writerow(head)

        for time in (range(0, lead)):
            line = [time, ]

            leading0_clima = np.load(
                "/scratch/iu60/yl3101/PEFGAN/new_crps/save/crps_ss/mean_climatology/" + year + "/window" + str(
                    window) + "/climat_lead_time_" + str(
                    time) + ".npy", allow_pickle=True)
            leading0_clima = leading0_clima.astype("float32")
            leading0_clima = np.ma.masked_array(leading0_clima, mask=qm_mask)

            leading0_clima_log = np.load(
                "/scratch/iu60/yl3101/PEFGAN/new_crps/save/crps_ss/mean_climatology/" + year + "/window" + str(
                    window) + "/log_climat_lead_time_" + str(
                    time) + ".npy", allow_pickle=True)
            leading0_clima_log = leading0_clima_log.astype("float32")
            leading0_clima_log = np.ma.masked_array(leading0_clima_log, mask=qm_mask)

            
            # leading0_bi = np.load(
            #     "/scratch/iu60/rw6151/new_crps/save/crps_ss/BI/" + year + "/lead_time" + str(time) + "_whole.npy")
            # leading0_bi = np.ma.masked_array(leading0_bi, mask=qm_mask)
            #
            
            #
            # leading0_vdsr = np.load(
            #     "/scratch/iu60/rw6151/new_crps/save/crps_ss/VDSR/" + year + "/lead_time" + str(time) + "_whole.npy")
            # leading0_vdsr = leading0_vdsr.mean(axis=0)
            # leading0_vdsr = np.ma.masked_array(leading0_vdsr, mask=qm_mask)

            leading0_v3 = np.load(
                "/scratch/iu60/yl3101/PEFGAN/new_crps/save/crps_ss/" + model_name + "/" + year + '/' + model_num + "/lead_time" + str(time) + "_whole.npy")
            leading0_v3 = np.ma.masked_array(leading0_v3, mask=qm_mask)

            leading0_v3_log = np.load(
                "/scratch/iu60/yl3101/PEFGAN/new_crps/save/crps_ss_log/" + model_name + "/" + year + '/' + model_num + "/lead_time" + str(time) + "_whole.npy")
            leading0_v3_log = np.ma.masked_array(leading0_v3_log, mask=qm_mask)

            model_mae = np.load(
                "/scratch/iu60/yl3101/PEFGAN/new_crps/save/mae/" + model_name + "/" + year + '/' + model_num + "/lead_time" + str(time) + "_whole.npy")
            model_mae = np.ma.masked_array(model_mae, mask=qm_mask)

            model_bias = np.load(
                "/scratch/iu60/yl3101/PEFGAN/new_crps/save/bias/" + model_name + "/" + year + '/' + model_num + "/lead_time" + str(time) + "_whole.npy")
            model_bias = np.ma.masked_array(model_bias, mask=qm_mask)

            model_relative_bias = np.load(
                "/scratch/iu60/yl3101/PEFGAN/new_crps/save/relative_bias/" + model_name + "/" + year + '/' + model_num + "/lead_time" + str(time) + "_whole.npy")
            model_relative_bias = np.ma.masked_array(model_relative_bias, mask=qm_mask)

            model_brier0 = np.load(
                "/scratch/iu60/yl3101/PEFGAN/new_crps/save/Brier0/" + model_name + "/" + year + '/' + model_num + "/lead_time" + str(time) + "_whole.npy")
            model_brier0 = np.ma.masked_array(model_brier0, mask=qm_mask)

            model_brier95 = np.load(
                "/scratch/iu60/yl3101/PEFGAN/new_crps/save/Brier95/" + model_name + "/" + year + '/' + model_num + "/lead_time" + str(time) + "_whole.npy")
            model_brier95 = np.ma.masked_array(model_brier95, mask=qm_mask)

            model_brier99 = np.load(
                "/scratch/iu60/yl3101/PEFGAN/new_crps/save/Brier99/" + model_name + "/" + year + '/' + model_num + "/lead_time" + str(time) + "_whole.npy")
            model_brier99 = np.ma.masked_array(model_brier99, mask=qm_mask)
            # data missing(outbreak)
            # leading0_clima[370:490, 200:350] = 'nan'
            # leading0_clima[290:370, 250:400] = 'nan'
            # leading0_clima[340:400, 490:550] = 'nan'

            # crpsss_qm = np.zeros((691, 886))
            # crpsss_v3 = np.zeros((413, 267))
            # crpsss_bi = np.zeros((691, 886))
            # crpsss_vdsr = np.zeros((691, 886))

            # crpsss_bi = 1 - (leading0_bi / leading0_clima)
            
            # crpsss_vdsr = 1 - (leading0_vdsr / leading0_clima)
            crpsss_v3 = 1 - (leading0_v3 / leading0_clima)
            
            # crpsss_v3_log = 1 - (leading0_v3_log / leading0_clima_log)

            # bi
            # total_bi = total_bi + crpsss_bi
            #
            # # qm
            # total_qm = total_qm + crpsss_qm
            #
            # # vdsr
            # total_vdsr = total_vdsr + crpsss_vdsr

            # desrgan
            total_crps_list.append(crpsss_v3)
            total = total + crpsss_v3

            # line.append(np.nanmean(crpsss_bi))
            
            line.append(np.nanmean(leading0_v3))
            line.append(np.nanmean(crpsss_v3))
            # line.append(np.nanmean(crpsss_v3_log))
            # line.append(np.nanmean(crpsss_qm))
            line.append(np.nanmean(model_mae))
            line.append(np.nanmean(model_bias))
            line.append(np.nanmean(model_relative_bias))
            line.append(np.nanmean(model_brier0))  
            line.append(np.nanmean(model_brier95))  
            line.append(np.nanmean(model_brier99))
            csv_file.writerow(line)

            if draw:
                # AWAP lat and lon
                lat = [-39.2, -39.15, -39.100002, -39.05, -39.,
                       -38.95, -38.9, -38.850002, -38.8, -38.75,
                       -38.7, -38.65, -38.600002, -38.55, -38.5,
                       -38.45, -38.4, -38.350002, -38.3, -38.25,
                       -38.2, -38.15, -38.100002, -38.05, -38.,
                       -37.95, -37.9, -37.850002, -37.8, -37.75,
                       -37.7, -37.65, -37.600002, -37.55, -37.5,
                       -37.45, -37.4, -37.350002, -37.3, -37.25,
                       -37.2, -37.15, -37.100002, -37.05, -37.,
                       -36.95, -36.9, -36.850002, -36.8, -36.75,
                       -36.7, -36.65, -36.600002, -36.55, -36.5,
                       -36.45, -36.4, -36.350002, -36.3, -36.25,
                       -36.2, -36.15, -36.100002, -36.05, -36.,
                       -35.95, -35.9, -35.850002, -35.8, -35.75,
                       -35.7, -35.65, -35.600002, -35.55, -35.5,
                       -35.45, -35.4, -35.350002, -35.3, -35.25,
                       -35.2, -35.15, -35.100002, -35.05, -35.,
                       -34.95, -34.9, -34.850002, -34.8, -34.75,
                       -34.7, -34.65, -34.600002, -34.55, -34.5,
                       -34.45, -34.4, -34.350002, -34.3, -34.25,
                       -34.2, -34.15, -34.100002, -34.05, -34.,
                       -33.95, -33.9, -33.850002, -33.8, -33.75,
                       -33.7, -33.65, -33.600002, -33.55, -33.5,
                       -33.45, -33.4, -33.350002, -33.3, -33.25,
                       -33.2, -33.15, -33.100002, -33.05, -33.,
                       -32.95, -32.9, -32.850002, -32.8, -32.75,
                       -32.7, -32.65, -32.600002, -32.55, -32.5,
                       -32.45, -32.4, -32.350002, -32.3, -32.25,
                       -32.2, -32.15, -32.100002, -32.05, -32.,
                       -31.95, -31.900002, -31.85, -31.800001, -31.75,
                       -31.7, -31.650002, -31.6, -31.550001, -31.5,
                       -31.45, -31.400002, -31.35, -31.300001, -31.25,
                       -31.2, -31.150002, -31.1, -31.050001, -31.,
                       -30.95, -30.900002, -30.85, -30.800001, -30.75,
                       -30.7, -30.650002, -30.6, -30.550001, -30.5,
                       -30.45, -30.400002, -30.35, -30.300001, -30.25,
                       -30.2, -30.150002, -30.1, -30.050001, -30.,
                       -29.95, -29.900002, -29.85, -29.800001, -29.75,
                       -29.7, -29.650002, -29.6, -29.550001, -29.5,
                       -29.45, -29.400002, -29.35, -29.300001, -29.25,
                       -29.2, -29.150002, -29.1, -29.050001, -29.,
                       -28.95, -28.900002, -28.85, -28.800001, -28.75,
                       -28.7, -28.65, -28.6, -28.550001, -28.5,
                       -28.45, -28.4, -28.35, -28.300001, -28.25,
                       -28.2, -28.15, -28.1, -28.050001, -28.,
                       -27.95, -27.9, -27.85, -27.800001, -27.75,
                       -27.7, -27.65, -27.6, -27.550001, -27.5,
                       -27.45, -27.4, -27.35, -27.300001, -27.25,
                       -27.2, -27.15, -27.1, -27.050001, -27.,
                       -26.95, -26.9, -26.85, -26.800001, -26.75,
                       -26.7, -26.65, -26.6, -26.550001, -26.5,
                       -26.45, -26.4, -26.35, -26.300001, -26.25,
                       -26.2, -26.15, -26.1, -26.050001, -26.,
                       -25.95, -25.9, -25.85, -25.800001, -25.75,
                       -25.7, -25.65, -25.6, -25.550001, -25.5,
                       -25.45, -25.4, -25.35, -25.300001, -25.25,
                       -25.2, -25.15, -25.1, -25.050001, -25.,
                       -24.95, -24.9, -24.85, -24.800001, -24.75,
                       -24.7, -24.65, -24.6, -24.550001, -24.5,
                       -24.45, -24.4, -24.35, -24.300001, -24.25,
                       -24.2, -24.15, -24.1, -24.050001, -24.,
                       -23.95, -23.9, -23.85, -23.800001, -23.75,
                       -23.7, -23.65, -23.6, -23.550001, -23.5,
                       -23.45, -23.4, -23.35, -23.300001, -23.25,
                       -23.2, -23.15, -23.1, -23.050001, -23.,
                       -22.95, -22.9, -22.85, -22.800001, -22.75,
                       -22.7, -22.65, -22.6, -22.550001, -22.5,
                       -22.45, -22.4, -22.35, -22.300001, -22.25,
                       -22.2, -22.15, -22.1, -22.050001, -22.,
                       -21.95, -21.9, -21.85, -21.800001, -21.75,
                       -21.7, -21.65, -21.6, -21.550001, -21.5,
                       -21.45, -21.4, -21.35, -21.300001, -21.25,
                       -21.2, -21.15, -21.1, -21.050001, -21.,
                       -20.95, -20.9, -20.85, -20.800001, -20.75,
                       -20.7, -20.65, -20.6, -20.550001, -20.5,
                       -20.45, -20.4, -20.35, -20.300001, -20.25,
                       -20.2, -20.15, -20.1, -20.050001, -20.,
                       -19.95, -19.9, -19.85, -19.800001, -19.75,
                       -19.7, -19.65, -19.6, -19.550001, -19.5,
                       -19.45, -19.4, -19.35, -19.300001, -19.25,
                       -19.2, -19.15, -19.1, -19.050001, -19.,
                       -18.95, -18.9, -18.85, -18.800001, -18.75,
                       -18.7, -18.65, -18.6]
                lon = [140.6, 140.65001, 140.70001, 140.75, 140.8, 140.85,
                       140.90001, 140.95001, 141., 141.05, 141.1, 141.15001,
                       141.20001, 141.25, 141.3, 141.35, 141.40001, 141.45001,
                       141.5, 141.55, 141.6, 141.65001, 141.70001, 141.75,
                       141.8, 141.85, 141.90001, 141.95001, 142., 142.05,
                       142.1, 142.15001, 142.20001, 142.25, 142.3, 142.35,
                       142.40001, 142.45, 142.5, 142.55, 142.6, 142.65001,
                       142.7, 142.75, 142.8, 142.85, 142.90001, 142.95,
                       143., 143.05, 143.1, 143.15001, 143.2, 143.25,
                       143.3, 143.35, 143.40001, 143.45, 143.5, 143.55,
                       143.6, 143.65001, 143.7, 143.75, 143.8, 143.85,
                       143.90001, 143.95, 144., 144.05, 144.1, 144.15001,
                       144.2, 144.25, 144.3, 144.35, 144.40001, 144.45,
                       144.5, 144.55, 144.6, 144.65001, 144.7, 144.75,
                       144.8, 144.85, 144.90001, 144.95, 145., 145.05,
                       145.1, 145.15001, 145.2, 145.25, 145.3, 145.35,
                       145.40001, 145.45, 145.5, 145.55, 145.6, 145.65,
                       145.7, 145.75, 145.8, 145.85, 145.9, 145.95,
                       146., 146.05, 146.1, 146.15, 146.2, 146.25,
                       146.3, 146.35, 146.4, 146.45, 146.5, 146.55,
                       146.6, 146.65, 146.7, 146.75, 146.8, 146.85,
                       146.9, 146.95, 147., 147.05, 147.1, 147.15,
                       147.2, 147.25, 147.3, 147.35, 147.4, 147.45,
                       147.5, 147.55, 147.6, 147.65, 147.7, 147.75,
                       147.8, 147.85, 147.9, 147.95, 148., 148.05,
                       148.1, 148.15, 148.2, 148.25, 148.3, 148.35,
                       148.4, 148.45, 148.5, 148.55, 148.6, 148.65,
                       148.7, 148.75, 148.8, 148.85, 148.9, 148.95,
                       149., 149.05, 149.09999, 149.15, 149.2, 149.25,
                       149.3, 149.34999, 149.4, 149.45, 149.5, 149.55,
                       149.59999, 149.65, 149.7, 149.75, 149.8, 149.84999,
                       149.9, 149.95, 150., 150.05, 150.09999, 150.15,
                       150.2, 150.25, 150.3, 150.34999, 150.4, 150.45,
                       150.5, 150.55, 150.59999, 150.65, 150.7, 150.75,
                       150.8, 150.84999, 150.9, 150.95, 151., 151.05,
                       151.09999, 151.15, 151.2, 151.25, 151.3, 151.34999,
                       151.4, 151.45, 151.5, 151.55, 151.59999, 151.65,
                       151.7, 151.75, 151.8, 151.84999, 151.9, 151.95,
                       152., 152.05, 152.09999, 152.15, 152.2, 152.25,
                       152.29999, 152.34999, 152.4, 152.45, 152.5, 152.54999,
                       152.59999, 152.65, 152.7, 152.75, 152.79999, 152.84999,
                       152.9, 152.95, 153., 153.04999, 153.09999, 153.15,
                       153.2, 153.25, 153.29999, 153.34999, 153.4, 153.45,
                       153.5, 153.54999, 153.59999, 153.65, 153.7, 153.75,
                       153.79999, 153.84999, 153.9]
                # print(crpsss_v3[0])
                if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/image/" + model_name + "/"):
                        os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/image/"+ model_name)
                if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/image/" + model_name + "/" + str(year)):
                        os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/image/"+ model_name + "/" + str(year))
                if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/image/" + model_name + "/" + str(year) + "/window_" + str(
                                 window)):
                        os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/image/" + model_name + "/" + str(year) + "/window_" + str(
                                 window))
                if not os.path.exists("/scratch/iu60/yl3101/PEFGAN/new_crps/image/" + model_name + "/" + str(year) + "/window_" + str(
                                 window) + '/' + model_num):
                        os.mkdir("/scratch/iu60/yl3101/PEFGAN/new_crps/image/" + model_name + "/" + str(year) + "/window_" + str(
                                 window) + '/' + model_num)     
                title_dict = {"vdynamic_weights":"PEFGAN", "vVersion1":"DESRGAN version1", "vOriginal_DESRGAN":"DESRGAN 18 ensemble"}  
                draw_aus(crpsss_v3[0], mode='crps-ss', lat=lat, lon=lon, title="CRPS_SS_"+ title_dict[model_name] + "_" + str(time),
                             titles_on=True,
                             save=True,
                             path="/scratch/iu60/yl3101/PEFGAN/new_crps/image/" + model_name + "/" + str(year) + "/window_" + str(
                                 window) + '/' + model_num + "/CRPSSS_" + str(time) + ".jpeg")
                if time == 27:
                      fortnight_crps = total_crps_list[14:]
                      print("shape1: ", len(fortnight_crps))
                      print("shape1 mean: ", np.mean(fortnight_crps, axis = 0).shape)
                      draw_aus(np.mean(fortnight_crps, axis = 0).squeeze(), mode='crps-ss', lat=lat, lon=lon, title="CRPS_SS_14-27_" + title_dict[model_name],
                             titles_on=True,
                             save=True,
                             path="/scratch/iu60/yl3101/PEFGAN/new_crps/image/window_" + str(
                                 window) + "/crps_ss/" + year + "/" + title_dict[model_name]+ "_14-27.jpeg")
                if time == 41:
                      fortnight_crps = total_crps_list[28:]
                      print("shape2: ", len(fortnight_crps))
                      
                      draw_aus(np.mean(fortnight_crps, axis = 0).squeeze(), mode='crps-ss', lat=lat, lon=lon, title="CRPS_SS_28-41_" + title_dict[model_name],
                             titles_on=True,
                             save=True,
                             path="/scratch/iu60/yl3101/PEFGAN/new_crps/image/window_" + str(
                                 window) + "/crps_ss/" + year + "/" + title_dict[model_name]+ "_28-41.jpeg")
                      
                      draw_aus(np.mean(fortnight_crps, axis = 0).squeeze(), mode='crps-ss', lat=lat, lon=lon, title="CRPS_SS_28-41_" + title_dict[model_name],
                             titles_on=True,
                             save=True,
                             path="/scratch/iu60/yl3101/PEFGAN/new_crps/image/window_" + str(
                                 window) + "/crps_ss/" + year + "/" + title_dict[model_name]+ "_28-41_withcolorbar.jpeg", color_bar=True)
                # if time== 6 or time == 13 or time == 29 or time == 41:
                    # total_bi_save = total_bi / (time + 1)
                    # draw_aus(total_bi_save, mode='crps-ss', lat=lat, lon=lon, title="CRPS_SS_BI_" + str(time + 1),
                    #          titles_on=True,
                    #          save=True,
                    #          path="/scratch/iu60/rw6151/new_crps/image/window_" + str(
                    #              window) + "/crps_ss/" + year + "/BI_" + str(
                    #              time + 1) + ".pdf")
                    #
                    # total_qm_save = total_qm / (time + 1)
                    # draw_aus(total_qm_save, mode='crps-ss', lat=lat, lon=lon, title="CRPS_SS_QM_" + str(time + 1),
                    #          titles_on=True,
                    #          save=True,
                    #          path="/scratch/iu60/rw6151/new_crps/image/window_" + str(
                    #              window) + "/crps_ss/" + year + "/QM_" + str(
                    #              time + 1) + ".pdf")
                    #
                    # total_vdsr_save = total_vdsr / (time + 1)
                    # draw_aus(total_vdsr_save, mode='crps-ss', lat=lat, lon=lon, title="CRPS_SS_VDSR_" + str(time + 1),
                    #          titles_on=True,
                    #          save=True,
                    #          path="/scratch/iu60/rw6151/new_crps/image/window_" + str(
                    #              window) + "/crps_ss/" + year + "/VDSR_" + str(
                    #              time + 1) + ".pdf")
                    #
                    # total_save = total / (time + 1)
                    # draw_aus(total_save.squeeze(), mode='crps-ss', lat=lat, lon=lon, title="CRPS_SS_DESRGAN_" + str(time + 1),
                    #          titles_on=True,
                    #          save=True,
                    #          path="/scratch/iu60/yl3101/DESRGAN/new_crps/image/window_" + str(
                    #              window) + "/crps_ss/" + year + "/" + title_dict[model_name]+ "_" + str(
                    #              time + 1) + ".jpeg")


# evaluate("2009", 42, window=1, draw=True)
evaluate("2006", 42, window=1, draw=True)
evaluate("2007", 42, window=1, draw=True)
# evaluate("2010", 42, window=1, draw=True)
# evaluate("2012", 42, window=1, draw=True)
evaluate("2015", 42, window=1, draw=True)
evaluate("2018", 42, window=1, draw=True)

