from datetime import timedelta, date, datetime
import os
import time
import numpy as np
import xarray as xr

def date_range(start_date, end_date):
    """This function takes a start date and an end date as datetime date objects.
    It returns a list of dates for each date in order starting at the first date and ending with the last date"""
    return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

def get_date_without_en(rootdir, start_date, end_date):
    '''
    Check the file from ACCESS- S2 file folder e09, record the dates in the range [start_date, end_date]
    which emerged in e09 folder.
    '''

    _files = []
    dates = date_range(start_date, end_date)
    for date in dates:
        access_path = rootdir + "e09" + "/da_pr_" + date.strftime("%Y%m%d") + "_" + "e09" + ".nc"
        # print(access_path)
        if os.path.exists(access_path):
            _files.append(date)
    return _files

year = '2012'
e_number = 'e02'

rootdir = "/g/data/ux62/access-s2/hindcast/raw_model/atmos/pr/daily/"
# rootdir = "/scratch/iu60/yl3101/test_data/" + year + "/e01/"
start_date = date(int(year), 1, 1)
end_date = date(int(year), 12, 31)

# Due to the fact that the dates of the .nc files in different ensembles are also different, we use the dates
# from ensemble 9 to calibrate and filter valid .nc files with same dates from other ensembles

files_date = get_date_without_en(rootdir, start_date, end_date)
files_date.sort()
for date in files_date:
    fn = rootdir + e_number + "/da_pr_" + date.strftime("%Y%m%d") + "_" + e_number + ".nc"
    ds_raw = xr.open_dataset(fn) * 86400
    ds_raw = ds_raw.fillna(0)
    # ds_raw = np.clip(ds_raw, 0, 1000)
    # ds_raw = np.log1p(ds_raw) / 7
    da_selected = ds_raw.isel(time=0)["pr"]
    lon = ds_raw["lon"].values
    lat = ds_raw["lat"].values
    a = np.logical_and(lon >= 111.975, lon <= 156.275)
    b = np.logical_and(lat >= -44.525, lat <= -9.975)
    da_selected_au = da_selected[b, :][:, a].copy()
    i = ds_raw['time'].values[0]
    da_interp = xr.DataArray(da_selected_au, dims=("lat", "lon"),
                             coords={"lat": da_selected_au.lat.values, "lon": da_selected_au.lon.values, "time": i},
                             name='pr')
    ds_total = xr.concat([da_interp], "time")
    for i in ds_raw['time'].values[:30]:
        ds_selected_domained = ds_raw.sel(time=i)['pr']
        da_selected_au = ds_selected_domained[b, :][:, a].copy()
        da_interp = xr.DataArray(da_selected_au, dims=("lat", "lon"),
                                 coords={"lat": da_selected_au.lat.values, "lon": da_selected_au.lon.values, "time": i},
                                 name='pr')
        expanded_da = xr.concat([da_interp], "time")
        ds_total = xr.merge([ds_total, expanded_da])
    save_path = "/scratch/iu60/yl3101/Test_data/" + year + "/" + e_number + "/da_pr_" + date.strftime("%Y%m%d") + "_" + \
                                                                                                        e_number + ".nc"
    ds_total.to_netcdf(save_path)
    ds_raw.close()


