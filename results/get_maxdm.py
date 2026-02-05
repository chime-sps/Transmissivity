import numpy as np
from beamformer.utilities.common import find_closest_pointing, get_data_list
from beamformer.strategist.strategist import PointingStrategist
from sps_databases import db_api, db_utils
import datetime

#RA	Dec	Date	TPA_idx	f		DM		S			FWHM-dependent quantity	predict_sigma	 
data = np.loadtxt('all_real_output.txt', skiprows = 1)
ra = data[:, 0]
dec = data[:, 1]
date = data[:, 2]
dm = data[:, 5]

data_11_13 = data[date == 20251113.0]
pointings = np.unique(data_11_13[:, 0:2], axis = 0)
maxdm = np.zeros(len(pointings))

mode = 'database'
db = db_utils.connect(host='sps-archiver1', name='test')
with open('maxdm_per_pointing.txt', 'w') as f:
    for i in range(len(pointings)):
        ap = find_closest_pointing(pointings[i, 0], pointings[i, 1], mode=mode)
        f.write(f'{pointings[i, 0]} {pointings[i, 1]} {ap.maxdm}\n')



