import numpy as np
from beamformer.utilities.common import find_closest_pointing, get_data_list
from beamformer.strategist.strategist import PointingStrategist
from sps_databases import db_api, db_utils
import datetime

ra = 120.0
dec = 32.1
mode = 'local'
mode = 'database'
db = db_utils.connect(host='sps-archiver1', name='test')
ap = find_closest_pointing(ra, dec, mode=mode)
print(ap.maxdm)
