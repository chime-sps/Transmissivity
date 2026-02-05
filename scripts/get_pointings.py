import numpy as np
from beamformer.utilities.common import find_closest_pointing, get_data_list
from beamformer.strategist.strategist import PointingStrategist
from sps_databases import db_api, db_utils
import datetime
from multiprocessing import Pool
from functools import partial
from sps_pipeline.processing import find_active_pointings

def get_pointings(day, full_transit, db_name, db_host, db_port, beams):

    strat = PointingStrategist(create_db=False)

    first_coordinates = None
    last_coordinates = None
    with Pool(32) as pool:
        active_pointings_list = pool.map(
            partial(
                find_active_pointings,
                day=day,
                strat=strat,
                full_transit=full_transit,
                db_name=db_name,
                db_host=db_host,
                db_port=db_port,
            ),
            beams,
        )

    active_pointings = [
        ap for ap_list in active_pointings_list for ap in ap_list
    ]

    if len(active_pointings) >= 1:
        first_coordinates = (
            active_pointings[0].ra,
            active_pointings[0].dec,
        )
        last_coordinates = (
            active_pointings[-1].ra,
            active_pointings[-1].dec,
        )

    return active_pointings

date_path = '/mnt/beegfs-client/raw/2025/11/16'
db_mode = 'database'
db_host = 'sps-archiver1'
db_name = 'test'
db_port = 27017
db = db_utils.connect(host='sps-archiver1', name='test')
beams = [0, 1, 2, 20, 21, 22, 40, 41, 42, 60, 61, 62, 80, 81, 82, 100, 101, 102, 120, 121, 122, 140, 141, 142, 160, 161, 162, 180, 181, 181, 
    200, 201, 202, 220, 201, 202]
active_pointings = get_pointings(date_path, True, db_name, db_host, db_port, beams)
pointings = np.zeros((len(active_pointings), 3))
for i in range(len(active_pointings)):
    pointings[i] = np.array([active_pointings[i].ra, active_pointings[i].dec, active_pointings[i].sub_pointing])

np.save('20251116_pointings.npy', pointings)
