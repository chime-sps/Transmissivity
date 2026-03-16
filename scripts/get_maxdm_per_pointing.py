import numpy as np
from sps_databases import db_utils
from beamformer.utilities.common import find_closest_pointing

def get_max_dm(ra, dec):
    db_mode = 'database'
    db = db_utils.connect(host='sps-archiver1', name='test')
    ap = find_closest_pointing(ra, dec, mode=db_mode)
    return ap.maxdm
