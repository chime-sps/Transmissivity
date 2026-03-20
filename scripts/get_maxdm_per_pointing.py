import numpy as np
import json
from sps_databases import db_utils
from beamformer.utilities.common import find_closest_pointing

def get_max_dm(ra, dec):
    db_mode = 'database'
    db = db_utils.connect(host='sps-archiver1', name='test')
    ap = find_closest_pointing(ra, dec, mode=db_mode)
    return ap.maxdm

pointings = np.load('stack_pointings.npy')
maxdm_dict = {}

for i in range(len(pointings)):
    print(f'Running at RA = {pointings[i, 0]}, Dec = {pointings[i, 1]}.')
    maxdm = get_max_dm(pointings[i, 0], pointings[i, 1])
    print(f'Maximum DM is {maxdm} pcc.')
    maxdm_dict[f'{pointings[i, 0]} {pointings[i, 1]}'] = maxdm

with open('stack_maxdm.json', 'w') as json_file:
    json.dump(maxdm_dict, json_file, indent=4)

