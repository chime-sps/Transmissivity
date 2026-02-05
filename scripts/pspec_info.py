from ps_processes.processes.ps_stack import PowerSpectraStack
from sps_common.interfaces import PowerSpectra

pspec_path = 'blank_pspec.hdf5'
pspec = PowerSpectra.read(pspec_path)
print(min(pspec.dms), max(pspec.dms))
