#this is the new version of make_fake.py that takes an npz file as input instead of an npy file
#use this if you are including width!!!

from ps_processes.processes import ps_inject
import click
import yaml
import os
import numpy.random as rand
import numpy as np

@click.command()
@click.option(
        "--n-injections",
        "--n",
        default = 1,
        type = int,
        help = ("Number of injections")
)

@click.option(
        "--file-name",
        "--fn",
        default = "test_injections.yml",
        type = str,
        help = ("Name of target file")
)

@click.option(
        "--injection-path",
        "--path",
        default = 'TPA_pulses.npz',
        help = ("Path to injection profile npz file")
)

@click.option(
        "--focus",
        default = None,
        help = ("Iterates over selected field (sigma, flux, frequency, or DM).")
)


def get(n_injections, file_name, injection_path, focus):
    
    if injection_path != 'random':
        load_profs = np.load(injection_path)
        if len(load_profs.shape) == 1:
            n_injections = 1
        else:
            n_injections = len(load_profs)
    
    if focus == 'frequency' or focus == 'freq':
        #frequencies = np.logspace(1.8, 2.3, n_injections)
        frequencies = np.logspace(-2, 2.4, n_injections)
        dms = 57.3817479147*np.ones(n_injections)
        fluxes = 1*np.ones(n_injections)
        sigmas = None

    elif focus == 'dm' or focus == 'DM':
        dms = np.linspace(3, 200, n_injections)
        frequencies = 8.138748235982394*np.ones(n_injections)
        fluxes = 1*np.ones(n_injections)
        sigmas = None

    elif focus == 'sigma' or focus == 'sig':
        sigmas = np.linspace(6, 17, n_injections)
        dms = 107.3817479147*np.ones(n_injections)
        frequencies = 8.138748235982394*np.ones(n_injections) 
        fluxes = None

    elif focus == 'duty':
        frequencies = 8.138748235982394*np.ones(n_injections)
        dms = 107.3817479147*np.ones(n_injections)
        #sigmas = 11.28372911*np.ones(n_injections)
        fluxes = 1*np.ones(n_injections)
        sigmas = None

    elif focus == 'flux':
        frequencies = 8.138748235982394*np.ones(n_injections)
        dms = 107.3817479147*np.ones(n_injections)
        fluxes = np.linspace(0.05, 5, n_injections)
        sigmas = None

    else:
        sigmas = np.random.uniform(10, 17, n_injections)
        frequencies = np.random.uniform(10, 100, n_injections)
        dms = np.random.uniform(3, 200, n_injections)
        fluxes = None

    data = []
    print(f"Creating {n_injections} fake pulsars into {injection_path}")
    
    for i in range(n_injections):
        
        n_dict = {}

        # .item() allows a simpler output in the yanl file
        # alternatively could use float()
        n_dict['frequency'] = frequencies[i].item()
        n_dict['DM'] = dms[i].item()
        if fluxes == None:
            n_dict['sigma'] = sigmas[i].item()
        else:
            n_dict['flux'] = fluxes[i].item()

        print(f"{i}: {n_dict}")
        if injection_path == 'random':
            n_dict['profile'] = ps_inject.generate_pulse().tolist()

        else:
            if n_injections == 1:
                n_dict['profile'] = load_profs.tolist()
            else:
                n_dict['profile'] = load_profs[i].tolist()
        
        data.append(n_dict)

    file_name = os.getcwd()+'/'+file_name
    stream = open(file_name, 'w')
    yaml.dump(data, stream)

if __name__ == "__main__":
    get()
