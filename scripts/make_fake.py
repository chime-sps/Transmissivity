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
        default = "random",
        help = ("Path to injection profile npy file")
)

@click.option(
        "--focus",
        default = None,
        help = ("Iterates over selected field (sigma, flux, frequency, or DM).")
)

@click.option(
        "--tpa_idx",
        "--i",
        type = int,
        default = None,
        help = ("Index of profile in TPA dataset.")
)

@click.option(
        "--f",
        type = float,
        default = None,
        help = ("Set frequency.")
    )

@click.option(
        "--dm",
        type = float,
        default = None,
        help = ("Set DM.")
)

@click.option(
        "--flux",
        type = float,
        default = None,
        help = ("Set flux in mJy.")
)

def get(n_injections, file_name, injection_path, focus, tpa_idx, f, dm, flux):
    make_yaml(n_injections, file_name, injection_path, focus, tpa_idx, f, dm, flux)

def make_yaml(n_injections, file_name, injection_path, focus, tpa_idx, f, dm, flux):
    if injection_path != 'random' and injection_path not in ['TPA', 'tpa']:
        load_profs = np.load(injection_path)
        if len(load_profs.shape) == 1:
            n_injections = 1
        else:
            n_injections = len(load_profs)
    
    if injection_path in ['TPA', 'tpa']:
        profiles = np.load('../profiles/smoothed_baselined_TPA_pulses.npy')
        tpa_idx = np.array(tpa_idx)
        frequencies = np.array(f)
        dms = np.array(dm)
        fluxes = np.array(flux)
        sigmas = None

    elif focus == 'frequency' or focus == 'freq':
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
        profiles = np.load('../Transmissivity/profiles/smoothed_baselined_TPA_pulses.npy')
        sigmas = np.random.uniform(10, 17, n_injections)
        frequencies = np.random.uniform(10, 100, n_injections)
        dms = np.random.uniform(3, 200, n_injections)
        fluxes = None

    data = []
    print(f"Creating {n_injections} fake pulsars into {injection_path}")
    
    for i in range(n_injections):
        
        n_dict = {}

        # .item() allows a simpler output in the yaml file
        # alternatively could use float()
        n_dict['frequency'] = frequencies[i].item()
        n_dict['DM'] = dms[i].item()
        if type(fluxes) == type(None):
            n_dict['sigma'] = sigmas[i].item()
        else:
            n_dict['flux'] = fluxes[i].item()

        #print(f"{i}: {n_dict}")
        if injection_path == 'random':
            tpa_idx = np.random.choice(range(len(profiles)))
            print(f'Your randomly assigned TPA index is {tpa_idx}.')
            n_dict['profile'] = profiles[tpa_idx].tolist()
        
        elif injection_path in ['TPA', 'tpa']:
            n_dict['profile'] = profiles[tpa_idx[i]].tolist()
        else:
            if n_injections == 1:
                n_dict['profile'] = load_profs.tolist()
            else:
                n_dict['profile'] = load_profs[i].tolist()
        
        data.append(n_dict)
    file_name = file_name
    stream = open(file_name, 'w')
    yaml.dump(data, stream)

if __name__ == "__main__":
    get()
