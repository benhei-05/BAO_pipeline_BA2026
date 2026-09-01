#!/usr/bin/env python

import os
import glob
import argparse
import numpy as np
import fitsio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cosmoprimo.fiducial import DESI
from pypower import CatalogFFTPower, setup_logging


from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
from desilike.profilers import MinuitProfiler
from desilike.theories.galaxy_clustering import BAOPowerSpectrumTemplate, DampedBAOWigglesTracerPowerSpectrumMultipoles
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.samples import plotting

def main():
    

    p = argparse.ArgumentParser()
    p.add_argument("--indir",
                   default=os.path.join(os.environ.get("HPCWORK", "."), "desi", "mock0"))
    p.add_argument("--cap", default="NGC")
    p.add_argument("--outdir", default="results")
    p.add_argument("--zmin", type=float, default=0.4)
    p.add_argument("--zmax", type=float, default=1.1)
    p.add_argument("--nmesh", type=int, default=512)
    p.add_argument("--subsample", type=float, default=1.0,
                   help="keep this fraction of the DATA (the Phase-1 knob)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    setup_logging()


    # DESI fiducial cosmology == the mock's TRUE cosmology (AbacusSummit c000),
    # so there is no Alcock-Paczynski distortion here. Distances in Mpc/h.
    # (If the class engine fails to build, try engine="camb" or "astropy".)
    cosmo = DESI(engine="class")
    dist = cosmo.comoving_radial_distance

    # redshift bin
    zmin = 0.6
    zmax = 0.8

    
    

    
    #loading the data
    data = np.load('cov_LRG_NGC_z0.6-0.8.npz')

    nmocks = data['nmocks'] # number of mocks
    cov = data['cov'] / nmocks # covariance matrix; devided by nmocks since data is mean from nmocks measurements
    mean = data['mean'] # multipoles in flat array
    k = data['k'] # k
    ells = data['ells'] # multipole orders

    # wavenumber limits
    kmin = data['kmin']
    kmax = data['kmax']
    dk = data['dk']

    # there is no effective redshift given, so it is estimated as mean of the bin
    z_eff = (zmax + zmin) / 2 

    # setting the BAO template, containing the linear power spectrum
    template = BAOPowerSpectrumTemplate(z=z_eff, fiducial='DESI', apmode='qiso')

     # implementing the theory that provides the theoretical galaxy power spectrum multipoles
    theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(template=template, k=k, ells=ells, broadband='power') 

    # the observable is the desilike object linking the measured power spectrum with the theoretical
    # one and the covariance matrix 
    observable = TracerPowerSpectrumMultipolesObservable(data=mean, covariance=cov, ells=ells, k=k, 
                                                         klim={0: (kmin, kmax, dk), 2: (kmin, kmax, dk)}, 
                                                         theory=theory)


    '''
    The last steps for the BAO fitting are to define a likelihood from the observable and the covariance
    and maximizing it with a minuit profiler. The fitting results are printed in a table.
    '''
    
    likelihood = ObservablesGaussianLikelihood(observables=[observable]) 

    profiler = MinuitProfiler(likelihood, seed=42)

    profiles = profiler.maximize(niterations=3)

    print(profiles.to_stats(tablefmt='pretty'))


    '''
    To get an overwiev of the fitting results, the fits can be plotted together with the input data.
    observable.plot shows the full power spectrum multipoles while observable.plot_bao extracts the BAO 
    wiggles.
    '''
    
    likelihood(**profiles.bestfit.choice(input=True))
    observable.plot(fn=os.path.join(args.outdir, "plot_cov_LRG_NGC_z0.6-0.8.png"))
    observable.plot_bao(fn=os.path.join(args.outdir, "plot_BAO_cov_LRG_NGC_z0.6-0.8.png"));
    
    

    
    
    
    

   
    
    
     





if __name__ == "__main__":
    main()

                                                         
