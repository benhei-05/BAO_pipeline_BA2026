#!/usr/bin/env python
"""
measure_pk.py -- pre-reconstruction power-spectrum multipoles of one DESI DR1
AbacusSummit cutsky mock (LRG, "complete"). First end-to-end smoke test, and
already the Phase-1 engine: the --subsample flag down-samples the DATA so you
can study how the BAO signal degrades with galaxy number.

Run on a COMPUTE node (~15 GB RAM for nmesh=512), via submit_smoke.slurm or an
interactive allocation -- NOT on the login node:

    python measure_pk.py --indir data/mock0 --cap NGC --outdir results
    python measure_pk.py --indir data/mock0 --subsample 0.25 --seed 1
"""
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
from lsstypes import Mesh2SpectrumPole, Mesh2SpectrumPoles
from desilike.theories.galaxy_clustering import BAOPowerSpectrumTemplate, DampedBAOWigglesTracerPowerSpectrumMultipoles
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.observables.galaxy_clustering import CutskyFootprint, ObservablesCovarianceMatrix
from desilike.samples import plotting

from pyrecon import IterativeFFTReconstruction







def read_catalog(path, zmin, zmax):
    """Return RA, DEC, Z and total weight (WEIGHT * WEIGHT_FKP if present)."""
    colnames = fitsio.FITS(path)[1].get_colnames()
    cols = ["RA", "DEC", "Z"] + [c for c in ("WEIGHT", "WEIGHT_FKP") if c in colnames]
    d = fitsio.read(path, columns=cols)
    sel = (d["Z"] >= zmin) & (d["Z"] < zmax)
    d = d[sel]
    w = np.ones(d.size)
    if "WEIGHT" in d.dtype.names:
        w *= d["WEIGHT"]
    if "WEIGHT_FKP" in d.dtype.names:
        w *= d["WEIGHT_FKP"]
    return d["RA"], d["DEC"], d["Z"], w

def volume(area, cosmo, zrange): #function from desilike covariance calculation
        r"""Volume, in :math:`(\mathrm{Mpc} / h)^{3}`."""
        volume = cosmo.comoving_radial_distance(zrange)**3
        return area / (180. / np.pi)**2 / 3. * np.diff(volume, axis=-1).sum()



def main():
    mock = 'mock0' # specifying the LRG mock to use, from 'mock0' to 'mock24'
    p = argparse.ArgumentParser()
    p.add_argument("--indir",
                   default=os.path.join(os.environ.get("HPCWORK", "."), "desi", mock))
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

    # subsample of the data to use; should be 1 to use the whole data set or <1 to just use as part
    subsample = 1
    seed = 42 
    print('mock =', mock)
    print('seed =', seed)

    b1_rec = 2.00 # linear galaxy bias used in reconstruction
    b1_cov = 2.00 # linear galaxy bias used for the analytical covariance estimation
    f = 0.83 # structure groth rate used in reconstruction
    sigma_sm = 15 # smoothing scale (in Mpc/h) used in reconstruction
    mode = 'recsym' # reconstruction mode; should be either 'recsym' or 'reciso'

    apmode= 'qiso' # Alcock-Paczynski parameterization for the BAO fit

    print('b1_rec =', b1_rec)
    print('b1_cov =', b1_cov)
    print('f =', f)
    print(f'sigma_sm = {sigma_sm} Mpc/h')
    print('mode =', mode)
    print('apmode =', apmode)

    # DESI fiducial cosmology == the mock's TRUE cosmology (AbacusSummit c000),
    # so there is no Alcock-Paczynski distortion here. Distances in Mpc/h.
    # (If the class engine fails to build, try engine="camb" or "astropy".)
    cosmo = DESI(engine="class")
    dist = cosmo.comoving_radial_distance

    # redshift bin 
    zmin = 0.4 
    zmax = 1.1 

    
    

    # --- data ---------------------------------------------------------------
    
    dfile = os.path.join(os.path.join(os.environ.get("HPCWORK", "."), "desi", mock), 
                         f"LRG_complete_{args.cap}_clustering.dat.fits")
    ra, dec, z, w = read_catalog(dfile, zmin, zmax) # right ascension, declination, redshift and weight
    if subsample < 1.0: # just keeps random amount of subsample of the data
        rng = np.random.default_rng(seed)
        keep = rng.random(ra.size) < subsample
        ra, dec, z, w = ra[keep], dec[keep], z[keep], w[keep]
    data_pos = [ra, dec, dist(z)]          # position_type='rdd' wants a list of 3 arrays
    N_gal = ra.size # number of galaxies
    print(f"[data]    N = {N_gal:,}   (subsample = {subsample})")

    # --- randoms (concatenate every *_clustering.ran.fits for this cap) ------
    rfiles = sorted(glob.glob(os.path.join(
        os.path.join(os.environ.get("HPCWORK", "."), "desi", mock),
        f"LRG_complete_{args.cap}_*_clustering.ran.fits")))
    if not rfiles:
        raise SystemExit(f"No random files in {args.indir} for cap {args.cap}")
    R = [read_catalog(f, zmin, zmax) for f in rfiles]
    rra = np.concatenate([r[0] for r in R])
    rdec = np.concatenate([r[1] for r in R])
    rz = np.concatenate([r[2] for r in R])
    rw = np.concatenate([r[3] for r in R])
    rand_pos = [rra, rdec, dist(rz)]       # position_type='rdd' wants a list of 3 arrays
    N_ran = rra.size # number of random galaxies
    N_cat_ran = len(rfiles) # number of random catalogs
    print(f"[randoms] N = {N_ran:,}   from {N_cat_ran} file(s)")



    # --- reconstruction ---------------------------------------------------------------
    
    A_deg = N_ran / N_cat_ran / 2500 
    # angular area of the survey in deg^2; random files have a density of 2500 deg^(-2)
    
    V = volume(area=A_deg, cosmo=cosmo, zrange=(zmin, zmax))
    # survey volume in (Mpc/h)^3; value is not needed for the code but provided for future analysis
    
    print(f'A_deg = {A_deg:.2f} deg^(-2)' )
    print(f'V = {V:.2f} (Mpc/h)^3')


    '''
    The following code block carries out the reconstruction. There are four different reconstruction
    algorithms provided in pycorr. This code uses IterativeFFTReconstruction as suggested by the DESI
    publication, but the other algorithms can be implemented as well.
    The data is provided twice to make the code more robust against acurring errors. The shifted random
    position calculation depends on wether mode is set to 'recsym' or 'reciso'.
    '''



    nmesh = args.nmesh # number of mesh nodes along each axis.
    boxpad = 1.5 # boxsize is to be calculated as boxpad * smallest possible box
    position_type = "rdd"
    los = "firstpoint" # defines line of sight; firstpoint -> uses first point 
    resampler = "tsc" # assigns particles to mesh; 
    
    
    recon = IterativeFFTReconstruction(f=f, bias=b1_rec, 
                                       data_positions=data_pos, data_weights=w,
                                       randoms_positions=rand_pos,randoms_weights=rw,
                                       nmesh=nmesh, boxpad=boxpad, position_type=position_type, 
                                       los=None, resampler=resampler, dtype="f8")

    data_pos = np.asarray(data_pos, dtype=np.float64)
    rand_pos = np.asarray(rand_pos, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    rw = np.asarray(rw, dtype=np.float64)

    # providing the data to recon in an acceptable form
    recon.assign_data(data_pos, w, dtype="f8")
    recon.assign_randoms(rand_pos, rw)

    # setting the smoothing scale
    recon.set_density_contrast(smoothing_radius=sigma_sm)
    
    recon.run()


    positions_rec_data = recon.read_shifted_positions(data_pos)

    if mode == 'recsym':
        positions_rec_randoms = recon.read_shifted_positions(rand_pos)
    elif mode == 'reciso':
        positions_rec_randoms = recon.read_shifted_positions(rand_pos, field='disp')

    

    

    # --- power-spectrum multipoles (FFT / Yamamoto estimator) ---------------
   
    '''
    The next part of the code calculates the power spectrum multipoles, now based on the reconstructed
    catalogs. Some of the input parameters in CatalogFFTPower are the same as the ones used in 
    reconstruction above.
    '''

    ells=(0, 2, 4) # multipole orders
    kedges = np.arange(0.0, 0.4, 0.005) # start, stop and steps for edges of k-values for poles

    result = CatalogFFTPower(
        data_positions1=positions_rec_data, data_weights1=w,
        randoms_positions1=positions_rec_randoms, randoms_weights1=rw,
        position_type=position_type, edges=kedges, ells=ells, los=los, 
        nmesh=nmesh, resampler=resampler, 
        interlacing=2, # reduces aliasing when resampling; order 2
        boxpad=boxpad, dtype="f8")

    # saving result with a name containing all relevant parameters for future uses
    tag = f"LRG_{os.path.basename(args.indir)}_{args.cap}_sub{subsample:.2f}"
    result.save(os.path.join(args.outdir, f"pk_{mode}_b1{b1_rec}{b1_cov}_f{f}_ssm{sigma_sm}_{tag}.npy"))

    # quick calculation of the Nyquist frequency in h/Mpc (not needed for the code)
    boxsize = result.boxsize
    print(f'boxsize = {boxsize[0]:.2f} Mpc/h')
    nyq = nmesh * np.pi / boxsize
    print(f'nyq = {nyq[0]:.2f} h/Mpc')

    # extracting the k and multipole arrays from result
    poles = result.poles
    k = poles.k # array of k-values
    P0 = np.real(poles(ell=0, remove_shotnoise=True)) # arrays for multipoles of power spectrum
    P2 = np.real(poles(ell=2))
    P4 = np.real(poles(ell=4))


    # --- fitting ---------------------------------------------------------------
    
    '''
    The arrays returned from result proved to contain 'nan' entries (for high k values) which could not be 
    dealt with inthe BAO fitting process, so they are removed here instead.
    '''
    cleaning = ~np.isnan(k)
    k, P0, P2, P4 = k[cleaning], P0[cleaning], P2[cleaning], P4[cleaning] 
    print('Size of cleaned arrays:', len(P0))

    
    clean_poles = []
    clean_poles.append(P0)
    clean_poles.append(P2)
    clean_poles.append(P4)
    
    mean = [Mesh2SpectrumPole(k=k, num_raw=value, ell=ell) for ell, value in zip(ells, clean_poles)]
    fit_data = Mesh2SpectrumPoles(mean) # k and poles as suitable data type

    # estimating the effective redshift as a weighted mean
    z_eff = np.average(np.array(z), weights=np.array(w)) 
    print('z_eff =', z_eff)

    
    # setting the BAO template, containing the linear power spectrum
    template = BAOPowerSpectrumTemplate(z=z_eff, fiducial='DESI', apmode=apmode)

    # implementing the theory that provides the theoretical galaxy power spectrum multipoles
    theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(template=template, k=k, ells=ells,
                                                           mode=mode, smoothing_radius=sigma_sm,
                                                           broadband='power') 
    
    # the observable is the desilike object linking the measured power spectrum with the theoretical
    # one (and the covariance matrix if we already had one)
    observable = TracerPowerSpectrumMultipolesObservable(data=fit_data, covariance=None, theory=theory)
    

    

    '''
    If there is no external covariance data, an analytical covariance can be calculated with desilike
    by defining a footprint (in this case the CutskyFootprint that needs the number density in deg^(-2),
    the area in deg^2 and the redshift bin). If the number density is known to differ along the redshifts,
    zrange and nbar can also be provided as arrays; in this case nbar has to be in (h/Mpc)^3.
    To get the actual covariance matrix, in the last step the galaxy bias has to provided at which the
    theoretical power spectrum in the calculation is to be evaluated at.
    '''
    footprint = CutskyFootprint(nbar=N_gal/A_deg, area=A_deg, cosmo=cosmo, zrange=(zmin, zmax)) 
    
    covariance = ObservablesCovarianceMatrix(observables=[observable], footprints=[footprint])
    cov = covariance(b1=b1_cov)   # evaluate covariance matrix at this parameter
  
    
    '''
    The last steps for the BAO fitting are to define a likelihood from the observable and the covariance
    and maximizing it with a minuit profiler. The fitting results are printed in a table.
    '''
    
    likelihood = ObservablesGaussianLikelihood(observables=[observable], covariance=cov) 

    profiler = MinuitProfiler(likelihood, seed=42)

    profiles = profiler.maximize(niterations=3)

    print(profiles.to_stats(tablefmt='pretty'))



    '''
    To get an overwiev of the fitting results, the fits can be plotted together with the input data.
    observable.plot shows the full power spectrum multipoles while observable.plot_bao extracts the BAO 
    wiggles.
    '''

    likelihood(**profiles.bestfit.choice(input=True))
    observable.plot(fn=os.path.join(args.outdir, "plot_rec_observable.png"))
    observable.plot_bao(fn=os.path.join(args.outdir, "plot_rec_BAO.png"))
    
    
    
  
                                                            
    
    
    
    np.savetxt(os.path.join(args.outdir, f"pk_{tag}.txt"),
               np.column_stack([k, P0, P2, P4]),
               header="k[h/Mpc]  P0  P2  P4   [(Mpc/h)^3]")

    # --- quick-look plot ----------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(k, k * P0, label=r"$kP_0$")
    plt.plot(k, k * P2, label=r"$kP_2$")
    plt.axvspan(0.05, 0.15, color="k", alpha=0.05)  # rough BAO band -- look for wiggles here
    plt.xlabel(r"$k\ [h/\mathrm{Mpc}]$")
    plt.ylabel(r"$k\,P_\ell(k)\ [(\mathrm{Mpc}/h)^2]$")
    plt.legend(); plt.title(tag); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"recon_pk_{tag}.png"), dpi=150)
    print(f">>> wrote {args.outdir}/pk_{tag}.{{npy,txt,png}}")



   


if __name__ == "__main__":
    main()
