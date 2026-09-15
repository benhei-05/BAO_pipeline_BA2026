import numpy as np
import matplotlib.pyplot as plt
from desilike.theories.galaxy_clustering import BAOPowerSpectrumTemplate, DampedBAOWigglesTracerPowerSpectrumMultipoles
from scipy.integrate import simpson
import os
import argparse
import glob
import fitsio
from desilike.theories import Cosmoprimo
from scipy.interpolate import interp1d

from cosmoprimo.fiducial import DESI


mock = 'mock0'
p = argparse.ArgumentParser()
p.add_argument("--indir", default=os.path.join(os.environ.get("HPCWORK", "."), "desi", mock))
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
#setup_logging()

b1 = 2.0 # linear galaxy bias to later estimate the galaxy power spectrum
z_eff = 0.786 # effective redshift simulating the one from the measurements with full_BAO_pipeline.py
V = 6220589819.16 # survey volume in (Mpc/h)^3 from full_BAO_pipeline.py

# the following values are taken from https://arxiv.org/pdf/2002.04035
S_NL = 3.0 # non-linear damping Mpc/h
S_Silk = 5 # Silk damping in Mpc/h
l_BAO = 105 # BAO scale in Mpc/h

alpha = 1.0 # fiducial value of alpha_iso

Cosmo = Cosmoprimo(engine='class')
Cosmo()


cosmo = DESI(engine="class")

rd = cosmo.rs_drag

print(f"r_d = {rd:.2f} Mpc/h")



k_grid = np.linspace(0.03, 0.3, 100) # k array


# setting the BAO template, containing the linear power spectrum
template = BAOPowerSpectrumTemplate(z=z_eff, k=k_grid, fiducial='DESI', apmode='qiso')
template.init.update(cosmo=Cosmo)
template()

P = template.pk_dd # extracting an array containing the linear power spectrum from the template      
P_no_wiggle = template.pknow_dd # this returns only the no-wiggle part of the linear power spectrum

P_wiggle = P - P_no_wiggle # wiggle part of the power spectrum

P = np.array(P)
P_wiggle = np.array(P_wiggle)
P_no_wiggle = np.array(P_no_wiggle)





P_nw = interp1d(k_grid, P_no_wiggle, kind='cubic', bounds_error=False, fill_value="extrapolate")

def P_w(k, Sigma_Silk, Sigma_NL, l): # model of the wiggle part of the power spectrum
    exp = np.exp(- 0.5 * k**2 * (Sigma_Silk**2 + Sigma_NL**2))
    sin = np.sin(k * l / alpha)
    return 0.05 * P_nw(k) * sin * exp


# comparing the P_w model with different parameters with the template P_w

model = P_w(k_grid, S_Silk, S_NL, l_BAO) # standard model
model_rd = P_w(k_grid, S_Silk, S_NL, rd) # using the cosmoprimo sound horizon instead
model_snl0 = P_w(k_grid, S_Silk, 0, l_BAO) # standard model with S_NL=0
model_rd_snl0 = P_w(k_grid, S_Silk, 0, rd) # cosmoprimo sound horizon and S_NL=0


fig_pw, ax_pw = plt.subplots()

ax_pw.plot(k_grid, P_wiggle, label='true', alpha=0.5)
ax_pw.plot(k_grid, model, label=r'model $l_{BAO}$', alpha=0.5, color='red')
ax_pw.plot(k_grid, model_rd, label='model cosmoprimo', alpha=0.5, color='purple')
ax_pw.plot(k_grid, model_snl0, label=r'$l_{BAO}$ $\Sigma_{NL}=0$', alpha=0.5, color='brown')
ax_pw.plot(k_grid, model_rd_snl0, label=r'cosmoprimo $\Sigma_{NL}=0$', alpha=0.5, color='yellow')
ax_pw.legend()
ax_pw.set_xlabel('k [h/Mpc]')
ax_pw.set_ylabel(r'$P_w$ [(Mpc/h)^3]')
ax_pw.grid()


fig_pw.savefig(os.path.join(args.outdir, "Pw_comparison.png"), dpi=150)





def V_eff(N_gal): # effective volume for constant number density
    P_gal = b1**2 * P # estimation of the galaxy power spectrum ignoring RSD 
    return ( P_gal * N_gal / ( V + P_gal * N_gal ) )**2 * V



def w(N_gal): # weight function for the Fisher matrix integral
    return V_eff(N_gal) * (k_grid / (2*np.pi))**3




def dlnP_dalpha(Sigma_Silk, Sigma_NL, l): # derivative following the P_w model from above
    cos = np.cos(k_grid * l / alpha)
    exp = np.exp(- 0.5 * k_grid**2 * (Sigma_Silk**2 + Sigma_NL**2))
    return - 0.05 * k_grid * l / alpha**2 * cos * exp

def F(N_gal, Sigma_Silk, Sigma_NL, l): # Fisher matrix approximation
    integrand = dlnP_dalpha(Sigma_Silk, Sigma_NL, l)**2 * w(N_gal) 
    lnk_grid = np.log(k_grid)
    return 2*np.pi * simpson(y=integrand, x=lnk_grid)


def s_qiso(N_gal, Sigma_Silk, Sigma_NL, l): # sigma_alpha_iso prediction from the Fisher matrix
    return 1 / np.sqrt(F(N_gal, Sigma_Silk, Sigma_NL, l))







#plotting the resulting predictions for both the standard model and the cosmoprimo sound horizon

fig, ax = plt.subplots()

        
N_min = 200000
N_max = 2500000
N_gal_val = np.linspace(N_min, N_max, 100)

s_qiso_val = []




for N_gal in N_gal_val:
    s_qiso_val.append(s_qiso(N_gal, S_Silk, S_NL, l_BAO))

ax.plot(N_gal_val, s_qiso_val, label=r'model $l_{BAO}$', alpha=0.5, color='red')



s_qiso_val = []

for N_gal in N_gal_val:
    s_qiso_val.append(s_qiso(N_gal, S_Silk, S_NL, rd))

ax.plot(N_gal_val, s_qiso_val, label='model cosmoprimo', alpha=0.5, color='purple')



# plotting the measured sigma_alpha_iso values from the tests with full_BAO_pipeline.py for comparison


def f(x, A, B):
    return  (A + B * V / x)

def plot(x, y, err_y, err_x, label, c, cf, fitting=False):
    ax.errorbar(x, y, yerr=err_y, xerr=err_x, linestyle='', marker='o', ms=1, label=f'{label} results', 
                alpha=0.5, color=c)


    if fitting == True:
        popt, pcov = sp.optimize.curve_fit(f, x, y, sigma=err_y, absolute_sigma=True)
        A = popt[0]
        B = popt[1]
            #C = popt[2]
        fit = f(x, A, B)
        
            #print(C)
        
        res = y - fit
        chiq = np.sum((res/err_y)**2)
        ndof = len(x) - len(popt)
        print(label)
        print('A = ', A, ' +/- ', np.sqrt(pcov[0,0]))
        print('B = ', B, ' +/- ', np.sqrt(pcov[1,1]))
        print(f'chiq/ndof = {chiq:.2f} / {ndof:.1f} = {chiq/ndof:.2f}')
        print()
        
        ax[0].plot(x, fit, label=f'{label} Fit', color=cf)
        ax[1].errorbar(x, res, yerr=err_y, linestyle='', marker='o', ms=2, label=label, alpha=0.5, color=c)




N_0125 = np.array([309722, 309722, 309965, 309021, 310121, 310419, 310880, 310739, 309885, 310631])
N_0250 = np.array([620230, 620230, 620732, 618842, 621045, 621696, 622621, 622352, 620563, 622139])
N_0375 = np.array([930569, 930569, 931343, 928504, 931816, 932744, 934146, 933756, 931069, 933409])
N_0500 = np.array([1241368, 1241368, 1242401, 1238622, 1243032, 1244282, 1246178, 1245639, 1242040, 1245161])
N_0625 = np.array([1551967, 1551966, 1553248, 1548526, 1554052, 1555621, 1558003, 1557323, 1552798, 1556736])
N_0750 = np.array([1861361, 1861360, 1862910, 1857198, 1863881, 1865755, 1868614, 1867791, 1862353, 1867090])
N_0813 = np.array([2017810, 2017809, 2019499, 2013286, 2020552, 2022577, 2025685, 2024794, 2018896, 2024028])
N_0875 = np.array([2171610, 2171609, 2173429, 2166755, 2174562, 2176751, 2180077, 2179111, 2172778, 2178303])
N_0938 = np.array([2327967, 2327966, 2329924, 2322773, 2331128, 2333488, 2337070, 2336032, 2329227, 2335167])
N_1000 = np.array([2481782, 2481781, 2483871, 2476259, 2485164, 2487692, 2491514, 2490403, 2483132, 2489475])

s_qiso_0125 = np.array([0.017, 0.013, 0.020, 0.017, 0.014, 0.019, 0.016, 0.015, 0.017, 0.011])
s_qiso_0250 = np.array([0.0093, 0.0093, 0.011, 0.0097, 0.010, 0.0089, 0.0090, 0.0097, 0.013, 0.0082])
s_qiso_0375 = np.array([0.0086, 0.0075, 0.0092, 0.0076, 0.0074, 0.0076, 0.0078, 0.0073, 0.0090, 0.0076])
s_qiso_0500 = np.array([0.0083, 0.0068, 0.0073, 0.0062, 0.0064, 0.0068, 0.0068, 0.0067, 0.0075, 0.0069])
s_qiso_0625 = np.array([0.0075, 0.0062, 0.0072, 0.0054, 0.0057, 0.0063, 0.0063, 0.0060, 0.0070, 0.0062])
s_qiso_0750 = np.array([0.0071, 0.0062, 0.0066, 0.0054, 0.0056, 0.0059, 0.0057, 0.0054, 0.0065, 0.0058])
s_qiso_0813 = np.array([0.0066, 0.0061, 0.0063, 0.0052, 0.0055, 0.0059, 0.0053, 0.0052, 0.0065, 0.0057])
s_qiso_0875 = np.array([0.0066, 0.0059, 0.0062, 0.0049, 0.0055, 0.0058, 0.0053, 0.0051, 0.0061, 0.0055])
s_qiso_0938 = np.array([0.0063, 0.0057, 0.0059, 0.0050, 0.0054, 0.0058, 0.0053, 0.0050, 0.0058, 0.0054])
s_qiso_1000 = np.array([0.0061, 0.0054, 0.0058, 0.0048, 0.0052, 0.0057, 0.0052, 0.0049, 0.0061, 0.0052])


N_gals = [N_0125, N_0250, N_0375, N_0500, N_0625, N_0750, N_0813, N_0875, N_0938, N_1000]
s_qisos = [s_qiso_0125, s_qiso_0250, s_qiso_0375, s_qiso_0500, s_qiso_0625, s_qiso_0750, s_qiso_0813, s_qiso_0875, s_qiso_0938, s_qiso_1000]

N_gal = []
err_N_gal = []
s_qiso_sym = []
err_s_qiso_sym = []


for value in N_gals:
    N_gal.append(np.mean(value))
    err_N_gal.append(np.std(value, ddof=1) / np.sqrt(len(value))) 

for value in s_qisos:
    s_qiso_sym.append(np.mean(value))
    err_s_qiso_sym.append(np.std(value, ddof=1) / np.sqrt(len(value))) 
    
    

    

s_qiso_sym = np.array(s_qiso_sym)
err_s_qiso_sym = np.array(err_s_qiso_sym)

N_gal = np.array(N_gal)
err_N_gal = np.array(err_N_gal)


N_gal_sym = N_gal - 1.5e4
N_gal_iso = N_gal + 1.5e4




plot(N_gal_sym, s_qiso_sym, err_y=err_s_qiso_sym, err_x=err_N_gal, label='recsym', c='blue', cf='green')




# RecIso


s_qiso_0125 = np.array([0.019, 0.015, 0.023, 0.020, 0.016, 0.019, 0.018, 0.017, 0.020, 0.012])
s_qiso_0250 = np.array([0.010, 0.010, 0.012, 0.010, 0.011, 0.0093, 0.0096, 0.011, 0.014, 0.0087])
s_qiso_0375 = np.array([0.0090, 0.0077, 0.0097, 0.0079, 0.0078, 0.0079, 0.0081, 0.0076, 0.0093, 0.0078])
s_qiso_0500 = np.array([0.0087, 0.0070, 0.0077, 0.0063, 0.0067, 0.0071, 0.0070, 0.0069, 0.0076, 0.0070])
s_qiso_0625 = np.array([0.0077, 0.0067, 0.0075, 0.0056, 0.0060, 0.0066, 0.0065, 0.0062, 0.0071, 0.0063])
s_qiso_0750 = np.array([0.0073, 0.0063, 0.0068, 0.0056, 0.0059, 0.0061, 0.0058, 0.0056, 0.0068, 0.0059])
s_qiso_0813 = np.array([0.0070, 0.0062, 0.0065, 0.0054, 0.0057, 0.0060, 0.0056, 0.0054, 0.0066, 0.0058])
s_qiso_0875 = np.array([0.0068, 0.0060, 0.0064, 0.0050, 0.0057, 0.0059, 0.0054, 0.0053, 0.0064, 0.0056])
s_qiso_0938 = np.array([0.0064, 0.0058, 0.0061, 0.0051, 0.0056, 0.0059, 0.0054, 0.0051, 0.0061, 0.0055])
s_qiso_1000 = np.array([0.0062, 0.0056, 0.0060, 0.0049, 0.0054, 0.0058, 0.0052, 0.0050, 0.0061, 0.0052])



s_qisos = [s_qiso_0125, s_qiso_0250, s_qiso_0375, s_qiso_0500, s_qiso_0625, s_qiso_0750, s_qiso_0813, s_qiso_0875, s_qiso_0938, s_qiso_1000]

s_qiso_iso = []
err_s_qiso_iso = []


for value in s_qisos:
    s_qiso_iso.append(np.mean(value))
    err_s_qiso_iso.append(np.std(value, ddof=1) / np.sqrt(len(value))) 
    

s_qiso_iso = np.array(s_qiso_iso)
err_s_qiso_iso = np.array(err_s_qiso_iso)





plot(N_gal_iso, s_qiso_iso, err_y=err_s_qiso_iso, err_x=err_N_gal, label='reciso', c='orange', cf='purple')






# prerec

s_qiso_0125 = np.array([0.018, 0.022, 0.020, 0.023, 0.014, 0.024, 0.021, 0.019, 0.020, 0.014])
s_qiso_0250 = np.array([0.014, 0.013, 0.011, 0.015, 0.011, 0.011, 0.013, 0.012, 0.013, 0.010])
s_qiso_0375 = np.array([0.013, 0.011, 0.0100, 0.011, 0.0089, 0.011, 0.013, 0.010, 0.011, 0.010])
s_qiso_0500 = np.array([0.012, 0.011, 0.0069, 0.0079, 0.0079, 0.011, 0.011, 0.0093, 0.010, 0.0096])
s_qiso_0625 = np.array([0.011, 0.010, 0.0081, 0.0079, 0.0071, 0.010, 0.010, 0.0081, 0.0093, 0.0085])
s_qiso_0750 = np.array([0.0096, 0.0096, 0.0077, 0.0076, 0.0070, 0.0095, 0.0090, 0.0075, 0.0095, 0.0087])
s_qiso_0813 = np.array([0.0093, 0.0098, 0.0074, 0.0072, 0.0069, 0.0094, 0.0085, 0.0071, 0.0088, 0.0052])
s_qiso_0875 = np.array([0.0094, 0.0093, 0.0075, 0.0070, 0.0068, 0.0095, 0.0083, 0.0071, 0.0079, 0.0080])
s_qiso_0938 = np.array([0.0092, 0.0094, 0.0066, 0.0068, 0.0067, 0.0095, 0.0083, 0.0069, 0.0087, 0.0075])
s_qiso_1000 = np.array([0.0062, 0.0089, 0.0057, 0.0052, 0.0066, 0.0094, 0.0080, 0.0070, 0.0086, 0.0076])



s_qisos = [s_qiso_0125, s_qiso_0250, s_qiso_0375, s_qiso_0500, s_qiso_0625, s_qiso_0750, s_qiso_0813, s_qiso_0875, s_qiso_0938, s_qiso_1000]


s_qiso = []
err_s_qiso = []


for value in s_qisos:
    s_qiso.append(np.mean(value))
    err_s_qiso.append(np.std(value, ddof=1) / np.sqrt(len(value))) 




plot(N_gal, s_qiso, err_y=err_s_qiso, err_x=err_N_gal, label='prerec', c='lightgreen', cf='red')






ax.set_xlabel('$N_{gal}$')
ax.set_ylabel(r'$\sigma_{\alpha_{iso}}$')
ax.legend()
ax.grid()
fig.savefig(os.path.join(args.outdir, "sqiso_Ngal_final.png"), dpi=150)









        



print('Done')

