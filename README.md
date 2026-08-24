# BAO_pipeline_BA2026
This repository contains codes based on cosmodesi I have used in my bachelor project regarding the BAO pipeline. They are free to use for future work.

The codes are designed to use LRG mocks published by DESI as input files, calculate the corresponding power spectrum (either pre- or post-reconstruction) and carry out the BAO fit. The main purpose of this publication however is to give a working example of how the cosmodesi code can be used in practice, hoping that it can serve as a base for future projects.

The following codes are provided:
- full_BAO_pipeline.py contains the full BAO pipeline - including reconstruction, the power spectrum measurement and the BAO fitting - to be applied onto an input mock file.
- prerec_BAO_pipeline.py is a simplified version of full_BAO_pipeline.py, skipping the reconstruction process and instead calculating the power spectrum directly from the input catalogs. It can therefore be used to carry out pre reconstruction BAO fits.
- BAO_fitting.py only keeps the part of full_BAO_pipeline.py responsible to carry out the BAO fit and therefore requires a pypower power spectrum object that can be created by the previous codes as an input file. It is meant to save computing time and resources demanded by reconstruction and the power spectrum measurement if different BAO fits are to be made for the same data. The uploaded version of this code still requires the mock this data is obtained from as an input file to calculate some of the physical parameters necessary for the fit, but if these are well known this part of the code can be easily removed.
- ugugu is a simple example of a code also carrying out the BAO fit, but this time using an input file already containing the power spectrum and the corresponding covariance matrix as numpy arrays.
- sqiso_Ngal.py is a code used to make a prediction of the dependency of the error of the isotropic dilation parameter (the main BAO fitting parameter from the codes above) on the number of galaxies in the catalogs. The calculations are based on https://arxiv.org/pdf/2002.04035 and https://arxiv.org/pdf/astro-ph/9706198 and further explained in my bachelor project.
