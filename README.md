# TauLSS

This repo is for running cosmology fits to CMB primary + lensing and DESI fullshape + BAO data with varying w0wa, reionization optical depth, neutrino mass, and more. The theory codes do not make use of any emulators; however most of the computation time comes from Cobaya's classy module.

## Setup

1. Create the cobaya environment with `create_cobaya_env.sh`
2. Use `get_act_data.sh` to get the ACT DR6 lensing likelihood
3. Get the DESI DR1 pre-recon power spectra, post-recon correlation functions, and associated windows and covariances with `get_LSS_data.sh`. This will create a `./data/` directory. 

## organization

- The `./likelihoods` directory contains backend likelihood and theory codes for EFT and BAO models
- The yamls in `./configs`, e.g. `test_RSDBAO_jax_w0wa_tau_h.yaml` are configuration files for a Cobaya fit. These will grab info from specified theory, likelihood and parameter files 
- `./configs/likelihoods_theories` contains seperate yamls for the likelihood and theory blocks and `./configs/params` contains files with parameter info
- Chains are stored in `./chains`

## Running chains
- submit jobs with `Joint_fit_reg.sh`
- see `test_RSDBAO_jax_w0wa_tau_h.yaml` for an example run with DESI fullshape and BAO.
- - To use additional CMB likelihoods one can include `./likelihoods_theories/act_dr6_lenslike`, `./likelihoods_theories/planck_2018_lowl.EE`, `./likelihoods_theories/planck_2018_lowl.TT`, `./likelihoods_theories/planck_NPIPE_highl_CamSpec.TTTEEE` in the likelihood block
- You can choose between sampling in H0 vs sampling in theta_star by using `cosmo_base_h.yaml` vs `cosmo_base_thetastar.yaml` in params for the base LCDM parameters (both of these files will also set priors for ns, logA, omega_b, and omega_cdm and set derived parameters like sigma8)
- Currently there is a Jax impleimented post-recon BAO and a non-jax version. At some point we will also have a jax velocileptors LPT and EPT module so then using jax makes more of a difference. Refer to `test_RSDBAO_nojax_w0wa_tau_h..yaml` if not using jax.




