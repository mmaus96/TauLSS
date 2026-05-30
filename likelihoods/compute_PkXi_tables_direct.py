import numpy as np

from classy import Class
from compute_xiell_tables_recsym import compute_xiells, compute_bao_pkmu, kint,sphr
# from linear_theory import f_of_a
from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD
from velocileptors.EPT.ept_fullresum_varyDz_nu_fftw import REPT as REPT_nu
from velocileptors.Utils.pnw_dst import pnw_dst
from velocileptors.Utils.loginterp import loginterp
from scipy.interpolate import RectBivariateSpline
# from shapefit import shapefit_factor

# k vector to use:

from scipy.special import spherical_jn
from scipy.integrate import simpson as simps
from scipy.interpolate import interp1d
# from linear_theory import*
from pnw_dst import pnw_dst


speed_of_light = 2.99792458e5
zmin=0.001 
zmax=1.8
Nz=80
class direct_fit_theory():
    
    def __init__(self, zs, fid_dists=None):
        self.zs = zs
        
        self.fid_dists = {}
        for ii,z in enumerate(zs):
            zstr = "%.2f" %(z)
            self.fid_dists[zstr] = fid_dists[ii]
        self.qpar = {} #geometric AP parameters
        self.qperp = {}
        self.fz = {}
        self.Dz = {}
        self.fsig8 = {}
        # self.alpha_par = {}
        # self.alpha_perp = {}


    def compute_xiell_tables(self,z,ki,pi,Hz,chiz,qbao,f0,  R=15., rmin=50, rmax=160, dr=0.1, sigs = (0.,0.,0.)):
    
        sig_s,sig_par,sig_perp = sigs

        zstr = "%.2f" %(z)
        Hzfid, chizfid = self.fid_dists[zstr]

        speed_of_light = 2.99792458e5
    
        # Caluclate AP parameters and growth rate
        apar, aperp = Hzfid / Hz, chiz / chizfid

        self.qpar[zstr] = apar
        self.qperp[zstr] = aperp
    
    
        # Do the Zeldovich reconstruction predictions
    
        knw, pnw = pnw_dst(ki, pi)
        pw = pi - pnw
                
    
        j0 = spherical_jn(0,ki*qbao)
        Sk = np.exp(-0.5*(ki*R)**2)
    
        sigmadd = simps( 2./3 * pi * (1-Sk)**2 * (1-j0), x = ki) / (2*np.pi**2)
        sigmas = (sigmadd, sigs[0], sigs[1], sigs[2])
        
        # Now make the multipoles!
        klin, plin = ki, pi
        routs = np.arange(rmin, rmax, dr)
        
        # this is 1
        xi0_00,xi2_00  = compute_xiells(routs, 0, 0, klin, plin, pnw, f0, apar, aperp, R, sigmas)
    
        # this is 1 + B1 + B1^2
        # and 1 + 2 B1 + 4 B1^2
        xi0_10,xi2_10  = compute_xiells(routs,1, 0, klin, plin, pnw, f0, apar, aperp, R, sigmas)
        xi0_20,xi2_20  = compute_xiells(routs,2, 0, klin, plin, pnw, f0, apar, aperp, R, sigmas)
        
        # this is 1 + F + F^2
        # and 1 + 2 F + 4 F^2
        xi0_01,xi2_01  = compute_xiells(routs,0, 1, klin, plin, pnw, f0, apar, aperp, R, sigmas)
        xi0_02,xi2_02  = compute_xiells(routs,0, 2, klin, plin, pnw, f0, apar, aperp, R, sigmas)
        
        # and 1 + B1 + F + B1^2 + F^2 + BF
        xi0_11,xi2_11 = compute_xiells(routs,1, 1, klin, plin, pnw, f0, apar, aperp, R, sigmas)
        
        xi0table, xi2table = np.zeros( (len(routs),6) ), np.zeros( (len(routs),6) )
        
        # Form combinations:
        xi0_B1 = 0.5 * (4 * xi0_10 - xi0_20 - 3*xi0_00)
        xi0_B1sq = xi0_10 - xi0_B1 - xi0_00
    
        xi0_F = 0.5 * (4 * xi0_01 - xi0_02 - 3*xi0_00)
        xi0_Fsq = xi0_01 - xi0_F - xi0_00
    
        xi0_BF = xi0_11 - xi0_B1 - xi0_F - xi0_B1sq - xi0_Fsq - xi0_00
        
        xi2_B1 = 0.5 * (4 * xi2_10 - xi2_20 - 3*xi2_00)
        xi2_B1sq = xi2_10 - xi2_B1 - xi2_00
    
        xi2_F = 0.5 * (4 * xi2_01 - xi2_02 - 3*xi2_00)
        xi2_Fsq = xi2_01 - xi2_F - xi2_00
    
        xi2_BF = xi2_11 - xi2_B1 - xi2_F - xi2_B1sq - xi2_Fsq - xi2_00
        
        # Load
        xi0table[:,0] = xi0_00
        
        xi0table[:,1] = xi0_B1
        xi0table[:,2] = xi0_F
    
        xi0table[:,3] = xi0_B1sq
        xi0table[:,4]= xi0_Fsq
    
        xi0table[:,5] = xi0_BF
        
        xi2table[:,0] = xi2_00
        
        xi2table[:,1] = xi2_B1
        xi2table[:,2] = xi2_F
    
        xi2table[:,3] = xi2_B1sq
        xi2table[:,4]= xi2_Fsq
    
        xi2table[:,5] = xi2_BF
        
        
        return routs,xi0table, xi2table

    def setup_REPT(self,ki,pi_0,qBAO):
        # Calculate and renormalize power spectrum
        _,pnw_0 = pnw_dst(ki,pi_0)

        kmin = 5e-3
        kmax = 0.5
        nk = 100

        self.kvec = np.logspace(np.log10(kmin), np.log10(kmax), nk)

        self.modEPT = REPT_nu(ki, pi_0, pnw=pnw_0, kmin = kmin, kmax = kmax, nk = nk,rbao =qBAO,\
              cutoff = 10,extrap_min = -4, extrap_max = 3,N=2000)  

    def compute_pell_tables_EPT(self,z,ki,pi_z,Hz,chiz,fz,Dz):
        zstr = "%.2f" %(z)
        Hzfid, chizfid = self.fid_dists[zstr]
        kvec = self.kvec

        # Caluclate AP parameters and growth rate
        apar, aperp = Hzfid / Hz, chiz / chizfid
        self.qpar[zstr] = apar
        self.qperp[zstr] = aperp
        self.fz[zstr] = fz
        self.Dz[zstr] = Dz
        
        # Calculate and renormalize power spectrum
        _,pnw_z = pnw_dst(ki,pi_z)
        pi_z  = loginterp(ki, pi_z)(kvec)
        pnw_z = loginterp(ki, pnw_z)(kvec)
        kv, P0tab,P2tab, P4tab = self.modEPT.compute_redshift_space_power_multipoles_tables(fz, Dz=Dz,pcb=pi_z, pcb_nw=pnw_z, ngauss=4, apar=apar, aperp=aperp)
        return kv, P0tab,P2tab, P4tab


        
        
        
   