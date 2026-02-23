import numpy as np

# from classy import Class
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
    
    def __init__(self, zs, pars = None):
        if pars != None:
            w, omega_b,omega_cdm, h, logA, ns = pars
        else:
            w, omega_b,omega_cdm, h, logA, ns = [-1., 0.02237, 0.12, 0.6736, np.log(1e10 * 2.0830e-9), 0.9649]

        nnu = 1
        nur = 2.0328
        # mnu = 0.06
        omega_nu = 0.0006442 #0.0106 * mnu
        # mnu = omega_nu / 0.0106
        
        As =  np.exp(logA)*1e-10
        w0 = w
        wa = 0.

        # omega_c = (OmegaM - omega_b/h**2 - omega_nu/h**2) * h**2

        pkparams = {
            'output': 'mPk',
            'P_k_max_h/Mpc': 20.,
            'z_pk': '0.0,10',
            'A_s': As,
            'n_s': ns,
            'h': h,
            'N_ur': nur,
            'N_ncdm': nnu,
            'omega_ncdm': omega_nu,
            # 'm_ncdm': mnu,
            'tau_reio': 0.0568,
            'omega_b': omega_b,
            'omega_cdm': omega_cdm,
            'Omega_Lambda': 0.,
            'w0_fld': w0,
            'wa_fld': wa}

        fid_class = Class()
        fid_class.set(pkparams)
        fid_class.compute()
        
        self.theta_star = fid_class.theta_star_100()
        
        self.fid_class = fid_class
        
        self.kvec = np.concatenate( ([0.0005,],\
                        np.logspace(np.log10(0.0015),np.log10(0.025),10, endpoint=True),\
                        np.arange(0.03,0.51,0.01)) )
        self.ki = np.logspace(-3.0,1.0,200)

        self.zs = zs
        
        self.fid_dists = {}

        for z in zs:
            zstr = "%.2f" %(z)
            self.fid_dists[zstr] = self.get_fid_dists(z)

        self.zint = np.linspace(zmin,zmax,Nz)  #For Pmm in Ckg

        self.rdfid = fid_class.rs_drag()*h
        self.qpar = {} #geometric AP parameters
        self.qperp = {}
        self.fz = {}
        self.Dz = {}
        self.fsig8 = {}
        self.alpha_par = {}
        self.alpha_perp = {}
        
    def get_fid_dists(self,z):
        
        speed_of_light = 2.99792458e5
        
        fid_class = self.fid_class
        h = fid_class.h()
        
        Hz_fid = fid_class.Hubble(z) * speed_of_light / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
        chiz_fid = fid_class.angular_distance(z) * (1.+z) * h # this is the comoving radius in units of Mpc/h 
        fid_dists = (Hz_fid, chiz_fid)
        
        return fid_dists

    def compute_pkclass(self,pars):
    
        w0,wa,Omega_m, logA = pars
        # speed_of_light = 2.99792458e5
        self.Npars = len(pars)
    
        omega_b = 0.02237

        theta_star = 1.0411
    
        As =  np.exp(logA)*1e-10 #2.0830e-9
        ns = 0.9649
    
        nnu = 1
        nur = 2.0328
        # mnu = 0.06
        omega_nu = 0.0006442 #0.0106 * mnu
        # mnu = omega_nu / 0.0106
            
        # omega_c = (OmegaM - omega_b/h**2 - omega_nu/h**2) * h**2
        # OmegaM = (omega_cdm + omega_b + omega_nu) / h**2
    
        # w0,wa = w0wa_from_wpdH(wp,dH,h*100,OmegaM)
        # print('w0,wa=', (w0,wa) )
        pkparams = {
            'output': 'mPk',
            'P_k_max_h/Mpc': 20.,
            'z_pk': '0.0,10',
            'A_s': As,
            'n_s': ns,
            # 'h': h,
            '100*theta_s': theta_star,
            'non linear': 'halofit',
            'N_ur': nur,
            'N_ncdm': nnu,
            'omega_ncdm': omega_nu,
            # 'm_ncdm': mnu,
            # 'tau_reio': 0.0568,
            'z_reio': 7.,
            'omega_b': omega_b,
            'Omega_m': Omega_m,
            'Omega_Lambda': 0.,
            'w0_fld': w0,
            'wa_fld': wa}
    
        pkclass = Class()
        pkclass.set(pkparams)
        pkclass.compute()

        self.pkclass = pkclass
        self.h = pkclass.h()
        self.sigma8_0 = pkclass.sigma8()
        self.rd = pkclass.rs_drag()*self.h
        
        
        return pkclass,self.h,self.sigma8_0


    def compute_pkclass_nopert(self,pars):
    
        w0,wa,Omega_m, logA = pars
        # speed_of_light = 2.99792458e5
        self.Npars = len(pars)
    
        omega_b = 0.02237

        theta_star = 1.0411
    
        As =  np.exp(logA)*1e-10 #2.0830e-9
        ns = 0.9649
    
        nnu = 1
        nur = 2.0328
        # mnu = 0.06
        omega_nu = 0.0006442 #0.0106 * mnu
        # mnu = omega_nu / 0.0106
            
        # omega_c = (OmegaM - omega_b/h**2 - omega_nu/h**2) * h**2
        # OmegaM = (omega_cdm + omega_b + omega_nu) / h**2
    
        # w0,wa = w0wa_from_wpdH(wp,dH,h*100,OmegaM)
        # print('w0,wa=', (w0,wa) )
        pkparams = {
            'output': '',
            'A_s': As,
            'n_s': ns,
            # 'h': h,
            '100*theta_s': theta_star,
            'N_ur': nur,
            'N_ncdm': nnu,
            'omega_ncdm': omega_nu,
            # 'm_ncdm': mnu,
            # 'tau_reio': 0.0568,
            'z_reio': 7.,
            'omega_b': omega_b,
            'Omega_m': Omega_m,
            'Omega_Lambda': 0.,
            'w0_fld': w0,
            'wa_fld': wa}
    
        pkclass = Class()
        pkclass.set(pkparams)
        pkclass.compute()

        self.pkclass_alt = pkclass
        self.h = pkclass.h()
        self.rd = pkclass.rs_drag()*self.h


    def compute_xiell_tables(self,z,ki,pi,rs_drag,h,Hz,chiz,  R=15., rmin=50, rmax=160, dr=0.5, sigs = (0.,0.,0.)):
    
        # w0,wa,Omega_m, theta_star, logA = pars
        sig_s,sig_par,sig_perp = sigs

        zstr = "%.2f" %(z)
        Hzfid, chizfid = self.fid_dists[zstr]

        h = self.h
        # pkclass = self.pkclass
        
        speed_of_light = 2.99792458e5
    
        # Caluclate AP parameters and growth rate
        try:
            apar,aperp = self.qpar[zstr],self.qperp[zstr]
        except:
            Hz = pkclass.Hubble(z) * speed_of_light / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
            chiz = pkclass.angular_distance(z) * (1.+z) * h # this is the comoving radius in units of Mpc/h 
            apar, aperp = Hzfid / Hz, chiz / chizfid
    
            self.qpar[zstr],self.qperp[zstr] = apar, aperp

        f0 = self.fz[zstr]
        Dz = self.Dz[zstr]
        
    
    
        # Do the Zeldovich reconstruction predictions
    
        knw, pnw = pnw_dst(ki, pi)
        pw = pi - pnw
                
        try:
            qbao = self.rd
        except:
            qbao = rs_drag*h  # want this in Mpc/h units
            self.rd = qbao
    
        j0 = spherical_jn(0,ki*qbao)
        Sk = np.exp(-0.5*(ki*R)**2)
    
        sigmads_dd = simps( 2./3 * pi * (1-Sk)**2, x = ki) / (2*np.pi**2)
        # sigmads_ss = simps( 2./3 * pi * (-Sk)**2, x = ki) / (2*np.pi**2)
        # sigmads_ds = -simps( 2./3 * pi * (1-Sk)*(-Sk)*j0, x = ki) / (2*np.pi**2) # this minus sign is because we subtract the cross term
    
        sigmas = (sigmads_dd, sigs[0], sigs[1], sigs[2])
        
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

    def compute_pell_tables_LPT(self,z):

        zstr = "%.2f" %(z)
        Hzfid, chizfid = self.fid_dists[zstr]

        h = self.h
        pkclass = self.pkclass
        
        speed_of_light = 2.99792458e5
    
        # Caluclate AP parameters and growth rate
        try:
            apar,aperp = self.qpar[zstr],self.qperp[zstr]
            f=self.fz[zstr]
        except:
            Hz = pkclass.Hubble(z) * speed_of_light / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
            chiz = pkclass.angular_distance(z) * (1.+z) * h # this is the comoving radius in units of Mpc/h 
            apar, aperp = Hzfid / Hz, chiz / chizfid
    
            self.qpar[zstr],self.qperp[zstr] = apar, aperp

            f = pkclass.scale_independent_growth_factor_f(z)
            self.fz[zstr] = f
    
        # Calculate and renormalize power spectrum
        ki = self.ki
        pi = np.array( [pkclass.pk_cb_lin(k*h, z ) * h**3 for k in ki] )

        # Now do the RSD
        self.modPT = LPT_RSD(ki, pi, kIR=0.2,use_Pzel = False,\
                    cutoff=10, extrap_min = -4, extrap_max = 3, N = 2000, threads=1, jn=5)
        self.modPT.make_pltable(f, kv=self.kvec, apar=apar, aperp=aperp, ngauss=3)

        return self.kvec,self.modPT.p0ktable, self.modPT.p2ktable, self.modPT.p4ktable

    def setup_REPT(self,ki,pi_0,rs_drag, h):
        # pkclass = self.pkclass
        # h = self.h
        try:
            rBAO = self.rd
        except:
            rBAO = rs_drag*h
            self.rd = rBAO

        # Calculate and renormalize power spectrum
        ki = self.ki
        pi_0 = np.array( [pkclass.pk_cb_lin(k*h, 0 ) * h**3 for k in ki] )
        _,pnw_0 = pnw_dst(ki,pi_0)

        kmin = 5e-3
        kmax = 0.5
        nk = 100

        self.kvec = np.logspace(np.log10(kmin), np.log10(kmax), nk)

        self.modEPT = REPT_nu(ki, pi_0, pnw=pnw_0, kmin = kmin, kmax = kmax, nk = nk,rbao =rBAO,\
              cutoff = 10,extrap_min = -4, extrap_max = 3,N=2000)  

    def compute_pell_tables_EPT(self,z,ki,pi_z,h,Hz,chiz):
        # pkclass = self.pkclass
        # h = self.h
        zstr = "%.2f" %(z)
        Hzfid, chizfid = self.fid_dists[zstr]
        kvec = self.kvec

        # Caluclate AP parameters
        # Caluclate AP parameters and growth rate
        
        # Caluclate AP parameters and growth rate
        try:
            apar,aperp = self.qpar[zstr],self.qperp[zstr]
        except:
            apar, aperp = Hzfid / Hz, chiz / chizfid
            self.qpar[zstr],self.qperp[zstr] = apar, aperp
        fz = self.fz[zstr]
        Dz = self.Dz[zstr]    

        # Calculate and renormalize power spectrum
        _,pnw_z = pnw_dst(ki,pi_z)
        pi_z  = loginterp(ki, pi_z)(kvec)
        pnw_z = loginterp(ki, pnw_z)(kvec)
        kv, P0tab,P2tab, P4tab = self.modEPT.compute_redshift_space_power_multipoles_tables(fz, Dz=Dz,pcb=pi_z, pcb_nw=pnw_z, ngauss=4, apar=apar, aperp=aperp)
        return kv, P0tab,P2tab, P4tab

    def compute_Pmm(self,args,z):
        pkclass = self.pkclass
        h = self.h
        ki = self.ki
        zevals = np.linspace(0.2,2.0,10)
        Pmm_grid = np.zeros((len(ki),10))
        for ii,zi in enumerate(zevals):
            pnl = np.array( [pkclass.pk(k*h, zi ) * h**3 for k in ki] )
            Pmm_grid[:,ii] = pnl
        zint = self.zint

        # Create the interpolator (assuming P_kz is ordered as P(k, z))
        interpolator = RectBivariateSpline(ki, zevals, Pmm_grid)
        Pmm_interp = interpolator(ki,zint)
        res = [Pmm_interp[:,i] for i in range(len(zint))]
        return np.array([ki]+res).T

    def compute_Pgm(self,pars,z):
        pkclass = self.pkclass
        h = self.h
        ki = self.ki
        zstr = "%.2f" %(z)
        try:
            f=self.fz[zstr]
        except:
            f = pkclass.scale_independent_growth_factor_f(z)
            self.fz[zstr] = f
        
        b1,b2,bs = pars[self.Npars:]
        b3 = 0

        p_cb = np.array( [pkclass.pk_cb_lin(k*h, z ) * h**3 for k in ki] )
        p_mm = np.array( [pkclass.pk_lin(k*h, z ) * h**3 for k in ki] )
        
        p_gm = np.sqrt(p_cb * p_mm)

        modPT_gm = LPT_RSD(ki, p_gm, kIR=0.2, use_Pzel = True ,\
                cutoff=10, extrap_min = -4, extrap_max = 3, N = 2000, threads=1, jn=5)
        modPT_gm.make_ptable(f, 0, kv=self.kvec)
        ptab = modPT_gm.pktables[0]

        kout,za = ptab[:,0],ptab[:,-1]
        res      = np.zeros((len(kout),3))
        res[:,0] = kout
        bias_monomials = np.array([1., 0.5*b1, 0,\
                                  0.5*b2, 0, 0,\
                                  0.5*bs, 0, 0, 0,\
                                  0.5*b3, 0.])
        res[:,1] = np.sum(ptab[:,1:-1]*bias_monomials,axis=1) 
        res[:,2] = -0.5*kout**2 * za

        return res

    def compute_Pgg(self,pars,z):
        pkclass = self.pkclass
        h = self.h
        ki = self.ki
        zstr = "%.2f" %(z)
        try:
            f=self.fz[zstr]
        except:
            f = pkclass.scale_independent_growth_factor_f(z)
            self.fz[zstr] = f
        
        b1,b2,bs = pars[self.Npars:]
        b3 = 0

        p_cb = np.array( [pkclass.pk_cb_lin(k*h, z ) * h**3 for k in ki] )

        modPT_gg = LPT_RSD(ki, p_cb, kIR=0.2, use_Pzel = True ,\
                cutoff=10, extrap_min = -4, extrap_max = 3, N = 2000, threads=1, jn=5)
        modPT_gg.make_ptable(f, 0, kv=self.kvec)
        ptab = modPT_gg.pktables[0]

        kout,za = ptab[:,0],ptab[:,-1]
        res      = np.zeros((len(kout),3))
        res[:,0] = kout
        bias_monomials = np.array([1, b1, b1**2,\
                                       b2, b1*b2, b2**2,\
                                       bs, b1*bs, b2*bs, bs**2,\
                                       b3, b1*b3 ])

        res[:,1] = np.sum(ptab[:,1:-1]*bias_monomials,axis=1) 
        res[:,2] = -0.5*kout**2 * za

        return res

    def compute_derived(self,z):
        pkclass = self.pkclass
        h = self.h
        zstr = "%.2f" %(z)
        Hzfid, chizfid = self.fid_dists[zstr]

        rd = self.rd
        rdfid = self.rdfid

        try:
            qpar,qperp = self.qpar[zstr],self.qperp[zstr]
        except:
            Hz = pkclass.Hubble(z) * speed_of_light / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
            chiz = pkclass.angular_distance(z) * (1.+z) * h # this is the comoving radius in units of Mpc/h 
            qpar, qperp = Hzfid / Hz, chiz / chizfid
            self.qpar[zstr],self.qperp[zstr] = qpar, qperp
        try:
            fz = self.fz[zstr]
        except:
            fz = pkclass.scale_independent_growth_factor_f(z)
        try:
            Dz = self.Dz[zstr]
        except:
            Dz = pkclass.scale_independent_growth_factor(z)

        sig80 = self.sigma8_0

        self.fsig8[zstr] = fz*sig80*Dz
        self.alpha_par[zstr] = qpar*rdfid / rd
        self.alpha_perp[zstr] = qperp*rdfid / rd
        

        return Dz,fz

    def compute_derived_nopert(self,z):
        pkclass = self.pkclass_alt
        h = self.h
        zstr = "%.2f" %(z)
        Hzfid, chizfid = self.fid_dists[zstr]

        rd = self.rd
        rdfid = self.rdfid

        
        Hz = pkclass.Hubble(z) * speed_of_light / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
        chiz = pkclass.angular_distance(z) * (1.+z) * h # this is the comoving radius in units of Mpc/h 
        qpar, qperp = Hzfid / Hz, chiz / chizfid
        self.qpar[zstr],self.qperp[zstr] = qpar, qperp
        
        
        self.alpha_par[zstr] = qpar*rdfid / rd
        self.alpha_perp[zstr] = qperp*rdfid / rd
        

        return self.alpha_par[zstr],self.alpha_perp[zstr]

        
        
        
   