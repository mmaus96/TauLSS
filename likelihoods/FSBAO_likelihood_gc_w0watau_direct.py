import numpy as np
import time
import json
import yaml

from cobaya.theory     import Theory
from cobaya.likelihood import Likelihood
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.signal import argrelmax
from scipy.interpolate import interp1d
from numpy.polynomial.polynomial import polyval

from scipy.special import spherical_jn
# from scipy.integrate import simps

# from taylor_approximation import taylor_approximate
from compute_PkXi_tables_direct import* 
# from linear_theory import f_of_a, D_of_a
# from compute_sigma8_class import Compute_Sigma8_wCDM
# from compute_pell_tables_direct import direct_fit_theory
# from compute_xiell_tables_recsym import compute_xiell_tables, compute_pkclass
# from Compute_zParams_class import Compute_zParams
# from aemulus_heft.heft_emu import NNHEFTEmulator

import scipy.integrate as integrate
from numpy import sqrt, sin, cos, pi



class JointLikelihood(Likelihood):
    
    zfids: list
    photo_tmp: bool
    
    basedir: str
    
    fs_sample_names: list
    bao_sample_names: list
    
    # optimize: turn this on when optimizng/running the minimizer so that the Jacobian factor isn't included
    # include_priors: this decides whether the marginalized parameter priors are included (should be yes)
    linear_param_dict_fn: str
    optimize: bool
    include_priors: bool
    
    fs_datfns: list
    bao_datfns: list

    covfn: str
    template_fn: str
    
    fs_kmins: list
    fs_mmaxs: list
    fs_qmaxs: list
    fs_hmaxs: list
    # fs_matMfns: list
    fs_matWfns: list
    w_kin_fn: str
    hexa: bool
    bao_rmaxs: list
    bao_rmins: list
    bao_ells: list
    
    # npoly: int
    cov_fac: float
    invcov_fac: float
    jeff: bool
    AP_scale: bool
    bsig8: float

    def initialize(self):
        """Sets up the class."""
        
        # Redshift Label for theory classes
        # self.zstr = "%.2f" %(self.zfid)

        # Load the linear parameters of the theory model theta_a such that
        # P_th = P_{th,nl} + theta_a P^a for some templates P^a we will compute
        self.linear_param_dict = yaml.load(open(self.linear_param_dict_fn), Loader=yaml.SafeLoader)
        self.linear_param_means = {key: self.linear_param_dict[key]['mean'] for key in self.linear_param_dict.keys()}
        self.linear_param_stds  = np.array([self.linear_param_dict[key]['std'] for key in self.linear_param_dict.keys()])
        self.Nlin = len(self.linear_param_dict) 
        
        # broadband parameter names:
        self.mpoly_names = [ 'M%d'%(ii) for ii in range(2)]
        self.qpoly_names = [ 'Q%d'%(ii) for ii in range(2)]
        self.qspl_names = [ 'Q%dsp'%(ii) for ii in range(2)]
        
        # Places to store theory curves
        self.pconv = {}
        self.xith = {}
        
#         self.sp_kmax = {}
        
#         for ll, fs_sample_name in enumerate(self.fs_sample_names):
#             self.sp_kmax[fs_sample_name] = self.kmax_spline[ll]
        
        # Binning matrix for correlation function, one for each BAO sample
        self.binmat = dict((name, None) for name in self.bao_sample_names)
        
        # rmin, rmax, dr = 50, 160, 0.1
        # rvec = np.arange(rmin, rmax, dr)
        
        
        
        self.loadData()
        
        self.delta = 0.06
        self.B20 = {}
        self.B21 = {}
        for bao_sample_name in self.bao_sample_names:
            rvec = self.rdats[bao_sample_name]
            self.B20[bao_sample_name] = np.array([self.B20_x(self.delta*r) for r in rvec])
            self.B21[bao_sample_name] = np.array([self.B21_x(self.delta*r) for r in rvec])
        
        
        #
    
    def Si_x(self,x):
        result = integrate.quad(lambda t: sin(t)/t, 0, x)
        return result[0]

    def B20_x(self,x):
        prefac = 2./x**6
        terms = x**3 * self.Si_x(x) - 2*x**3*self.Si_x(2*x)
        terms += x**2 * cos(x) - x**2*cos(2*x) - x*sin(x) - 16*cos(x)
        terms += 4*cos(2*x)+x*sin(x)*cos(x)+12
        return prefac*terms

    def B21_x(self,x):
        prefac = -1./(2*x**3)
        terms = 6*x**3*self.Si_x(x) - 32*x**3*self.Si_x(2*x) + 27*x**3*self.Si_x(3*x)
        terms += 8*x**2 + 6*x**2*cos(x) -16*x**2*cos(2*x) + 9*x**2*cos(3*x)
        terms += -6*x*sin(x) + 8*x*sin(2*x) -3*x*sin(3*x)
        terms += -96*cos(x) + 64*cos(2*x) - 16*cos(3*x) + 48
        return prefac*terms
    
    def get_requirements(self):
        
        # Here we will comment out all the linear parameters
        
        req = {'taylor_pk_ell_mod': None,\
               'taylor_xi_ell_mod': None,\
               'zPars': None,\
               'ns': None,\
               'H0': None,\
            #    'sig8': None,\
               'sigma8': None,\
               # 'omega_b': None,\
               # 'omega_cdm': None,\
               'omegam': None,\
               'w': None,\
               'wa': None,\
                'logA': None}
        
        for fs_sample_name in self.fs_sample_names:
            req_bias = { \
                   'bsig8_' + fs_sample_name: None,\
                   # 'b1_' + fs_sample_name: None,\
                   'b2sig8_' + fs_sample_name: None,\
                   'bssig8_' + fs_sample_name: None,\
                   'b3sig8_' + fs_sample_name: None,\
                   #'alpha0_' + fs_sample_name: None,\
                   #'alpha2_' + fs_sample_name: None,\
                   #'SN0_' + fs_sample_name: None,\
                   #'SN2_' + fs_sample_name: None\
                   }
            req = {**req, **req_bias}
        
        for bao_sample_name in self.bao_sample_names:
            req_bao = {\
                   'B1_' + bao_sample_name: None,\
                   'F_' +  bao_sample_name: None,\
                   # 'Sigpar_' +  bao_sample_name: None,\
                   # 'Sigperp_' +  bao_sample_name: None,\
                   # 'Sigs_' +  bao_sample_name: None,\
                   #'M0_' + bao_sample_name: None,\
                   #'M1_' + bao_sample_name: None,\
                   #'M2_' + bao_sample_name: None,\
                   #'Q0_' + bao_sample_name: None,\
                   #'Q1_' + bao_sample_name: None,\
                   #'Q2_' + bao_sample_name: None,\
                    }
            req = {**req, **req_bao}
            
        return(req)
    
    def full_predict(self, thetas=None):
        
        thy_obs = []

        if thetas is None:
            thetas = self.linear_param_means
        
        for zfid,fs_sample_name in zip(self.zfids,self.fs_sample_names):
            fs_thy  = self.fs_predict(fs_sample_name,zfid,thetas=thetas)
            fs_obs  = self.fs_observe(fs_thy, fs_sample_name,thetas=thetas)
            thy_obs = np.concatenate( (thy_obs,fs_obs) )
            
            if fs_sample_name in self.bao_sample_names:
                bao_thy = self.bao_predict(fs_sample_name,zfid)
                bao_obs = self.bao_observe(bao_thy,fs_sample_name,thetas=thetas)
                thy_obs = np.concatenate( (thy_obs, bao_obs) )
        
        # for bao_sample_name in self.bao_sample_names:
        #     bao_thy = self.bao_predict(bao_sample_name,thetas=thetas)
        #     bao_obs = self.bao_observe(bao_thy,bao_sample_name)
        #     thy_obs = np.concatenate( (thy_obs, bao_obs) )
            
        return thy_obs
    
    def logp(self,**params_values):
        """Return a log-likelihood."""
        
        # Compute the theory prediction with lin. params. at prior mean
        #t1 = time.time()
        thy_obs_0 = self.full_predict()
        self.Delta = self.dd - thy_obs_0
        #t2 = time.time()
        
        # Now compute template
        self.templates = []
        for param in self.linear_param_dict.keys():
            thetas = self.linear_param_means.copy()
            thetas[param] += 1.0
            self.templates += [ self.full_predict(thetas=thetas) - thy_obs_0 ]
        
        self.templates = np.array(self.templates)
        #t3 = time.time()
        
        # Make dot products
        self.Va = np.dot(np.dot(self.templates, self.cinv), self.Delta)
        self.Lab = np.dot(np.dot(self.templates, self.cinv), self.templates.T) + self.include_priors * np.diag(1./self.linear_param_stds**2)
        #self.Va = np.einsum('ij,jk,k', self.templates, self.cinv, self.Delta)
        #self.Lab = np.einsum('ij,jk,lk', self.templates, self.cinv, self.templates) + np.diag(1./self.linear_param_stds**2)
        self.Lab_inv = np.linalg.inv(self.Lab)
        #t4 = time.time()
        
        # Compute the modified chi2
        lnL  = -0.5 * np.dot(self.Delta,np.dot(self.cinv,self.Delta)) # this is the "bare" lnL
        lnL +=  0.5 * np.dot(self.Va, np.dot(self.Lab_inv, self.Va)) # improvement in chi2 due to changing linear params
        self.detFish = 0.5 * np.log( np.linalg.det(self.Lab) )
        if not self.optimize:
            if self.jeff:
                lnL += 0.5 * self.Nlin * np.log(2*np.pi)
            else:
                lnL += - 0.5 * np.log( np.linalg.det(self.Lab) ) + 0.5 * self.Nlin * np.log(2*np.pi) # volume factor from the determinant
        
        #t5 = time.time()
        
        #print(t2-t1, t3-t2, t4-t3, t5-t4)
        
        return lnL
        #
        
    def get_best_fit(self):
        try:
            self.p0_nl  = self.dd - self.Delta
            self.bf_thetas = np.einsum('ij,j', np.linalg.inv(self.Lab), self.Va)
            self.p0_lin = np.einsum('i,il', self.bf_thetas, self.templates)
            return self.p0_nl + self.p0_lin
        except:
            print("Make sure to first compute the posterior.")
        
    def loadData(self):
        """
        Loads the required data.
        
        Do this in two steps... first load full shape data then xirecon, concatenate after.
        
        The covariance is assumed to already be joint in the concatenated format.
        
        """
        # First load the data
        
        self.kdats = {}
        self.p0dats = {}
        self.p2dats = {}
        self.p4dats = {}
        self.fitiis = {}
        
        for ii, fs_datfn in enumerate(self.fs_datfns):
            fs_sample_name = self.fs_sample_names[ii]
            fs_dat = np.loadtxt(self.basedir+fs_datfn)
            self.kdats[fs_sample_name] = fs_dat[:,0]
            self.p0dats[fs_sample_name] = fs_dat[:,1]
            self.p2dats[fs_sample_name] = fs_dat[:,2]
            try:
                self.p4dats[fs_sample_name] = fs_dat[:,3]
                hex_dat = True
            except:
                print('No hexadecapole data found')
                hex_dat = False
            
            # Make a list of indices for the monopole and quadrupole only in Fourier space
            # This is specified to each sample in case the k's are different.
            yeses = self.kdats[fs_sample_name] > 0
            nos   = self.kdats[fs_sample_name] < 0
            if self.hexa:
                self.fitiis[fs_sample_name] = np.concatenate( (yeses, yeses, yeses ) )
            else:
                self.fitiis[fs_sample_name] = np.concatenate( (yeses,yeses, nos ) )
        
        self.rdats = {}
        self.xi0dats = {}
        self.xi2dats = {}
        
        for ii, bao_datfn in enumerate(self.bao_datfns):
            bao_sample_name = self.bao_sample_names[ii]
            bao_dat = np.loadtxt(self.basedir+bao_datfn)
            self.rdats[bao_sample_name] = bao_dat[:,0]
            self.xi0dats[bao_sample_name] = bao_dat[:,1]
            if len(self.bao_ells) ==2:
                self.xi2dats[bao_sample_name] = bao_dat[:,2]
        
        # Join the data vectors together
        self.dd = []
        
        for fs_sample_name in self.fs_sample_names:
            if self.hexa:
                self.dd = np.concatenate( (self.dd, self.p0dats[fs_sample_name], self.p2dats[fs_sample_name], self.p4dats[fs_sample_name]) )
            else:
                self.dd = np.concatenate( (self.dd, self.p0dats[fs_sample_name], self.p2dats[fs_sample_name]) )
            
            
            if fs_sample_name in self.bao_sample_names:
                self.dd = np.concatenate( (self.dd, self.xi0dats[fs_sample_name]) )
                if len(self.bao_ells) >1:
                    self.dd = np.concatenate( (self.dd, self.xi2dats[fs_sample_name]) )
        
        # We're only going to want some of the entries in computing chi^2.

        # Now load the covariance matrix.
        cov = np.loadtxt(self.basedir+self.covfn)/self.cov_fac
        
        # We're only going to want some of the entries in computing chi^2.
        
        # this is going to tell us how many indices to skip to get to the nth multipole
        startii = 0
        
        for ss, fs_sample_name in enumerate(self.fs_sample_names):
            
            kcut = (self.kdats[fs_sample_name] > self.fs_mmaxs[ss])\
                          | (self.kdats[fs_sample_name] < self.fs_kmins[ss])
            
            for i in np.nonzero(kcut)[0]:     # FS Monopole.
                ii = i + startii
                cov[ii, :] = 0
                cov[ :,ii] = 0
                cov[ii,ii] = 1e25
            
            startii += self.kdats[fs_sample_name].size
            
            kcut = (self.kdats[fs_sample_name] > self.fs_qmaxs[ss])\
                       | (self.kdats[fs_sample_name] < self.fs_kmins[ss])
            
            for i in np.nonzero(kcut)[0]:       # FS Quadrupole.
                ii = i + startii
                cov[ii, :] = 0
                cov[ :,ii] = 0
                cov[ii,ii] = 1e25
            
            startii += self.kdats[fs_sample_name].size

            if self.hexa:
                kcut = (self.kdats[fs_sample_name] > self.fs_hmaxs[ss])\
                           | (self.kdats[fs_sample_name] < self.fs_kmins[ss])

                for i in np.nonzero(kcut)[0]:       # FS Hexadecapole.
                    ii = i + startii
                    cov[ii, :] = 0
                    cov[ :,ii] = 0
                    cov[ii,ii] = 1e25

                startii += self.kdats[fs_sample_name].size
        
        for ss, bao_sample_name in enumerate(self.bao_sample_names):
            
            rcut = (self.rdats[bao_sample_name] < self.bao_rmins[ss])\
                              | (self.rdats[bao_sample_name] > self.bao_rmaxs[ss])
            
            for i in np.nonzero(rcut)[0]:
                ii = i + startii
                cov[ii,:] = 0
                cov[:,ii] = 0
                cov[ii,ii] = 1e25
                
            startii += self.rdats[bao_sample_name].size

            if len(self.bao_ells)>1:
            
                for i in np.nonzero(rcut)[0]:
                    ii = i + startii
                    cov[ii,:] = 0
                    cov[:,ii] = 0
                    cov[ii,ii] = 1e25
                
                startii += self.rdats[bao_sample_name].size
        
        
        
        # Copy it and save the inverse.
        self.cov  = cov
        self.cinv = np.linalg.inv(self.cov)*self.invcov_fac
        #print(self.sample_name, np.diag(self.cinv)[:10])
        
        #load templates for rotation:
        self.rot_templates = {}
        # for ii, fs_sample_name in enumerate(self.fs_sample_names):
        tmpfile = np.loadtxt(self.basedir + self.template_fn)
        self.rot_templates['mono'] = tmpfile[:,0]
        self.rot_templates['quad'] = tmpfile[:,1]
        if self.photo_tmp:
            self.photo_template = tmpfile[:,2]
        # print(self.rot_templates['mono'],self.rot_templates['quad'])
        
        
        # Finally load the window function matrix.
        self.matWs = {}
        for ii, fs_sample_name in enumerate(self.fs_sample_names):
            self.matWs[fs_sample_name] = np.loadtxt(self.basedir+self.fs_matWfns[ii])
        
        self.w_kin = np.loadtxt(self.basedir+self.w_kin_fn)
        
    def combine_bias_terms_pkell(self,bvec, p0ktable, p2ktable, p4ktable):
        '''
        Same as function above but for the multipoles.
        
        Returns k, p0, p2, p4, assuming AP parameters from input p{ell}ktable
        '''
    
        b1,b2,bs,b3,alpha0,alpha2,alpha4,alpha6,sn,sn2,sn4 = bvec

        bias_monomials = np.array([1, b1, b1**2,\
                                   b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2, b3, b1*b3,\
                                   alpha0, alpha2, alpha4,alpha6,sn,sn2,sn4])

        p0 = np.sum(p0ktable * bias_monomials,axis=1)
        p2 = np.sum(p2ktable * bias_monomials,axis=1)
        p4 = np.sum(p4ktable * bias_monomials,axis=1)
        
        return p0, p2, p4
    
    def fs_predict(self, fs_sample_name,zfid, thetas=None):
        """Use the PT model to compute P_ell, given biases etc."""
        
        pp   = self.provider
        # zfid = self.zfid
        zstr = "%.2f" %(zfid)
        taylorPTs = pp.get_result('taylor_pk_ell_mod')
        kv, p0ktable, p2ktable, p4ktable = taylorPTs[zstr]

        self.PT_model = taylorPTs['PT_model']

        #
        sig8 = pp.get_param('sigma8')
        Om = pp.get_param('omegam')
        
        zPars = pp.get_result('zPars')
        sig8_z = zPars[zstr][0]
        f_z = zPars[zstr][1]
        #sig8 = pp.get_result('sigma8')
        
        ##We sample in Lagrangian bias parameters scaled by sigma8
        b1_L   = pp.get_param('bsig8_' + fs_sample_name)/sig8_z**(self.bsig8) - 1.
        # b1   = pp.get_param('b1_' + fs_sample_name)
        b2_L   = pp.get_param('b2sig8_' + fs_sample_name)/(sig8_z**(2*self.bsig8))
        bs_L   = pp.get_param('bssig8_' + fs_sample_name)/(sig8_z**(2*self.bsig8))
        b3_L   = pp.get_param('b3sig8_' + fs_sample_name)/(sig8_z**(3*self.bsig8))

        if self.PT_model == 'EPT':  #If using Eulerian Perturbation Theory, convert Lagrangian biases to Eulerian
            b1 = b1_L + 1.
            b2 = b2_L + (8./21.)*b1_L
            bs = bs_L - (2./7.)*b1_L
            b3 = 3*b3_L + b1_L
        else:
            b1,b2,bs,b3 = b1_L, b2_L, bs_L, b3_L
        
        # Instead of calling the linear parameters directly we will now analytically marginalize over them
        
        if thetas is None:
            alp0_tilde = self.linear_param_means['alpha0_' + fs_sample_name]
            alp2_tilde = self.linear_param_means['alpha2_' + fs_sample_name]
            sn0 = self.linear_param_means['SN0_' + fs_sample_name]
            sn2 = self.linear_param_means['SN2_' + fs_sample_name]
            if self.hexa:
                alp4_tilde = self.linear_param_means['alpha4_' + fs_sample_name]
                sn4 = self.linear_param_means['SN4_' + fs_sample_name]
            else: alp4_tilde,sn4 = 0.,0.
        else:
            alp0_tilde = thetas['alpha0_' + fs_sample_name]
            alp2_tilde = thetas['alpha2_' + fs_sample_name]
            sn0 = thetas['SN0_' + fs_sample_name]
            sn2 = thetas['SN2_' + fs_sample_name]
            
            if self.hexa:
                alp4_tilde = thetas['alpha4_' + fs_sample_name]
                sn4 = thetas['SN4_' + fs_sample_name]
            else: alp4_tilde,sn4 = 0.,0.
            
        alp0 = (1+b1)**2 * alp0_tilde
        alp2 = f_z*(1+b1)*(alp0_tilde+alp2_tilde)
        alp4 = f_z*(f_z*alp2_tilde+(1+b1)*alp4_tilde)
        alp6 = f_z**2*alp4_tilde
 
        bias = [b1, b2, bs, b3]
        cterm = [alp0,alp2,alp4,alp6]
        stoch = [sn0, sn2, sn4]
        # print(bias+cterm+stoch)
        if self.AP_scale:
            qpar,qperp = zPars[zstr][2],zPars[zstr][3]
            Aap = (qpar*qperp**2)**(-1)
            cterm = [a/Aap for a in cterm]
            stoch = [n/Aap for n in stoch]
            bias  = [b/np.sqrt(Aap) for b in bias]
        
        bvec = bias + cterm + stoch
        # print(bvec)
        self.bvec = bvec
        #print(self.zstr, b1, sig8)
        
        
        p0, p2, p4 = self.combine_bias_terms_pkell(bvec, p0ktable, p2ktable, p4ktable)
        
        #np.savetxt('pells_' + self.zstr + '_' + self.sample_name + '.txt',[kv,p0,p2,p4])
        
        # Put a point at k=0 to anchor the low-k part of the Spline.
        kv,p0 = np.append([0.,],kv),np.append([0.0,],p0)
        p2 = np.append([0.,],p2)
        p4 = np.append([0.0,],p4)
        tt    = np.array([kv,p0,p2,p4]).T
        
        if np.any(np.isnan(tt)):
            H0 = pp.get_param('H0_emu')
            print("NaN's encountered. Parameter values are: ", str(w0,wa,Om,H0))
        
        return(tt)
        #
    
    def bao_predict(self, bao_sample_name,zfid):
        
        pp   = self.provider
        zstr = "%.2f" %(zfid)
        # delta = 0.06
        
        B1   = pp.get_param('B1_' + bao_sample_name)
        F   = pp.get_param('F_' + bao_sample_name)
        
        # # Analytically marginalize linear parmaeters so these are obtained differently
        # if thetas is None:
        #     Mpoly = [self.linear_param_means[param_name + '_' + bao_sample_name] for param_name in self.mpoly_names]
        #     if len(self.bao_ells)>1:
        #         Qpoly = [self.linear_param_means[param_name + '_' + bao_sample_name] for param_name in self.qpoly_names]
        #         Qspl = [self.linear_param_means[param_name + '_' + bao_sample_name] for param_name in self.qspl_names]
        # else:
        #     # Mpoly = [thetas[param_name + '_' + bao_sample_name] for param_name in ['M0','M1',]]
        #     # Qpoly = [thetas[param_name + '_' + bao_sample_name] for param_name in ['Q0','Q1',]]
        #     Mpoly = [thetas[param_name + '_' + bao_sample_name] for param_name in self.mpoly_names]
        #     if len(self.bao_ells)>1:
        #         Qpoly = [thetas[param_name + '_' + bao_sample_name] for param_name in self.qpoly_names]
        #         Qspl = [thetas[param_name + '_' + bao_sample_name] for param_name in self.qspl_names]
        #M0, M1, M2 = [pp.get_param(param_name + '_' + bao_sample_name) for param_name in ['M0','M1','M2']]
        #Q0, Q1, Q2 = [pp.get_param(param_name + '_' + bao_sample_name) for param_name in ['Q0','Q1','Q2']]
        
        taylorPTs = pp.get_result('taylor_xi_ell_mod')
        rvec, xi0table, xi2table = taylorPTs[zstr]
        
        xi0t = xi0table[:,0] + B1*xi0table[:,1] + F*xi0table[:,2] \
             + B1**2 * xi0table[:,3] + F**2 * xi0table[:,4] + B1*F*xi0table[:,5]
        
        xi2t = xi2table[:,0] + B1*xi2table[:,1] + F*xi2table[:,2] \
             + B1**2 * xi2table[:,3] + F**2 * xi2table[:,4] + B1*F*xi2table[:,5]
        
        # xi0t += Mpoly[0] + Mpoly[1]*(rvec*self.fs_kmins[0]/(2*pi))**2  #polyval(rvec*np.pi/, Mpoly)
        # if len(self.bao_ells)>1:
        #     xi2t += Qpoly[0] + Qpoly[1]*(rvec*self.fs_kmins[0]/(2*pi))**2
        #     xi2t += self.delta**3*(Qspl[0]*self.B20 + Qspl[1]*self.B21)
        
        
        return np.array([rvec,xi0t,xi2t]).T
    

        
    def fs_observe(self,tt,fs_sample_name,thetas=None):
        """Apply the window function matrix to get the binned prediction."""
        
        
        # print(fs_sample_name)
        # Have to stack ell=0, 2 & 4 in bins of 0.001h/Mpc from 0-0.4h/Mpc.
        # kv  = np.linspace(0.0,0.4,400,endpoint=False) + 0.0005
        # maxk = self.sp_kmax[fs_sample_name]
        # kv  = np.linspace(0.0,maxk,int(maxk/0.001),endpoint=False) + 0.0005
        kv  = self.w_kin
        thy =                     Spline(tt[:,0],tt[:,1],ext=3)(kv)
        # thy = np.concatenate([thy,Spline(tt[:,0],tt[:,1],ext=3)(kv)])
        thy = np.concatenate([thy,Spline(tt[:,0],tt[:,2],ext=3)(kv)])
        # thy = np.concatenate([thy,Spline(tt[:,0],tt[:,2],ext=3)(kv)])
        thy = np.concatenate([thy,Spline(tt[:,0],tt[:,3],ext=3)(kv)])
        
        if np.any(np.isnan(thy)) or np.max(thy) > 1e8:
            hub = self.provider.get_param('H0') / 100.
            sig8 = self.provider.get_param('sigma8')
            ns = self.provider.get_param('ns')
            Om = self.provider.get_param('omegam')
            print("NaN's encountered in PREDICT. Parameter values are: ns={},H0={},Om={},sig8={}".format(ns,hub,Om,sig8))
        
        # wide angle
        # expanded_model = np.matmul(self.matMs[fs_sample_name], thy )
        # Convolve with window (true) −> (conv) see eq. 2.18
        # Multiply by ad-hoc factor
        # convolved_model = np.matmul(self.matWs[fs_sample_name], expanded_model )
        convolved_model = np.matmul(self.matWs[fs_sample_name], thy )
        
        # keep only the monopole and quadrupole
        # convolved_model = convolved_model[self.fitiis[fs_sample_name]]
        
        #Marginalize over template parameters for rotation
        if thetas is None:
            s0 = self.linear_param_means['s0_' + fs_sample_name]
            s2 = self.linear_param_means['s2_' + fs_sample_name]
            if self.photo_tmp:
                ph = self.linear_param_means['ph_' + fs_sample_name]
                tmp_ph = self.photo_template
            else:
                ph = 0
                tmp_ph = np.zeros(len(convolved_model))
        else:
            s0 = thetas['s0_' + fs_sample_name]
            s2 = thetas['s2_' + fs_sample_name]
            if self.photo_tmp:
                ph = thetas['ph_' + fs_sample_name]
                tmp_ph = self.photo_template
            else:
                ph = 0
                tmp_ph = np.zeros(len(convolved_model))
            
        tmp0 = self.rot_templates['mono']
        tmp2 = self.rot_templates['quad']
        
        convolved_model += s0*tmp0 + s2*tmp2 + ph*tmp_ph
        
        # Save the model:
        self.pconv[fs_sample_name] = convolved_model
    
        return convolved_model
    
    def bao_observe(self, tt, bao_sample_name, matrix=True,thetas=None):
        '''
        Bin the BAO results... probabaly should eventually use a matrix.
        '''

        # Analytically marginalize linear parmaeters so these are obtained differently
        if thetas is None:
            Mpoly = [self.linear_param_means[param_name + '_' + bao_sample_name] for param_name in self.mpoly_names]
            if len(self.bao_ells)>1:
                Qpoly = [self.linear_param_means[param_name + '_' + bao_sample_name] for param_name in self.qpoly_names]
                Qspl = [self.linear_param_means[param_name + '_' + bao_sample_name] for param_name in self.qspl_names]
        else:
            # Mpoly = [thetas[param_name + '_' + bao_sample_name] for param_name in ['M0','M1',]]
            # Qpoly = [thetas[param_name + '_' + bao_sample_name] for param_name in ['Q0','Q1',]]
            Mpoly = [thetas[param_name + '_' + bao_sample_name] for param_name in self.mpoly_names]
            if len(self.bao_ells)>1:
                Qpoly = [thetas[param_name + '_' + bao_sample_name] for param_name in self.qpoly_names]
                Qspl = [thetas[param_name + '_' + bao_sample_name] for param_name in self.qspl_names]
        
        rdat = self.rdats[bao_sample_name]
        
        if matrix:
            # If no binning matrix for this sample yet, make it.
            if self.binmat[bao_sample_name] is None:  
                
                dr = rdat[1] - rdat[0]
                
                rth = tt[:,0]
                Nvec = len(rth)

                bin_mat = np.zeros( (len(rdat), Nvec) )

                for ii in range(Nvec):
                    # Define basis vector
                    xivec = np.zeros_like(rth); xivec[ii] = 1
    
                    # Define the spline:
                    thy = Spline(rth, xivec, ext='const')
    
                    # Now compute binned basis vector:
                    tmp = np.zeros_like(rdat)
    
                    for i in range(rdat.size):
                        kl = rdat[i]-dr/2
                        kr = rdat[i]+dr/2

                        ss = np.linspace(kl, kr, 100)
                        p     = thy(ss)
                        tmp[i]= np.trapezoid(ss**2*p,x=ss)*3/(kr**3-kl**3)
        
                    bin_mat[:,ii] = tmp
                
                self.binmat[bao_sample_name] = np.array(bin_mat)
            
            tmp0 = np.dot(self.binmat[bao_sample_name], tt[:,1])
            tmp2 = np.dot(self.binmat[bao_sample_name], tt[:,2])
        
        else:
            thy0 = Spline(tt[:,0],tt[:,1],ext='extrapolate')
            thy2 = Spline(tt[:,0],tt[:,2],ext='extrapolate')
            #thy4 = Spline(tt[:,0],tt[:,3],ext='extrapolate')
        
            dr   = rdat[1]- rdat[0]
        
            tmp0 = np.zeros_like(rdat)
            tmp2 = np.zeros_like(rdat)
        
            for i in range(rdat.size):
            
                kl = rdat[i]-dr/2
                kr = rdat[i]+dr/2

                ss = np.linspace(kl, kr, 100)
                p0     = thy0(ss)
                tmp0[i]= np.trapz(ss**2*p0,x=ss)*3/(kr**3-kl**3)
                p2     = thy2(ss)
                tmp2[i]= np.trapz(ss**2*p2,x=ss)*3/(kr**3-kl**3)
                #p4     = thy4(ss)
                #tmp4[i]= np.trapz(ss**2*p4,x=ss)*3/(kr**3-kl**3)
            
        #self.xith[bao_sample_name] = np.concatenate((tmp0,tmp2))
        tmp0 += Mpoly[0] + Mpoly[1]*(rdat*self.fs_kmins[0]/(2*pi))**2
        if len(self.bao_ells) == 2:
            tmp2 += Qpoly[0] + Qpoly[1]*(rdat*self.fs_kmins[0]/(2*pi))**2
            tmp2 += self.delta**3*(Qspl[0]*self.B20[bao_sample_name] + Qspl[1]*self.B21[bao_sample_name])
            return np.concatenate((tmp0,tmp2))
        else:
            return tmp0
    

class Taylor_pk_theory_zs(Theory):
    """
    A class to return a set of derivatives for the Taylor series of Pkell.
    """
    zfids: list
    Rsmooth: list
    s8_filenames: list
    # plin_filenames: list
    basedir: str
    # omega_nu: float
    bao_sample_names: list
    fid_dists: list
    
    def initialize(self):
        """Sets up the class by loading the derivative matrices."""
        
        # print("Loading Taylor series.")
        
        # First Load Sigma8 class:
        # self.compute_sigma8 = Compute_Sigma8_wCDM(self.basedir + self.s8_filename)
        
        self.compute_theory = direct_fit_theory(zs = self.zfids, fid_dists = self.fid_dists)
        
    
    def get_requirements(self):
        """What we need in order to provide P_ell."""
        zmax = max(self.zfids)
        zg  = np.linspace(0,zmax,100,endpoint=True)
        # Don't need sigma8_z, fsigma8 or radial distance
        # here, but want them up in likelihood and they
        # only depend on cosmological things (not biases).
        #
        req = {\
            #    'ns': None,\
               'omega_b': None,\
               'omega_cdm': None,\
            #    'omegam': None,\
               # 'H0': None,\
            #    'theta_s_100': None,\
            #    'Omega_k': None,\
            #    'tau': None,\
            #    'w': None,\
            #    'wa': None,\
            #    'logA': None,\
               'm_ncdm': None,\
               'Pk_interpolator': {'k_max': 30, 'z':zg,\
                                   'nonlinear': False,\
                                   'vars_pairs': [['delta_nonu','delta_nonu']]},\
            #    'Pk_grid': {'k_max': 30, 'z':zg,\
            #                        'nonlinear': False,\
            #                        'vars_pairs': [['delta_nonu','delta_nonu']]},\
               'Hubble':   {'z': [0.0,] + self.zfids},\
               'sigma8_z': {'z': [0.0,] + self.zfids},\
               'fsigma8':  {'z': [0.0,] + self.zfids},\
               'comoving_radial_distance': {'z': [0,] + self.zfids},\
            #    'scale_independent_growth_factor': {'z': [0,] + self.zfids},\
            #    'scale_independent_growth_factor_f': {'z': [0,] + self.zfids},\
               'rs_drag': None,\
               'h': None,\
                }
              
        
        for bao_sample_name in self.bao_sample_names:
            req_bao = {\
                   'Sigpar_' +  bao_sample_name: None,\
                   'Sigperp_' +  bao_sample_name: None,\
                   'Sigs_' +  bao_sample_name: None,\
                    }
            req = {**req, **req_bao}
        
        return(req)
    def get_can_provide(self):
        """What do we provide: a Taylor series class for pkells."""
        return ['taylor_pk_ell_mod','taylor_xi_ell_mod','zPars']
    
    # def get_can_provide_params(self):
    #     # return ['sig8','H0']
    #     return ['sig8']
    
    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        Just load up the derivatives and things.
        """
        pp = self.provider

        # print(pp)
        # w0 = pp.get_param('w')
        # wa = pp.get_param('wa')
        # ns = pp.get_param('ns')

        # theta_star = pp.get_param('theta_s_100')
        # # ns = 0.96
        # # H0 = pp.get_param('H0')/
        # # hub =  H0/ 100.
        # logA = pp.get_param('logA')
        # As = 1e-10 * np.exp(logA)
        h = pp.get_param('h')
        omega_b = pp.get_param('omega_b')
        omega_cdm = pp.get_param('omega_cdm')
        Mnu = pp.get_param('m_ncdm')
        OmM = (omega_cdm + omega_b + Mnu/93.14)/h**2

        ki   = np.logspace(-3.0,1.0,200)
        pi_interp   = pp.get_Pk_interpolator(nonlinear=False,var_pair=['delta_nonu','delta_nonu'])
        pi_0   = pi_interp.P(0.0,ki*h)*h**3
        # print(k_arr,z_arr)
        # print(pi.shape)
        # fsigma8s = pp.get_fsigma8()

        speed_of_light = 2.99792458e5
        # Hubbles = pp.get_Hubble()
        # h = Hubbles[0]/100.
        # print('h = ',h)
        # chizs = pp.get_comoving_radial_distance()/h

        # fzs = pp.scale_independent_growth_factor_f()
        # Dzs = pp.get_scale_independent_growth_factor()
        sigma8_0 = pp.get_sigma8_z(0.0)[0]
        rs_drag = pp.get_param('rs_drag')

        qBAO = rs_drag * h


        # Initialize EPT
        self.compute_theory.setup_REPT(ki,pi_0, qBAO)
        ptables = {'PT_model': 'EPT'} #Can include LPT once fast version is ready
        xitables = {}
        zPars = {}
            
        # ll=0
        # for zfid,R in zip(self.zfids,self.Rsmooth):
        for ii,zfid in enumerate(self.zfids):
            zstr = "%.2f" %(zfid)

            Hz = pp.get_Hubble(zfid)[0]/h #pkclass.Hubble(zfid) * speed_of_light / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
            chiz = pp.get_comoving_radial_distance(zfid)[0]*h #pkclass.angular_distance(zfid) * (1.+zfid) * h # this is the comoving radius in units of Mpc/h 
            # fz = pkclass.scale_independent_growth_factor_f(zfid)
            # Dz = pkclass.scale_independent_growth_factor(zfid)
            sig8z = pp.get_sigma8_z(zfid)[0]
            fsigma8 = pp.get_fsigma8(zfid)[0]
            fz = fsigma8/sig8z
            Dz = sig8z/sigma8_0

            pi_z = pi_interp.P(zfid,ki*h)*h**3
            kv,p0ktable,p2ktable,p4ktable = self.compute_theory.compute_pell_tables_EPT(zfid,ki,pi_z,Hz,chiz,fz,Dz)
            
            ptables[zstr] = (kv, p0ktable, p2ktable, p4ktable)
            
            
            sigmas = (pp.get_param('Sigpar_' +  self.bao_sample_names[ii]),\
                        pp.get_param('Sigperp_' +  self.bao_sample_names[ii]),\
                        pp.get_param('Sigs_' +  self.bao_sample_names[ii]))
            
            rv, xi0table, xi2table = self.compute_theory.compute_xiell_tables(zfid,ki,pi_z,Hz,chiz,qBAO,fz,  R=self.Rsmooth[ii], rmin=50, rmax=160, dr=0.1, sigs = sigmas)
            
            xitables[zstr] = (rv, xi0table, xi2table)
            qpar,qperp = self.compute_theory.qpar[zstr], self.compute_theory.qperp[zstr]
            # print(qpar,qperp,chiz,self.fid_dists[ii][1])
            zPars[zstr] = [sig8z,fz]
            zPars[zstr] += [qpar,qperp]
            
        
            # ll += 1
            
        #state['sigma8'] = sig8
        # H0_emu = compute_zpars.compute_H0(cosmopars,order = 5)*100 
        # state['derived'] = {'sig8': sigma8_zs[0],'H0': h*100.,'Omegam': OmM}
        # state['derived'] = {'sigma8': sigma8_0,'omegam': OmM}
        state['zPars'] = zPars
        state['taylor_pk_ell_mod'] = ptables
        state['taylor_xi_ell_mod'] = xitables


class direct_pk_theory_zs(Theory):
    """
    A class to return a set of derivatives for the Taylor series of Pkell.
    """
    zfids: list
    fid_dists: list
    
    def initialize(self):
        """Sets up the class"""
        
        self.compute_theory = direct_fit_theory(zs = self.zfids, fid_dists = self.fid_dists)
        
    
    def get_requirements(self):
        """What we need in order to provide P_ell."""
        zmax = max(self.zfids)
        zg  = np.linspace(0,zmax,100,endpoint=True)
        req = {\
               'omega_b': None,\
               'omega_cdm': None,\
               'm_ncdm': None,\
               'Pk_interpolator': {'k_max': 30, 'z':zg,\
                                   'nonlinear': False,\
                                   'vars_pairs': [['delta_nonu','delta_nonu']]},\
               'Hubble':   {'z': [0.0,] + self.zfids},\
               'sigma8_z': {'z': [0.0,] + self.zfids},\
               'fsigma8':  {'z': [0.0,] + self.zfids},\
               'comoving_radial_distance': {'z': [0,] + self.zfids},\
               'rs_drag': None,\
               'h': None,\
                }
        
        
        return(req)
    def get_can_provide(self):
        """What do we provide: a Taylor series class for pkells."""
        return ['taylor_pk_ell_mod','zPars']
    
    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        Just load up the derivatives and things.
        """
        pp = self.provider
        h = pp.get_param('h')
        omega_b = pp.get_param('omega_b')
        omega_cdm = pp.get_param('omega_cdm')
        Mnu = pp.get_param('m_ncdm')
        OmM = (omega_cdm + omega_b + Mnu/93.14)/h**2

        ki   = np.logspace(-3.0,1.0,200)
        pi_interp   = pp.get_Pk_interpolator(nonlinear=False,var_pair=['delta_nonu','delta_nonu'])
        pi_0   = pi_interp.P(0.0,ki*h)*h**3

        speed_of_light = 2.99792458e5
        sigma8_0 = pp.get_sigma8_z(0.0)[0]
        rs_drag = pp.get_param('rs_drag')

        qBAO = rs_drag * h


        # Initialize EPT
        self.compute_theory.setup_REPT(ki,pi_0, qBAO)
        ptables = {'PT_model': 'EPT'} #Can include LPT once fast version is ready
        xitables = {}
        zPars = {}
            
        # ll=0
        # for zfid,R in zip(self.zfids,self.Rsmooth):
        for ii,zfid in enumerate(self.zfids):
            zstr = "%.2f" %(zfid)

            Hz = pp.get_Hubble(zfid)[0]/h #pkclass.Hubble(zfid) * speed_of_light / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
            chiz = pp.get_comoving_radial_distance(zfid)[0]*h #pkclass.angular_distance(zfid) * (1.+zfid) * h # this is the comoving radius in units of Mpc/h 
            sig8z = pp.get_sigma8_z(zfid)[0]
            fsigma8 = pp.get_fsigma8(zfid)[0]
            fz = fsigma8/sig8z
            Dz = sig8z/sigma8_0

            pi_z = pi_interp.P(zfid,ki*h)*h**3
            kv,p0ktable,p2ktable,p4ktable = self.compute_theory.compute_pell_tables_EPT(zfid,ki,pi_z,Hz,chiz,fz,Dz)
            qpar,qperp = self.compute_theory.qpar[zstr], self.compute_theory.qperp[zstr]
            ptables[zstr] = (kv, p0ktable, p2ktable, p4ktable)

            zPars[zstr] = [sig8z,fz]
            zPars[zstr] += [qpar,qperp]
            
 
        state['zPars'] = zPars
        state['taylor_pk_ell_mod'] = ptables

class direct_Xi_theory_zs(Theory):
    """
    A class to return a set of derivatives for the Taylor series of Pkell.
    """
    zfids: list
    Rsmooth: list
    fid_dists: list
    bao_sample_names: list
    
    def initialize(self):
        """Sets up the class."""
        
        self.compute_theory = direct_fit_theory(zs = self.zfids, fid_dists = self.fid_dists)
        
    
    def get_requirements(self):
        """What we need in order to provide P_ell."""
        zmax = max(self.zfids)
        zg  = np.linspace(0,zmax,100,endpoint=True)
        # Don't need sigma8_z, fsigma8 or radial distance
        # here, but want them up in likelihood and they
        # only depend on cosmological things (not biases).
        #
        req = {'Pk_interpolator': {'k_max': 30, 'z':zg,\
                                   'nonlinear': False,\
                                   'vars_pairs': [['delta_nonu','delta_nonu']]},\
            #    'Pk_grid': {'k_max': 30, 'z':zg,\
            #                        'nonlinear': False,\
            #                        'vars_pairs': [['delta_nonu','delta_nonu']]},\
               'Hubble':   {'z': [0.0,] + self.zfids},\
               'sigma8_z': {'z': [0.0,] + self.zfids},\
               'fsigma8':  {'z': [0.0,] + self.zfids},\
               'comoving_radial_distance': {'z': [0,] + self.zfids},\
               'rs_drag': None,\
               'h': None,\
                }
              
        
        for bao_sample_name in self.bao_sample_names:
            req_bao = {\
                   'Sigpar_' +  bao_sample_name: None,\
                   'Sigperp_' +  bao_sample_name: None,\
                   'Sigs_' +  bao_sample_name: None,\
                    }
            req = {**req, **req_bao}
        
        return(req)
    def get_can_provide(self):
        """What do we provide: a Taylor series class for pkells."""
        return ['taylor_xi_ell_mod']
    
    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        Just load up the derivatives and things.
        """
        pp = self.provider
        h = pp.get_param('h')

        ki   = np.logspace(-3.0,1.0,200)
        pi_interp   = pp.get_Pk_interpolator(nonlinear=False,var_pair=['delta_nonu','delta_nonu'])

        speed_of_light = 2.99792458e5
        sigma8_0 = pp.get_sigma8_z(0.0)[0]
        rs_drag = pp.get_param('rs_drag')

        qBAO = rs_drag * h

        xitables = {}
        for ii,zfid in enumerate(self.zfids):
            zstr = "%.2f" %(zfid)

            Hz = pp.get_Hubble(zfid)[0]/h #pkclass.Hubble(zfid) * speed_of_light / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
            chiz = pp.get_comoving_radial_distance(zfid)[0]*h #pkclass.angular_distance(zfid) * (1.+zfid) * h # this is the comoving radius in units of Mpc/h 
            sig8z = pp.get_sigma8_z(zfid)[0]
            fsigma8 = pp.get_fsigma8(zfid)[0]
            fz = fsigma8/sig8z
            Dz = sig8z/sigma8_0

            pi_z = pi_interp.P(zfid,ki*h)*h**3
            
            
            sigmas = (pp.get_param('Sigpar_' +  self.bao_sample_names[ii]),\
                        pp.get_param('Sigperp_' +  self.bao_sample_names[ii]),\
                        pp.get_param('Sigs_' +  self.bao_sample_names[ii]))
            
            rv, xi0table, xi2table = self.compute_theory.compute_xiell_tables(zfid,ki,pi_z,Hz,chiz,qBAO,fz,  R=self.Rsmooth[ii], rmin=50, rmax=160, dr=0.1, sigs = sigmas)
            
            xitables[zstr] = (rv, xi0table, xi2table)
        state['taylor_xi_ell_mod'] = xitables

class direct_Xi_theory_zs_jax(Theory):
    
    """
    A class to return a set of derivatives for the Taylor series of Pkell.
    """
    
    zfids: list
    Rsmooth: list
    fid_dists: list
    bao_sample_names: list
    
    def initialize(self):
        """Sets up the class."""
        import jax
        import jax.numpy as jnp
        from compute_xiell_tables_recsym_jax import compute_xiell_tables
        from cosmoprimo import PowerSpectrumInterpolator1D
        from cosmoprimo import PowerSpectrumBAOFilter

        self.PowerSpectrumInterpolator1D = PowerSpectrumInterpolator1D
        self.PowerSpectrumBAOFilter = PowerSpectrumBAOFilter
        self.compute_xiell_tables = compute_xiell_tables
        
        
        def Pint_func(ki,pi):
            pint = PowerSpectrumInterpolator1D(ki,pi,extrap_kmin=1e-5, extrap_kmax=29.)
            return pint
        
        self.Pinterpolate = jax.jit(Pint_func)
        self.rout = np.arange(50, 160, 0.1)
        
    
    def get_requirements(self):
        """What we need in order to provide P_ell."""
        zmax = max(self.zfids)
        zg  = np.linspace(0,zmax,100,endpoint=True)
        # Don't need sigma8_z, fsigma8 or radial distance
        # here, but want them up in likelihood and they
        # only depend on cosmological things (not biases).
        #
        req = {'Pk_interpolator': {'k_max': 30, 'z':zg,\
                                   'nonlinear': False,\
                                   'vars_pairs': [['delta_nonu','delta_nonu']]},\
            #    'Pk_grid': {'k_max': 30, 'z':zg,\
            #                        'nonlinear': False,\
            #                        'vars_pairs': [['delta_nonu','delta_nonu']]},\
               'Hubble':   {'z': [0.0,] + self.zfids},\
               'sigma8_z': {'z': [0.0,] + self.zfids},\
               'fsigma8':  {'z': [0.0,] + self.zfids},\
               'comoving_radial_distance': {'z': [0,] + self.zfids},\
               'rs_drag': None,\
               'h': None,\
                }
              
        
        for bao_sample_name in self.bao_sample_names:
            req_bao = {\
                   'Sigpar_' +  bao_sample_name: None,\
                   'Sigperp_' +  bao_sample_name: None,\
                   'Sigs_' +  bao_sample_name: None,\
                    }
            req = {**req, **req_bao}
        
        return(req)
    def get_can_provide(self):
        """What do we provide: a Taylor series class for pkells."""
        return ['taylor_xi_ell_mod']
    
    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        Just load up the derivatives and things.
        """
        pp = self.provider
        h = pp.get_param('h')

        ki   = np.logspace(-3.0,1.0,200)
        pi_interp   = pp.get_Pk_interpolator(nonlinear=False,var_pair=['delta_nonu','delta_nonu'])

        speed_of_light = 2.99792458e5
        sigma8_0 = pp.get_sigma8_z(0.0)[0]
        rs_drag = pp.get_param('rs_drag')

        qBAO = rs_drag * h

        xitables = {}
        for ii,zfid in enumerate(self.zfids):
            # print(ii,zfid)
            zstr = "%.2f" %(zfid)

            Hz = pp.get_Hubble(zfid)[0]/h #pkclass.Hubble(zfid) * speed_of_light / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
            chiz = pp.get_comoving_radial_distance(zfid)[0]*h #pkclass.angular_distance(zfid) * (1.+zfid) * h # this is the comoving radius in units of Mpc/h 
            sig8z = pp.get_sigma8_z(zfid)[0]
            fsigma8 = pp.get_fsigma8(zfid)[0]
            fz = fsigma8/sig8z
            Dz = sig8z/sigma8_0

            Hzfid, chizfid = self.fid_dists[ii]
            qpar, qperp = Hzfid / Hz, chiz / chizfid

            pi_z = pi_interp.P(zfid,ki*h)*h**3
            
            
            Sigpar = pp.get_param('Sigpar_' +  self.bao_sample_names[ii])
            Sigperp = pp.get_param('Sigperp_' +  self.bao_sample_names[ii])
            Sigs    =  pp.get_param('Sigs_' +  self.bao_sample_names[ii])
            bao_pars = [qpar,qperp,Sigpar,Sigperp,Sigs]

        
            pint = self.Pinterpolate(ki,pi_z) 
            filt = self.PowerSpectrumBAOFilter(pint)
            
            rv,xi0table, xi2table = self.compute_xiell_tables(pars=bao_pars,ki=filt.k,pi=filt.pk,pnw=filt.pknow,\
                                                         qbao=qBAO,f0 = fz, rout = self.rout,z=zfid,\
                                                         R=self.Rsmooth[ii])
            
            xitables[zstr] = (np.asarray(rv), np.asarray(xi0table), np.asarray(xi2table))
        state['taylor_xi_ell_mod'] = xitables