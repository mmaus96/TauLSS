import numpy as np
import jax
import jax.numpy as jnp
#from jax.scipy.special import gammaln
from scipy.special import loggamma, gamma
import numpy as np
from jax import jit

from spherical_bessel_transform_ncol_jax import SphericalBesselTransform as SBT
from velocileptors.Utils.spherical_bessel_transform import SphericalBesselTransform as SBT0
from interpax import interp1d
# from linear_theory import f_of_a
# from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD

# from scipy.special import spherical_jn
# from scipy.integrate import simpson as simps
# from scipy.interpolate import interp1d
# from linear_theory import*
from pnw_dst import pnw_dst
# from shapefit import shapefit_factor

#from zeldovich_rsd_recon_fftw import Zeldovich_Recon


# k vector to use:
kvec = jnp.concatenate( (jnp.array([0.0005,]),\
                        jnp.logspace(jnp.log10(0.0015),jnp.log10(0.025),10, endpoint=True),\
                        jnp.arange(0.03,0.51,0.01)) )

kint = jnp.logspace(-3, 2, 2000)
sphr = SBT(kint,L=5,fourier=True,low_ring=False,ncol=1)
sphr0 = SBT0(kint,L=5,fourier=True,low_ring=False)

ngauss = 4
nus, ws = np.polynomial.legendre.leggauss(2*ngauss)
nus_calc = nus[0:ngauss]
nus = jnp.array(nus)
nus_calc = jnp.array(nus_calc)
ws = jnp.array(ws) 
    
L0 = jnp.array(np.polynomial.legendre.Legendre((1))(nus))
L2 = jnp.array(np.polynomial.legendre.Legendre((0,0,1))(nus))

def sphr_func(ell,pt):
    rr, xit = sphr.sph(ell,pt)
    return rr,xit

sphr_jit = jit(sphr_func)

@jit
def compute_bao_pkmu(mu_obs, B1, F, klin, plin, pnw, f0, apar, aperp, R, sigmas):
    '''
    Helper function to get P(k,mu) post-recon in RecSym.
        
    This is turned into Pkell and then Hankel transformed in the bao_predict funciton.
    '''

    
    sigma_par,sigma_perp, sigma_s = sigmas
    # sigma_par,sigma_perp, sigma_s = sigmas
    pw = plin - pnw
    

    Sk = jnp.exp(-0.5*(klin*R)**2)
        
    # Our philosophy here is to take kobs = klin
    # Then we predict P(ktrue) on the klin grid, so that the final answer is
    # Pobs(kobs,mu_obs) ~ Ptrue(ktrue, mu_true) = interp( klin, Ptrue(klin, mu_true) )(ktrue)
    # (Up to a normalization that we drop.)
    # Interpolating this way avoids having to interpolate Pw and Pnw separately.
        
    F_AP = apar/aperp
    AP_fac = jnp.sqrt(1 + mu_obs**2 *(1./F_AP**2 - 1) )
    mu = mu_obs / F_AP / AP_fac
    ktrue = klin/aperp*AP_fac
    
    Fog = (1 + 0.5 * klin**2 * mu**2 * sigma_s**2)**(-2)
        
    # First construct P_{dd,ss,ds} individually
    #New changes: same damp fac for P_{dd,ss,ds}
    # dampfac_dd = np.exp( -0.5 * klin**2 * sigmadd * (1 + f0*(2+f0)*mu**2) )
    damp_free = jnp.exp(-0.5 * klin**2 * (sigma_par**2 * mu**2 + (1-mu**2)*sigma_perp**2))
    pw_damp = damp_free* pw
    
    
    ptrue = (1+B1+F*mu**2)**2 * (pw_damp + Fog*pnw)
    # pmodel = interp1d(klin, ptrue, kind='cubic', fill_value=0,bounds_error=False)(ktrue)
    pmodel = interp1d(ktrue,klin, ptrue, method='cubic', fill_value=0,extrap=0)
    
    return pmodel

@jit
def compute_xiells(rout, B1, F, klin, plin, pnw, f0, apar, aperp, R, sigmas):
        

    # Generate the sampling
    # ngauss = 4
    # nus, ws = np.polynomial.legendre.leggauss(2*ngauss)
    # nus_calc = nus[0:ngauss]
    # nus = jnp.array(nus)
    # nus_calc = jnp.array(nus_calc)
    # ws = jnp.array(ws) 
        
    # L0 = jnp.array(np.polynomial.legendre.Legendre((1))(nus))
    # L2 = jnp.array(np.polynomial.legendre.Legendre((0,0,1))(nus))
    
    pknutable = jnp.zeros((len(nus),len(klin)))
    
    for ii, nu in enumerate(nus_calc):
        pknutable = pknutable.at[ii,:].set(compute_bao_pkmu(nu, B1, F, klin, plin, pnw, f0, apar, aperp, R, sigmas))
 
    pknutable = pknutable.at[ngauss:,:].set(jnp.flip(pknutable[0:ngauss],axis=0))
        
    p0 = 0.5 * jnp.sum((ws*L0)[:,None]*pknutable,axis=0) #+ 1000 * polyval(klin,[m0,m1,m2,m3,m4,m5]) / klin
    p2 = 2.5 * jnp.sum((ws*L2)[:,None]*pknutable,axis=0) #+ 1000 * polyval(klin,[q0,q1,q2,q3,q4,q5]) / klin

    # p0t = interp1d(klin,p0, kind='cubic', bounds_error=False, fill_value=0)(kint)
    # p2t = interp1d(klin,p2, kind='cubic', bounds_error=False, fill_value=0)(kint)
    p0t = interp1d(kint,klin,p0, method='cubic', fill_value=0,extrap=0)
    p2t = interp1d(kint,klin,p2, method='cubic', fill_value=0,extrap=0)
    # print(pknutable)
    damping = jnp.exp(-(kint/10)**2)
    # print(p0t[:10],damping[0],p0t[0]*damping[0])
    # rr0, xi0t = sphr0.sph(0,p0t * damping)
    # rr2, xi2t = sphr0.sph(2,p2t * damping); xi2t *= -1
    
    rr0, xi0t = sphr_jit(0,p0t * damping)
    rr2, xi2t = sphr_jit(2,p2t * damping); xi2t *= -1
    # print(xi0t)
    
    # print(np.shape(xi0t),len(rr0))
    
    # jax.debug.print(xi0t)
    # return interp1d(rr0,xi0t,kind='cubic')(rout), interp1d(rr0,xi2t,kind='cubic')(rout)
    return interp1d(rout,rr0,xi0t[0,:],method='cubic'), interp1d(rout,rr0,xi2t[0,:],method='cubic')
    # return interp1d(rout,rr0,xi0t,method='cubic'), interp1d(rout,rr0,xi2t,method='cubic')
    
@jit
def compute_xiell_tables(pars,ki,pi,pnw,qbao,f0,rout, z=0.51, R=15.):
    

    apar, aperp,Sigpar,Sigperp,Sigs = pars
    sigs = (Sigpar,Sigperp,Sigs)
    # apar, aperp = AP
    speed_of_light = 2.99792458e5


    # Do the Zeldovich reconstruction predictions

    # knw, pnw = pnw_dst(ki, pi)
    pw = pi - pnw
            
    # qbao   = pkclass.rs_drag() * h # want this in Mpc/h units

    # j0 = spherical_jn(0,ki*qbao)
    # Sk = np.exp(-0.5*(ki*R)**2)

    # sigmads_dd = simps( 2./3 * pi * (1-Sk)**2, x = ki) / (2*np.pi**2)
    # sigmads_ss = simps( 2./3 * pi * (-Sk)**2, x = ki) / (2*np.pi**2)
    # sigmads_ds = -simps( 2./3 * pi * (1-Sk)*(-Sk)*j0, x = ki) / (2*np.pi**2) # this minus sign is because we subtract the cross term

    sigmas = (sigs[0], sigs[1], sigs[2])
    
    # Now make the multipoles!
    klin, plin = jnp.array(ki), jnp.array(pi)
    routs = jnp.array(rout)
    
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
    
    xi0table, xi2table = jnp.zeros( (len(routs),6) ), jnp.zeros( (len(routs),6) )
    
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
    xi0table = xi0table.at[:,0].set(xi0_00)
    
    xi0table = xi0table.at[:,1].set(xi0_B1)
    xi0table = xi0table.at[:,2].set(xi0_F)

    xi0table = xi0table.at[:,3].set(xi0_B1sq)
    xi0table = xi0table.at[:,4].set(xi0_Fsq)

    xi0table = xi0table.at[:,5].set(xi0_BF)
    
    xi2table = xi2table.at[:,0].set(xi2_00)
    
    xi2table = xi2table.at[:,1].set(xi2_B1)
    xi2table = xi2table.at[:,2].set(xi2_F)

    xi2table = xi2table.at[:,3].set(xi2_B1sq)
    xi2table = xi2table.at[:,4].set(xi2_Fsq)

    xi2table = xi2table.at[:,5].set(xi2_BF)
    

    return routs,xi0table, xi2table

