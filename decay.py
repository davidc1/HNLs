import numpy as np
from math import pi, sqrt, log10, log
import numpy as np

_aEM = 1./137. #
_Gf  = 1.166 * 1e-5 # GeV-2
_m_pi0 = 0.134977 # GeV
_m_pi = 0.139570 # GeV
_f_pi = 0.13041 # GeV [??? wikipedia mentions convention differing by sqrt(2)...]

_m_elec = 0.000511 # GeV
_m_muon = 0.105658 # GeV

_Vud2 = 0.97427**2 # |V_ud|^2

_sW = 0.22290 # square of sine of the Weinberg angle
_C1 = (1/4.) * (1 - 4 * _sW + 8 * _sW**2)
_C2 = (1/2.) * _sW * (2 * _sW - 1)
_C3 = (1/4.) * (1 + 4 * _sW + 8 * _sW**2)
_C4 = (1/2.) * _sW * (2 * _sW + 1)

def _DIRAC(alpha,beta):
    if (alpha==beta): return 1
    return 0

def _LEPTON_MASS(alpha):
    if (alpha==1): return _m_elec
    if (alpha==2): return _m_muon
    if (alpha==3): return _m_muon
    return 0

def _L(yl): # ??? is this log10
    N = 1-4*(yl**2)
    if (N < 0): return -1
    N = sqrt(N)
    #B = ( 1 - 3*yl2 - (1-yl2) * N ) / ( yl2 * (1 + N ) )
    B = (1 + N) / (2 * yl)
    #print ("N = %g"%N)
    #print ("B = %g"%B)
    if (B <= 0): return -1
    return - 4 * log( B )


# channel decaying to \nu + \nu + \nu [A.1]

def _Gamma_nununu(HNL_MASS,U):
    return _Gf**2 * HNL_MASS**5 * U**2 / (192 * pi**3)

# channel decaying to \nu_{\alpha} + \gamma [A.2]
def _Gamma_nugamma(HNL_MASS,U):
    return 9 * _aEM * _Gf**2 * HNL_MASS**5 * U**2 / (512 * pi**4)

# channel decaying to \pi^0 + \nu_{\alpha} [A.3]
def _Gamma_nupi0(HNL_MASS,U):
    if (HNL_MASS < _m_pi0): return 0
    return ( (_Gf**2 * _f_pi**2 * HNL_MASS**3 * U**2) / (32 * pi) ) * ( 1 - (_m_pi0/HNL_MASS)**2 )**2

# channel decaying to \pi^+ and lepton [A.4]
def _Gamma_pilepton(HNL_MASS,U,alpha):
    ML = _LEPTON_MASS(alpha)
    if (HNL_MASS < (_m_pi + ML)): return 0
    A = (1 - ( ( (_m_pi - ML) / HNL_MASS )**2 ) ) * (1 - ( (_m_pi + ML) / HNL_MASS )**2 )
    if (A <= 0): return 0
    return ( _Gf**2 * _f_pi**2 * HNL_MASS**3 * _Vud2 * U**2 / (16 * pi) ) * ( ( 1 - (_LEPTON_MASS(alpha)/HNL_MASS)**2 )**2 - (_m_pi/HNL_MASS)**2 * (1 + (_LEPTON_MASS(alpha) / HNL_MASS)**2 ) )**2 * sqrt( A )

# channel decaying to l^{-}_{\alpha} + l^{+}_{\beta} + \nu_{\beta} [A.5]
# ??? is log in this expression base 10?
def _Gamma_llnu1(HNL_MASS,U,alpha,beta):
    if (alpha == beta): return 0
    if (HNL_MASS < (_LEPTON_MASS(alpha) + _LEPTON_MASS(beta)) ): return 0
    xl = max(_LEPTON_MASS(beta),_LEPTON_MASS(alpha)) / HNL_MASS
    #xl = _m_elec / HNL_MASS
    return ( _Gf**2 * HNL_MASS**5 * U**2 / (192 * pi**3) ) * (1 - 8 * xl**2 + 8 * xl**6 - xl**8 - 12 * xl**4 * log(xl**2) )

# channel decaying to l^{-}_{\beta} + l^{+}_{\beta} + \nu_{\alpha} [A.6]
def _Gamma_llnu2(HNL_MASS,U,alpha,beta):
    if (HNL_MASS < 2 * _LEPTON_MASS(beta)): return 0
    yl  = (_LEPTON_MASS(beta) / HNL_MASS)
    yl2 = yl**2
    #print ("yl2 = %g"%yl2)
    A = 1-4*yl2
    #print ("A = %g"%A)
    if (A <= 0): return 0
    A = sqrt(A)
    delta = _DIRAC(alpha,beta)
    L = _L(yl)
    #print ("L = %g"%L)
    if (L == -1): return 0
    TERM1 = ( _Gf**2 * HNL_MASS**5 * U**2 / (192 * pi**3) )
    TERM2 = (_C1 * (1 - delta ) + _C3 * delta )
    TERM3 = ( ( 1 - 14*yl2 - 2*yl2**2 - 12*yl2**3 ) * A + 12*yl2**2 * (yl2**2-1) * L )
    TERM4 = 4 * (_C2 * (1-delta) + _C4 * delta) * (yl2 * (2+10*yl2-12*yl2**2) * A + 6*yl2**2*(1-2*yl2+2*yl2**2) * L )
    #print ("TERM1 : %g"%TERM1)
    #print ("TERM2 : %g"%TERM2)
    #print ("TERM3 : %g"%TERM3)
    #print ("TERM4 : %g"%TERM4)
    return TERM1 * ( TERM2 * TERM3 + TERM4 )


_Gamma_nununu_v   = np.vectorize(_Gamma_nununu)
_Gamma_nugamma_v  = np.vectorize(_Gamma_nugamma)
_Gamma_nupi0_v    = np.vectorize(_Gamma_nupi0)
_Gamma_pilepton_v = np.vectorize(_Gamma_pilepton)
_Gamma_llnu1_v    = np.vectorize(_Gamma_llnu1)
_Gamma_llnu2_v    = np.vectorize(_Gamma_llnu2)
