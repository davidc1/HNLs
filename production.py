import numpy as np

# load BNB neutrino flux [from MicroBooNE paper XXX]

def MicroBooNE_FLUX():
    flux_data = np.loadtxt('MicroBooNE_Flux_Updated.csv',skiprows=2,delimiter=',')

    # flux in units of neutrinos / POT / GeV / cm2
    kaon_flux = []
    pion_flux = []
    for flux_row in flux_data:
        kaon_f = np.abs(flux_row[3]) + np.abs(flux_row[5])
        pion_f = np.abs(flux_row[1])
        kaon_flux.append(kaon_f)
        pion_flux.append(pion_f)

    MAXFLUXPION = np.max(pion_flux)
    MAXFLUXKAON = np.max(kaon_flux)
    BINWIDTH = 0.05 # GeV

# sample from pion/kaon BNB flux at MicroBooNE
# input MASS : HNL mass
# input MESON: pion (211) or kaon (411)
# input N    : number of HNLs to simulate
# return Kinetic Energy in MeV
def BNB_HNL_ENERGY(MASS,MESON,N):
    simulated_v = []

    # set constants appropriately based on whether simulating a pion or a kaon

    # set maximum energy to sample
    # set which flux to use 
    ENERGY_MAX = 0. # GeV
    MAXFLUX = 0
    flux_v = []
    if (MESON == 211):
        ENERGY_MAX = 2.95 # GeV
        flux_v = _PION_FLUX
        MAXFLUX = _MAXFLUXPION
    elif (MESON == 411):
        ENERGY_MAX = 4.95 # GeV
        flux_v = _KAON_FLUX
        MAXFLUX = _MAXFLUXKAON
    else:
        return simulated_v

    while len(simulated_v) < N:
        # simulate an energy from 0 to ENERGY_MAX #GeV
        energy = np.random.uniform(MASS/1e3,ENERGY_MAX)
        # find the bin it falls in [50 MeV bins]
        energybin = int(energy/_BINWIDTH)
        # probability of seeing an event in this energy bin
        p = np.random.uniform(0,_MAXFLUX)
        #print (' p is ',p)
        # is this value smaller than flux prediction at this energy? 
        # if so, continue to simulate event
        if (p < flux_v[energybin]):
            simulated_v.append(energy*1e3-MASS)
    return np.array(simulated_v)

# from article 1912.07622

# strength based on mixing [7.2]
def MIXING(U):
    return ((U**2)/(1-U**2))

# input are the masses of the meson, HNL, and lepton respectively [7.2]
def RHO(MESON,HNL,LEPTON):
    x = (LEPTON/MESON)**2
    y = (HNL/MESON)**2
    return ((x+y-(x-y)**2)) * np.sqrt(1+x**2+y**2-2*(x+y+x*y)) / (x*(1-x)**2)
