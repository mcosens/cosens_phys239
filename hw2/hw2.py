'''
    Maren Cosens
    PHYS 239: Radiative Processes in Astrophysics
    HW #2
    hw2.py: radiative transfer problems
    using Python 2.7
'''
import numpy as np
import matplotlib.pyplot as pl

##Define variables given in problem statement
D=100 #cloud depth [pc]
D_cm=D*3.086e+18    #cloud depth [cm]
n=1 #density [cm^-3]

##Part 1:
N=n*D_cm   #column density [cm^-2]
#find cross section: tau=n*sigma*s
tau=np.array((1e-3, 1, 1e+3))   #optical depth given in problem statement [unitless]
sigma=tau / (n*D_cm)    #cross section [cm^-2]
print("""PHYS239 HW2:
      Part 1:
      Column Density= {0} cm^-2
      Cross Section (sigma_nu):
      \t a) {1}
      \t b) {2}
      \t c) {3}
      """.format(N, sigma[0], sigma[1], sigma[2]))

##Part 2:
#define function to calculate intensity after passing through medium at one frequency
def intensity_one_freq(sigma, I_0, S, n, D):  #takes cross-section, initial intensity, source function, density [cm^-3] and size of region [pc] as inputs
    ds=np.arange(0,D,0.1)    #steps through medium
    I_nu=np.zeros(len(ds))  #create array of zeros the same size as the distance array to be filled via radiative transfer equation
    I_nu[0]=I_0
    #loop through each distance step and calculate intensity from radiative transfer equation
    tau=sigma*n*(ds*3.086e+18)    #calculate optical depths at each step from inputs
    #calculate intensity at each step
    for i in range(1,len(ds)):
        I_nu[i]=I_0*np.exp(-tau[i])+S*(1-np.exp(-tau[i]))
    return I_nu[len(I_nu)-1]

#calculate intensity at s=D
I_D=intensity_one_freq(3.24e-21, 10, 100, 1, 100)
print("""Part 2:
      sigma=3.24e-21 cm^-2, I(0)=10, S_nu=100, n=1cm^-3, D=100pc:
      I(s=D) = {0}
      """.format(I_D))

##Part 3:
#define function to convert from FWHM to variance
def Gamma2sigma(Gamma):
    return Gamma/ ( np.sqrt(2 * np.log(2)) * 2 )
#define function to create cross section as function of frequency
def cross_section(freq_range, fwhm, height):    #takes frequency range and peak frequency as inputs
    x=freq_range
    x0=x[len(x)/2]
    return height * np.exp( - 0.5*((x-x0)/Gamma2sigma(fwhm))**2 )

#generate plots of cross-section for peak values in part 1
frequency=np.arange(10,100,1)
sigma_nu=[cross_section(frequency, 1, sigma[i]) for i in range(len(sigma))]
pl.figure("Cross-sections as function of frequency")
pl.plot(frequency, sigma_nu[0], 'r--', label='$\\sigma_{\\nu,0}=3.24\\times10^{-24} cm^{-2}$')
pl.plot(frequency, sigma_nu[1], 'b--', label='$\\sigma_{\\nu,0}=3.24\\times10^{-21} cm^{-2}$')
pl.plot(frequency, sigma_nu[2], 'k--', label='$\\sigma_{\\nu,0}=3.24\\times10^{-18} cm^{-2}$')
pl.xlabel('Frequency [arbitrary units]', fontsize=15)
pl.ylabel('Cross section [$cm^{-2}$]', fontsize=15)
pl.title('$\\sigma_{\\nu}: \\nu_0=55, FWHM=1$')
pl.yscale('log')
pl.legend(loc='upper right', numpoints=1, ncol=1, fontsize=8, edgecolor='k')   #create legend
#pl.show()
