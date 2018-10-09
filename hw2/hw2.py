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
#write function to calculate cross-section peak
def sigma_peak(tau):  #given optical depth
    return tau / (n*D_cm)    #cross section [cm^-2]
#calculate sigma for tau given in problem statement
sigma=[sigma_peak(tau[i]) for i in range(len(tau))]
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
I_D=intensity_one_freq(3.24e-21, 10, 100, n, D)
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
sigma_nu=[cross_section(frequency, 15, sigma[i]) for i in range(len(sigma))]
pl.figure("Cross-sections as function of frequency")
pl.suptitle('$\\sigma_{\\nu}: \\nu_0=55, FWHM=15$')
pl.subplot(1,3,1)
pl.plot(frequency, sigma_nu[0], 'r--')
pl.title('$\\sigma_{\\nu,0}=3.24\\times10^{-24} cm^{-2}$')
pl.ylabel('Cross section [$cm^{-2}$]', fontsize=15)
pl.subplot(1,3,2)
pl.plot(frequency, sigma_nu[1], 'b--')
pl.title('$\\sigma_{\\nu,0}=3.24\\times10^{-21} cm^{-2}$')
pl.xlabel('Frequency [arbitrary units]', fontsize=15)
pl.subplot(1,3,3)
pl.plot(frequency, sigma_nu[2], 'k--')
pl.title('$\\sigma_{\\nu,0}=3.24\\times10^{-18} cm^{-2}$')
#pl.yscale('log')
pl.legend(loc='upper right', numpoints=1, ncol=1, fontsize=8, edgecolor='k')   #create legend
#pl.show()

#Part 4:
#use functions above to re-produce plots from handout
#part a
sigma_nu_a=cross_section(frequency,15, sigma_peak(1e1000))
I_D_nu_a=np.zeros(len(sigma_nu_a))
for i in range(len(sigma_nu_a)):
    I_D_nu_a[i]=intensity_one_freq(sigma_nu_a[i],10,10,n,D)
#part b
sigma_nu_b=cross_section(frequency,15, sigma_peak(0.5))
I_D_nu_b=np.zeros(len(sigma_nu_b))
for i in range(len(sigma_nu_b)):
    I_D_nu_b[i]=intensity_one_freq(sigma_nu_b[i],0,10,n,D)
#part c
sigma_nu_c=cross_section(frequency,15, sigma_peak(0.5))
I_D_nu_c=np.zeros(len(sigma_nu_c))
for i in range(len(sigma_nu_c)):
    I_D_nu_c[i]=intensity_one_freq(sigma_nu_c[i],1,10,n,D)
#part d
sigma_nu_d=cross_section(frequency,15, sigma_peak(0.5))
I_D_nu_d=np.zeros(len(sigma_nu_d))
for i in range(len(sigma_nu_d)):
    I_D_nu_d[i]=intensity_one_freq(sigma_nu_d[i],10,1,n,D)
#part e: superposition of optically thin and thick components
sigma_nu_e=cross_section(frequency,15, sigma_peak(0.5)) + cross_section(frequency,15,sigma_peak(10))
I_D_nu_e=np.zeros(len(sigma_nu_e))
for i in range(len(sigma_nu_e)):
    I_D_nu_e[i]=intensity_one_freq(sigma_nu_e[i],1,10,n,D)
#part f: superposition of optically thin and thick components
sigma_nu_f=cross_section(frequency,15, sigma_peak(0.5)) + cross_section(frequency,15,sigma_peak(10))
I_D_nu_f=np.zeros(len(sigma_nu_f))
for i in range(len(sigma_nu_f)):
    I_D_nu_f[i]=intensity_one_freq(sigma_nu_f[i],10,1,n,D)

#figure of all parts
pl.figure('Reproducing Handout Figures')
#a
pl.subplot(2,3,1)
pl.title('a) $\\tau_{\\nu}(D)>>1$')
pl.plot(frequency,I_D_nu_a,'k-')
pl.hlines(10, frequency[0], frequency[len(frequency)-1], colors='b', linestyles='dashed')
pl.ylabel('$I_{\\nu}$')
pl.xlabel('$\\nu$')
#b
pl.subplot(2,3,2)
pl.title('b) $I_{\\nu}(0)=0, \\tau_{\\nu}<1$')
pl.plot(frequency, I_D_nu_b, 'k-')
pl.hlines(10, frequency[0], frequency[len(frequency)-1], colors='b', linestyles='dashed')
pl.hlines(0, frequency[0], frequency[len(frequency)-1], colors='g', linestyles='dashed')
pl.ylabel('$I_{\\nu}$')
pl.xlabel('$\\nu$')
#c
pl.subplot(2,3,3)
pl.title('c) $I_{\\nu}(0)<S_{\\nu}, \\tau_{\\nu}<1$')
pl.plot(frequency, I_D_nu_c, 'k-')
pl.hlines(10, frequency[0], frequency[len(frequency)-1], colors='b', linestyles='dashed')
pl.hlines(1, frequency[0], frequency[len(frequency)-1], colors='g', linestyles='dashed')
pl.ylabel('$I_{\\nu}$')
pl.xlabel('$\\nu$')
#d
pl.subplot(2,3,4)
pl.title('d) $I_{\\nu}(0)>S_{\\nu}, \\tau_{\\nu}<1$')
pl.plot(frequency, I_D_nu_d, 'k-')
pl.hlines(1, frequency[0], frequency[len(frequency)-1], colors='b', linestyles='dashed')
pl.hlines(10, frequency[0], frequency[len(frequency)-1], colors='g', linestyles='dashed')
pl.ylabel('$I_{\\nu}$')
pl.xlabel('$\\nu$')
#e
pl.subplot(2,3,5)
pl.title('e) $I_{\\nu}(0)<S_{\\nu}, \\tau_{\\nu}<1, \\tau_{\\nu,0}>1$')
pl.plot(frequency, I_D_nu_e, 'k-')
pl.hlines(10, frequency[0], frequency[len(frequency)-1], colors='b', linestyles='dashed')
pl.hlines(1, frequency[0], frequency[len(frequency)-1], colors='g', linestyles='dashed')
pl.ylabel('$I_{\\nu}$')
pl.xlabel('$\\nu$')
#f
pl.subplot(2,3,6)
pl.title('f) $I_{\\nu}(0)>S_{\\nu}, \\tau_{\\nu}<1, \\tau_{\\nu,0}>1$')
pl.plot(frequency, I_D_nu_f, 'k-')
pl.hlines(1, frequency[0], frequency[len(frequency)-1], colors='b', linestyles='dashed', label='$S_{\\nu}$')
pl.hlines(10, frequency[0], frequency[len(frequency)-1], colors='g', linestyles='dashed', label='$I_{\\nu}(0)$')
pl.ylabel('$I_{\\nu}$')
pl.xlabel('$\\nu$')
pl.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
pl.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.)
pl.show()

