'''
    Maren Cosens
    PHYS 239: Radiative Processes in Astrophysics
    HW #2
    hw2.py: radiative transfer problems
    using Python 2.7
'''
import numpy as np

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
