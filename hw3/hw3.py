'''
    Maren Cosens
    PHYS 239: Radiative Processes in Astrophysics
    HW #3
    hw3.py: Bremsstrahlung radiation problem
    using Python 2.7
    '''
import numpy as np
import matplotlib.pyplot as pl
import scipy.integrate as int

##Part 1: Define initial variables
#constants
eps_0=8.8542*10**(-12)    #[C^2/N/m^2]; permittivity of free space
a_0=5.29177e-11 #[m]; Bohr radius
c=3e9   #[m/s] speed of light
#electron parameters
x_0=-300*a_0 #[m] initial x and y positions are 1000*Bohr radius (R=sqrt(x^2+y^2)~283)
y_0=400*a_0   #[m]; electron initially up and left from ion
R_0=np.sqrt(x_0**2+y_0**2)
v_0_x=100000   #[m/s]; initial velocity in x-direction
v_0_y=0 #no initial velocity in the y-direction
Q_e=-1.602e-19   #[Coulombs]
m_e=9.11e-31    #[kg]
#ion parameters
Z=1     #hydrogen ions
Q_ion=Z*(-1*Q_e) #[Coulombs]; Z*e
#position, velocity, acceleration and time arrays
tau=R_0/v_0_x   #interaction timescale extimate
t=np.arange(0,tau,1e-15) #[s]; time steps
x=np.zeros(len(t))  #empty array for x position
x[0]=x_0    #populate first cell with initial position
y=np.zeros(len(t))  #empyt array for y position
y[0]=y_0    #populate first cell with initial position
v_x=np.zeros(len(t))    #empty array for x velocity
v_x[0]=v_0_x    # populate first cell with initial velocity
v_y=np.zeros(len(t))    #empty array for y velocity
v_y[0]=v_0_y    #populate first cell with initial velocity
a_x=np.zeros(len(t))    #array of acceleration in x
a_y=np.zeros(len(t))    #array of acceleration in y

##Part 2: calclate force between electron and ion at each time intercal to fill in position, velocity, and acceleration arrays
for i in range(1,len(t)):
    R=np.sqrt(x[i-1]**2+y[i-1]**2)  #radius
    theta=np.arctan(x[i-1]/y[i-1])  #angle between x and y positions
    del_t=t[i]-t[i-1]   #time interval
    F= (1/(4*np.pi*eps_0))*(Q_ion*Q_e)/R**2     #force at that seperation
    a_x[i] =(F/m_e)*np.sin(theta)   #acceleration in the x-direction
    a_y[i]=(F/m_e)*np.cos(theta)    #acceleration in y-direction
    v_x[i]=v_x[i-1] +(a_x[i]*del_t) #previous velocity + change in velocity
    v_y[i]=v_y[i-1] +(a_y[i]*del_t)
    x[i]=x[i-1]+ (np.mean((v_x[i],v_x[i-1]))*del_t) #previous position + change in position
    y[i]=y[i-1]+ (np.mean((v_y[i],v_y[i-1]))*del_t)


##Part 3: Plot results of interaction
pl.figure('Part 3')
#position
pl.subplot(1,3,1)
pl.plot(x/a_0,y/a_0, '-b', label='electron path')
pl.xlabel('x position [$a_0^{-1}$]')
pl.ylabel('y position [$a_0^{-1}$]')
pl.legend()
#velocity
pl.subplot(1,3,2)
pl.title('$x_0=300a_0, y_0=400a_0, v_{x,0}=10^5 m \ s^{-1}$')
pl.plot(t, v_x, '-b', label='velocity in x')
pl.plot(t, v_y, '-k', label='velocity in y')
pl.xlabel('time [s]')
pl.ylabel('velocity [$m \ s^{-1}$]')
pl.legend()
#acceleration
pl.subplot(1,3,3)
pl.plot(t, a_x, '-b', label='acceleration in x')
pl.plot(t, a_y, '-k', label='accelleration in y')
pl.xlabel('time [s]')
pl.ylabel('acceleration [$m \ s^{-2}$]')
pl.legend()
#pl.show()

##Part 4: Fourier transform acceleration to get power spectrum as a function of frequency
#array of frequencies
w=np.arange(0,1600,1600.0/float(len(t)))   #angular frequency
nu=w/(2*np.pi)    #frequency
#convert to polar coordinates
R=np.sqrt(x**2 +y**2)
theta=np.arctan(x/y)
#acceleration in polar
a_r=(a_x*np.sin(theta) + a_y*np.cos(theta))
a_theta=(-1*a_x*np.cos(theta)+a_y*np.sin(theta))
#dipole moment
#w2d_r=(Q_e/(2*np.pi()*R))*int.trapz(
#Fourier transform each acceleration array
a_r_w=np.fft.fft(a_r)   #a_r(w)
a_theta_w=np.fft.fft(a_theta)   #a_theta(w)
#dW/(dOmege*Domega)
dW_dO_dw_r = ((w**2 * np.sin(theta)**2) *Q_e /(2*(np.pi**2)*c**3)) *a_r_w**2
dW_dO_dw_theta = ((w**2 * np.sin(theta)**2) *Q_e /(2*(np.pi**2)*c**3)) *a_theta_w**2
#dW/(dOmega*domega)
dW_dO_dnu_r = ((2*nu**2 * np.sin(theta)**2) *Q_e /(c**3)) *a_r_w**2
dW_dO_dnu_theta = ((2*nu**2 * np.sin(theta)**2) *Q_e /(c**3)) *a_theta_w**2

pl.figure('power spectrum')
#radial direction
pl.subplot(2,1,1)
pl.plot(nu,dW_dO_dnu_r)
pl.yscale('log')
pl.xlabel('frequency [Hz]')
pl.ylabel('Power/steradian')
#theta direction
pl.subplot(2,1,2)
pl.plot(nu, dW_dO_dnu_theta)
pl.yscale('log')
pl.xlabel('frequency [Hz]')
pl.ylabel('Power/steradian')
pl.show()

