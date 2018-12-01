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
from mpl_toolkits import mplot3d

##Part 1: Define initial variables
#constants
eps_0=8.8542*10**(-12)    #[C^2/N/m^2]; permittivity of free space
a_0=5.29177e-11 #[m]; Bohr radius
c=3e9   #[m/s] speed of light
#electron parameters
x_0=(-3e5)*a_0  #[m] initial x and y positions
y_0=10000*a_0   #[m]; electron initially up and left from ion
R_0=np.sqrt(x_0**2+y_0**2)
v_0_x=1e6   #[m/s]; initial velocity in x-direction
v_0_y=0 #no initial velocity in the y-direction
Q_e=-1.602e-19   #[Coulombs]
m_e=9.11e-31    #[kg]
#ion parameters
Z=1     #hydrogen ions
Q_ion=Z*(-1*Q_e) #[Coulombs]; Z*e
#position, velocity, acceleration and time arrays
tau=R_0/v_0_x   #interaction timescale estimate
t=np.arange(0,2*tau, 2*tau/(1e+6)) #[s]; 10^4 time steps
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
pl.figure('position velocity acceleration')
#position
pl.subplot(2,2,1)
pl.title('$x_0=-3e5*a_0, y_0=10000a_0, v_{x,0}=10^6 m \ s^{-1}$')
pl.plot(x/a_0,y/a_0, '-b', label='electron path')
pl.xlabel('x position [$a_0^{-1}$]')
pl.ylabel('y position [$a_0^{-1}$]')
pl.legend()
#velocity
pl.subplot(2,2,3)
pl.plot(t, v_x, '-b', label='velocity in x')
pl.xlabel('time [s]')
pl.ylabel('velocity [$m \ s^{-1}$]')
pl.legend()
pl.subplot(2,2,4)
pl.plot(t, v_y, '-k', label='velocity in y')
pl.xlabel('time [s]')
pl.ylabel('velocity [$m \ s^{-1}$]')
pl.legend()
#acceleration
pl.subplot(2,2,2)
pl.plot(t[1:], a_x[1:], '-b', label='acceleration in x')
pl.plot(t[1:], a_y[1:], '-k', label='accelleration in y')
pl.xlabel('time [s]')
pl.ylabel('acceleration [$m \ s^{-2}$]')
pl.legend()
#pl.show()

##Part 4: Fourier transform acceleration to get power spectrum as a function of frequency
#array of frequencies
sampling_rate=len(t)/(tau)  #[Hz]
period=1/sampling_rate  #period
nu=np.linspace(0.0, 1.0/(2.0*period), np.int(len(t)/2)) #from 0 to 1/2 of the period

#Fourier transform each acceleration array; python transforms into frequency in Hz
# referred to https://ericstrong.org/fast-fourier-transforms-in-python/
a_x_nu=np.fft.fft(a_x)  #a_x(nu), with both + and - frequencies
a_x_nu_p=(2.0/len(t))*np.abs(a_x_nu[0:np.int(len(t)/2)])    #positive frequencies
a_y_nu=np.fft.fft(a_y)  #a_y(nu)
a_y_nu_p=(2.0/len(t))*np.abs(a_y_nu[0:np.int(len(t)/2)])    #positive frequencies
a_mag_nu_p=np.sqrt((a_x_nu_p**2)+(a_y_nu_p**2)) #magnitude

#polar coordinates for determining power integrated over solid angle
R=np.sqrt(x**2 +y**2)
theta=np.arctan(x/y)
a_r_nu=(a_x_nu*np.sin(theta) + a_y_nu*np.cos(theta))
a_r_nu_p=(2.0/len(nu))*np.abs(a_r_nu[0:np.int(len(t)/2)])  #positive frequencies
a_theta_nu=(-1*a_x_nu*np.cos(theta) + a_y_nu*np.sin(theta))
a_theta_nu_p=(2.0/len(t))*np.abs(a_theta_nu[0:np.int(len(t)/2)])  #positive frequencies

#power
dW_dnu=(((Q_e**2)/2*np.pi)*a_r_nu_p**2 *(4/3)) #integrated over solid angle (integral of sin^3(x)dx from 0 to pi = 4/3); radiation would be due to radial acceleration
dW_dnu=dW_dnu/100   #convert m to cm for cgs units

#plot spectrum
pl.figure('power spectrum')
pl.title('Spectrum for single electron with $x_0=-3e5*a_0, y_0=10000a_0, v_{x,0}=10^6 m \ s^{-1}$')
pl.plot(nu, dW_dnu, label='spectrum')   #spectrum before a cut-off frequency
#pl.yscale('log')
pl.xscale('log')
pl.xlabel('Frequency [Hz]')
pl.ylabel('Power [$erg \ s^{-1}$]')
#pl.show()

##Part 5: peak as a function of b and v_0
b=np.arange(y_0/2.0,2.0*y_0, (2*y_0 - y_0/2.0)/10)  #[m]array of impact parameters (y_0 in Part 1)
#use same initial x position
v_0=np.concatenate((np.full(len(b),0.8e6,dtype=float), np.full(len(b),0.9e6,dtype=float), np.full(len(b),1e6,dtype=float), np.full(len(b),1.1e6,dtype=float), np.full(len(b),1.2e6,dtype=float), np.full(len(b),1.3e6,dtype=float), np.full(len(b),1.4e6,dtype=float)))
b=np.concatenate((b,b,b,b,b,b,b))
peak_freq=np.zeros(len(b))
peak_loc=np.zeros(len(b))

for j in range(0,len(b)):   #loop through initial positions
    R_0=np.sqrt(b[j]**2+x_0**2)    #inital radius
        #for k in range(0,len(v_0)): #loop through initial velocities
    tau=R_0/v_0[j]
    t=np.arange(0,2*tau, tau/(1e+4)) #[s]; 10^4 time steps
    x=np.zeros(len(t))  #empty array for x position
    x[0]=x_0    #populate first cell with initial position
    y=np.zeros(len(t))  #empyt array for y position
    y[0]=b[j]    #populate first cell with initial position
    v_x=np.zeros(len(t))    #empty array for x velocity
    v_x[0]=v_0[j]    # populate first cell with initial velocity
    v_y=np.zeros(len(t))    #empty array for y velocity
    v_y[0]=v_0_y    #populate first cell with initial velocity
    a_x=np.zeros(len(t))    #array of acceleration in x
    a_y=np.zeros(len(t))    #array of acceleration in y
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
    sampling_rate=len(t)/(tau)  #[Hz]
    period=1/sampling_rate  #period
    nu=np.linspace(0.0, 1.0/(2.0*period), np.int(len(t)/2)) #from 0 to 1/2 of the period
    #magnitude of acceleration in fourier space
    a_x_nu=np.fft.fft(a_x)  #a_x(nu)
    a_y_nu=np.fft.fft(a_y)  #a_y(nu)
    a_r_nu2=(a_x_nu*np.sin(theta) + a_y_nu*np.cos(theta))    #convert to polar coords
    a_r_nu_p2=(2.0/len(nu))*np.abs(a_r_nu2[0:np.int(len(t)/2)])  #positive frequencies
    #power using radial component of acceleration (leaving region)
    dW_dnu2=(((Q_e**2)/2*np.pi)*a_r_nu_p2**2 *(4/3)) #integrated over solid angle (integral of sin^3(x)dx from 0 to pi = 4/3)
    peak_loc[j]=np.argmax(dW_dnu2)  #array element corresponding to peak of power spectrum
    peak_freq[j]=nu[np.int(peak_loc[j])]  #frequency of peak

#plot peak frequancy as a function of b and v
fig=pl.figure('peak_frequencies')
ax=fig.add_subplot(111,projection='3d')
ax.scatter3D(b/a_0, v_0, peak_freq, c='g')
ax.set_xlabel('Impact Parameter [$a_0^{-1}$]')
ax.set_ylabel('Initial x Velocity [m/s]')
ax.set_zlabel('Peak Frequency [Hz]')
ax.view_init(7,133)
#ax.view_init(15,-45)
#pl.show()

pl.figure('peak frequencies2D')
pl.subplot(1,2,1)
pl.plot(b/a_0,peak_freq, '.b')
pl.xlabel('Impact Parameter [$a_0^{-1}$]')
pl.ylabel('Peak Frequency [Hz]')
pl.subplot(1,2,2)
pl.plot(v_0, peak_freq, '.b')
pl.xlabel('Initial x Velocity [m/s]')
pl.show()

