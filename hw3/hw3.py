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
from scipy import signal

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
t=np.arange(0,tau, tau/300) #[s]; 300 time steps
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
'''
    #this attempt doesn't look quite right
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
'''
#array of frequencies
#w=np.arange(0,1600,1600.0/float(len(t)))   #angular frequency
nu=np.arange(0,1000,1000.0/float(len(t)))    #frequency
#convert to polar coordinates
R=np.sqrt(x**2 +y**2)
theta=np.arctan(x/y)
#acceleration in polar
a_r=(a_x*np.sin(theta) + a_y*np.cos(theta))
a_theta=(-1*a_x*np.cos(theta)+a_y*np.sin(theta))
#Fourier transform each acceleration array; python transforms int o frequency in Hz not angular frequency?
a_r_nu=np.fft.fft(a_r)   #a_r(nu)
a_theta_nu=np.fft.fft(a_theta)   #a_theta(nu)
'''
    used to check radial and angular components only
a_x_nu=np.fft.fft(a_x)  #a_x(nu)
a_y_nu=np.fft.fft(a_y)  #a_y(nu)
'''
#radial acceleration would be only component leaving region with emission towards observer
dW_dnu=(((Q_e**2)/2*np.pi)*(a_r_nu*a_r_nu.conjugate()) *(4/3)) #integrated over solid angle (integral of sin^3(x)dx from 0 to pi = 4/3)
dW_dnu=dW_dnu/100   #convert m to cm for cgs units
#inflection point
grad=np.gradient(np.gradient(dW_dnu)*nu)*nu #second derivative with frequency in log space
inf=signal.argrelmax(np.abs(grad)) #local minima (1,2) shape array
inds=inf[0]    #indices of inflection point
inflection_point=inds[0]    #first inflection poins
#plot spectrum
pl.figure('radial power spectrum')
pl.plot(nu[:-100], dW_dnu[:-100], label='spectrum')   #spectrum before a cut-off frequency
#pl.plot(nu, dW_dnu)
#pl.yscale('log')
pl.xscale('log')
pl.xlabel('Frequency [Hz]')
pl.ylabel('Power [erg s]')
pl.ylim((-1,5))
pl.vlines(nu[inflection_point], ymin=-1, ymax=5, linestyle='--', label='inflection point')
pl.legend()
pl.show()
'''
#plotted angular to check spectrum matches with radial shape
dW_dnu_theta=((Q_e**2)/2*np.pi)*a_theta_nu**2 *(4/3)
pl.figure('angular power spectrum')
pl.plot(nu, dW_dnu_theta)
pl.yscale('log')
pl.xlabel('Frequency [Hz]')
pl.ylabel('Power')  #check on units
#pl.show()
'''
#end
##Part 5: peak as a function of b and v_0
b=np.arange(100,700, (600/10))
b=b*a_0 #[m]array of impact parameters (y_0 in Part 1)
#use same initial x position
#v_0=np.arange(50000,200000, (200000-50000)/len(b))  #[m/s] array of initial velocities in the x-direction
v_0=np.full(len(b), 100000, dtype=float)
#peak_freq=np.zeros((len(b),len(v_0)))
v_0=np.concatenate((np.full(len(b),100000,dtype=float), np.full(len(b),125000,dtype=float), np.full(len(b),150000,dtype=float), np.full(len(b),175000,dtype=float), np.full(len(b),200000,dtype=float)))
b=np.concatenate((b,b,b,b,b))
peak_freq=np.zeros(len(b))
peak_loc=np.zeros(len(b))

for j in range(0,len(b)):   #loop through initial positions
    R_0=np.sqrt(b[j]**2+x_0**2)    #inital radius
        #for k in range(0,len(v_0)): #loop through initial velocities
    tau=R_0/v_0[j]
    t=np.arange(0,tau, tau/300) #[s] 300 time steps
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
    nu=np.arange(0,1000,1000.0/float(len(t)))    #frequency
    #convert to polar coordinates
    R=np.sqrt(x**2 +y**2)
    theta=np.arctan(x/y)
    #acceleration in polar
    a_r=(a_x*np.sin(theta) + a_y*np.cos(theta))
    a_theta=(-1*a_x*np.cos(theta)+a_y*np.sin(theta))
    #Fourier transform each acceleration array; python transforms int o frequency in Hz not angular frequency?
    a_r_nu=np.fft.fft(a_r)   #a_r(nu)
    #radial acceleration would be only component leaving region with emission towards observer?
    dW_dnu=(((Q_e**2)/2*np.pi)*(a_r_nu*a_r_nu.conjugate()) *(4/3)) #integrated over solid angle (integral of sin^3(x)dx from 0 to pi = 4/3)
#peak_loc[j]=np.argmax(dW_dnu[:-100])  #array element corresponding to peak of power spectrum ignoring peak at 0 frequency
#peak_loc[j]=np.argmin(np.gradient(np.gradient(dW_dnu[:-100])*nu[:-100])*nu[:-100])
    grad=np.gradient(np.gradient(dW_dnu)*nu)*nu #second derivative with frequency in log space
    infl=signal.argrelmax(np.abs(grad)) #inflection point (1,2) shape array
    inds=infl[0]    #indices of local minima
    peak_loc[j]=inds[0]    #inflection point
    peak_freq[j]=nu[peak_loc[j]]  #frequency of peak


#plot peak frequancy as a function of b and v
#B, V, P = np.meshgrid(b, v_0, peak_freq)
fig=pl.figure('peak_frequencies')
ax=fig.add_subplot(111,projection='3d')
#ax.plot_wireframe(b/a_0, v_0, peak_freq, color='lightgreen')
ax.scatter3D(b/a_0, v_0, peak_freq, c='g')
ax.set_xlabel('Impact Parameter [$a_0^{-1}$]')
ax.set_ylabel('Initial x Velocity [m/s]')
ax.set_zlabel('Peak Frequency [Hz]')
ax.view_init(45,-45)
#ax.view_init(0,90)
#ax.set_zscale('log')
pl.show()

