import numpy as np
import os
from scipy.stats import norm
import scipy.optimize as optimization
print(os.getcwd())
#os.chdir('OneDrive/Pulpit/Studia/Magisterka/Stochastyczna Matematyka finansowa/Lab/Projekt 1')



def S_t(r, S_0, t, m):
    return([S_0*np.exp(r*(t/m)) for t in np.arange(0, t*m)])

def Ri(S):
    return([np.log(S[i]/S[i+1]) for i in np.arange(0,np.size(S)-1)])

def Mu(Ri, dt = 1/255):
    return(np.sum(Ri)/(np.size(Ri)*dt))

def Sigma(Ri, dt = 1/255):
    return(np.sum((Ri-np.mean(Ri))**2)/((np.size(Ri)-1)*dt))

def BS(strike,s0,mu,sigma,r,t,T,dt = 1/255):
    d1 = (np.log(s0/strike) + (r + 0.5*sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    d2 = d1 - sigma*np.sqrt(T-t)
    return(s0 * norm.cdf(d1) - strike * np.exp(-r) * norm.cdf(d2))

def Opti(Ri, N):
    return(optimization.curve_fit(a*x + b, N, Ri, kwargs))

def normal_dist(N, M, rho):
    return([N,rho * N + np.sqrt(1 - rho**2)*M])

def f(N,a,b):
    return(a*N + b)


#Otwarcie,Najwyzszy,Najnizszy,Zamkniecie,Wolumen
kghm = np.genfromtxt('kghm_28.03.2017_31.12.2018.csv', delimiter=",", skip_header = 1)
kghm = kghm[:,1:]
wig20 = np.genfromtxt('wig20_28.03.2017_31.12.2018.csv', delimiter=",", skip_header = 1)
wig20 = wig20[:,1:]

N1 , N2 = normal_dist(np.random.normal(0,1,np.size(Ri(wig20[:,3]))),
                    np.random.normal(0,1,np.size(Ri(wig20[:,3]))),
                    0.1)

np.set_printoptions(suppress=True)
print(optimization.curve_fit(f, N1, Ri(wig20[:,3]))[0])





