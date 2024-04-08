import numpy as np
seed = 0
global_rng = np.random.default_rng(seed)
const_num_day_year = 100 #252

def genSBM(t, dt):
    dt = min(t,dt)
    n = t // dt + 2 - (t % dt == 0)
    arr = np.zeros(n)
    arr[1:] = global_rng.normal(size=n - 1)
    return arr

def genYearFraction(t, dt):
    dt = min(t,dt)
    n = t // dt + 2 - (t % dt == 0)
    arr = np.zeros(n)
    arr[1:] = dt / const_num_day_year
    if (t % dt):
        arr[-1] = (t % dt) / const_num_day_year
    return arr

class Asset:
    def __init__(self, S_0, mu, sigma):
        self.S_0 = S_0
        self.mu = mu
        self.sigma = sigma

    def gen_traj(self, t, dt): # t liczba dni do generowania, dt co ile generować
        self.SBM = genSBM(t, dt)
        self.YearFraction = genYearFraction(t, dt)
        self.GBM = self.S_0 * np.exp(np.cumsum(self.SBM * self.sigma * self.YearFraction + self.mu * self.YearFraction))

class CorrAssets:
    def __init__(self, asset1, asset2, rho):
        self.a1 = asset1
        self.a2 = asset2
        self.rho = rho

    def gen_traj(self, t ,dt):
        n = dt // t + 2 
        if (t % dt):
            n -= 1
        self.BM1 = genSBM(t, dt)
        self.BM2 = self.rho * self.BM1 + np.sqrt(1 - self.rho ** 2) * genSBM(t, dt)
        self.YearFraction = genYearFraction(t, dt)
        self.GBM1 = self.a1.S_0 * np.exp(np.cumsum(self.BM1 * self.a1.sigma * self.YearFraction + self.a1.mu * self.YearFraction))
        self.GBM2 = self.a2.S_0 * np.exp(np.cumsum(self.BM2 * self.a2.sigma * self.YearFraction + self.a2.mu * self.YearFraction))
        
