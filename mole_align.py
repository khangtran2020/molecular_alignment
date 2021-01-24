import math
import scipy.constants as sc
from tqdm import tqdm
import numpy as np
from scipy.special import factorial as fac
import scipy
import cmath
from scipy.special import sph_harm
import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns
from scipy import interp

""" READ DATA"""

f = open('input.json', 'r')
data = json.load(f)

# Parameters
pi = np.pi
auT = 0.024189
pB = 219474.63068
pAlpha = 0.14818474347690475
c0 = scipy.constants.c
kB = 69.50348004
h = scipy.constants.h

# Laser
I = data['laser']['intensity']
omega0 = data['laser']['omega']
t_fwhm = data['laser']['fwhm'] / auT
laser_step = data['laser']['step']

# Molecules
type_of_mol = data['molecules']['type']
name_of_mol = data['molecules']['name']
B = data['molecules']['B'] / pB
alpha_para = data['molecules']['alpha_para'] / pAlpha
alpha_pedi = data['molecules']['alpha_pedi'] / pAlpha
jMax = data['molecules']['jMax']
mMax = data['molecules']['mMax']

# Free field process
T = data['temperature']
t_start = data['t_start']/auT
t_end = data['t_end']/auT
t_step = data['t_step']
angle_list = data['angle_list']
number_of_point = data['number_of_point']
number_of_step = data['number_of_step']

""" LASER """


def initialize_laser(I, omega0, t_fwhm, laser_step):
    time = []
    e = []
    ta = -2 * (t_fwhm)
    tb = 2 * t_fwhm
    t_step = laser_step
    delta_t = (tb - ta) / t_step
    Emax = math.sqrt(I / (3.5 * (10 ** 16)))
    for i in range(t_step):
        t = ta + i * delta_t
        time.append(t)
        e.append(Emax * np.exp(-2 * np.log(2) * ((t + delta_t / 2) ** 2 / (t_fwhm ** 2))) * math.cos(
            omega0 * (t + delta_t / 2)))
    time = np.array(time)
    e = np.array(e)
    return time, e, delta_t, ta, tb


""" INITIALIZE THE STAT """


def initialize_states(jMax, mMax):
    index = []
    c = []
    for j in range(jMax):
        if (mMax > j):
            for m in range(2 * j + 1):
                index.append((j, m - j))
        else:
            for m in range(2 * mMax + 1):
                index.append((j, m - mMax))
    nMax = np.array(index).shape[0]
    for k in range(nMax):
        temp = []
        for j in range(nMax):
            if (index[j][0] == index[k][0] and index[j][1] == index[k][1]):
                temp.append(1)
            else:
                temp.append(0)
        c.append(temp)
    c = np.array(c)
    return index, c, nMax


""" LEGENDRE PARAMETER"""
x, w = np.polynomial.legendre.leggauss(jMax)
phi_r = (2 * pi - 0) / 2
u_r = 1
phi_m = (2 * pi + 0) / 2
u_m = 0

theta_r = pi / 2
theta_m = pi / 2
theta = []
for i in range(jMax):
    theta.append(np.arccos(x[i]))
theta.reverse()

# wanted group
ntheta = len(angle_list)
angle_list_rad = np.array(angle_list)*pi/180

""" HELPER FUNCTION """


def a(j, m):
    return np.sqrt(((2 * j + 1) / (4 * pi)) * (fac(j - m) / fac(j + m)))


def ffi(m, mm):
    if (m == mm):
        return 1
    else:
        return 0


def ftheta_t(j, m, jj, mm, u, t, e, delta_t):
    if (j == jj or j == jj + 2 or j == jj - 2):
        return (scipy.special.lpmv(m, j, u) * scipy.special.lpmv(mm, jj, u) * (
            np.exp(-1j * ((-1) * (e[t] ** 2) * (alpha_para * u ** 2 + alpha_pedi * (1 - u ** 2)) / 2) * delta_t)))
    else:
        return 0


""" BOLTZMANN DISTRIBUTION """


def boltzmann_distribution(B, T, kB, nMax, index, pB):
    indx = []
    wb = []
    b = B * pB
    sum_exp = 0
    for j in range(nMax):
        if j % 2 == 0:
            sum_exp = sum_exp + np.exp(-1 * b * 100 * index[j][0] * (index[j][0] + 1) / (kB * T))
        else:
            sum_exp = sum_exp + 2 * np.exp(-1 * b * 100 * index[j][0] * (index[j][0] + 1) / (kB * T))
    for j in range(nMax):
        indx.append(j)
        if j % 2 == 0:
            wb.append(2 * np.exp(-1 * b * 100 * index[j][0] * (index[j][0] + 1) / (kB * T)) / sum_exp)
        else:
            wb.append(np.exp(-1 * b * 100 * index[j][0] * (index[j][0] + 1) / (kB * T)) / sum_exp)
    return indx, wb


index, C, nMax = initialize_states(jMax=jMax, mMax=mMax)
laser_time, E, delta_t, ta, tb = initialize_laser(I=I, omega0=omega0, t_fwhm=t_fwhm, laser_step=laser_step)
indx, wb = boltzmann_distribution(B=B, T=T, kB=kB, nMax=nMax, index=index, pB=pB)

""" PLOT LASER & BOLTZMANN DISTRIBUTION"""


def plot_laser_boltzdist(laser_time, E, indx, wb, auT):
    f, ax = plt.subplots(1, 2, figsize=(18, 8))
    time = laser_time * auT
    ax[0].plot(time, E, 'r')
    ax[0].set_title('Laser')
    ax[0].set_ylabel('E(t) (a.u)')
    ax[0].set_xlabel('time (fs)')
    ax[1].plot(indx, wb, 'g')
    ax[1].set_title('Boltzmann Distribution')
    ax[1].set_ylabel('Distribution')
    ax[1].set_xlabel('J')
    f.savefig("plot.png")


plot_laser_boltzdist(laser_time=laser_time, E=E, indx=indx, wb=wb, auT=auT)

""" SPLIT OPERATOR """

em = []
for i in range(nMax):
    em.append((cmath.exp(-1j * B * index[i][0] * (index[i][0] + 1) * delta_t / 2)))
em = np.array(em)

c_temp = []
for q in tqdm(range(nMax)):
    temp = C[q]
    for i in range(laser_step):
        t = ta + i * delta_t
        c_1 = em * temp
        c_1 = np.array(c_1)
        c_2 = []
        for m in range(nMax):
            sumt = 0j
            for n in range(nMax):
                if (index[m][1] == index[n][1]) and (
                        index[m][0] == index[n][0] or index[m][0] == (index[n][0] + 2) or index[m][0] == (
                        index[n][0] - 2)):
                    thet = 0
                    for k in range(jMax):
                        thet = thet + w[k] * ftheta_t(j=index[m][0], m=index[m][1], jj=index[n][0], mm=index[n][1],
                                                      u=x[k], t=i, e=E, delta_t=delta_t)
                    sumt = sumt + c_1[n] * a(j=index[m][0], m=index[m][1]) * a(j=index[n][0],
                                                                               m=index[n][1]) * thet * 2 * pi
                else:
                    sumt = sumt + 0
            c_2.append(sumt)
        c_2 = np.array(c_2)
        c_3 = em * c_2
        c_3 = np.array(c_3)
        temp = c_3
    c_temp.append(temp)

C = c_temp


""" FREE ROTATION """

delta_t = (t_end - t_start)/t_step
time = []
for i in range(t_step):
    time.append(t_start + i * delta_t)

ro = []
for i in tqdm(range(t_step)):
    t = t_start + i * delta_t
    ro_temp = []
    c_temp = []
    for m in range(nMax):
        c_1 = []
        for n in range(nMax):
            c_1.append((cmath.exp(-1j * B * index[n][0] * (index[n][0] + 1) * t) * C[m][n]))
        c_temp.append(c_1)
    for t in range(jMax):
        sumt = []
        for m in range(nMax):
            temp = 0j
            for n in range(nMax):
                temp = temp + c_temp[m][n] * sph_harm(index[n][1], index[n][0], 0, theta[t])
            sumt.append(temp)
        temp = 0j
        for i in range(nMax):
            temp = temp + (np.abs(sumt[i]) ** 2) * wb[i]
        ro_temp.append(temp.real)
    ro.append(ro_temp)

cos_theta = []
time = []

for i in tqdm(range(t_step)):
    time.append(t_start + i * delta_t)
    thet = 0
    for k in range(jMax):
        thet = thet + w[k] * ro[i][k] * ((u_r * x[k] + u_m) ** 2)
    cos_theta.append(thet * 2 * pi)

time = np.array(time)*auT/1000
df = pd.DataFrame({
    'time (ps)': time,
    'cos^2theta': cos_theta
})
df.to_csv("cos2theta.csv")

f2, ax2 = plt.subplots(1, 1, figsize=(18, 8))
ax2.plot(df['time (ps)'],df['cos^2theta'])
ax2.set_ylabel('degree of alignment')
ax2.set_xlabel('time (ps)')
ax2.set_title('COS^2 THETA')
f2.savefig('cos2theta.png')

df = pd.DataFrame({'angle': angle_list})

for k in range(number_of_point):
    ro_temp = ro[np.argmax(cos_theta) + k*number_of_step]
    res = interp(angle_list_rad, theta, ro_temp)
    df['point_'+str(k)] = res

df.to_csv("distribution.csv", index=False)
f3, ax3 = plt.subplots(1, 1, figsize=(18, 8))
sns.lineplot(x=range(jMax), y=ro[np.argmax(cos_theta)], ax=ax3)
sns.lineplot(x=range(ntheta), y=df['point_0'], ax=ax3)
f3.savefig('last.png')
