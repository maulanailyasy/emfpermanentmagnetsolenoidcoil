# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 10:03:05 2024

@author: Ilyas
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import ScalarFormatter

# Defining symmbol for B(z)
z = sp.symbols('z')
L, R, mu_0, M = sp.symbols('L R mu_0 M')

# Parameters
L_val = 0.1  # length of magnet (m)
R_val = 0.015  # radius of magnet (m)
mu_0_val = 4*np.pi*1e-7  # vacuum permeability (T·m/A)
M_val = 1e6  # magnetization (A/m)
Br = 1.25
N = 100
A = math.pi*(R_val**2)

# Equation for  B(z)
term1 = (z-L/2)/sp.sqrt(R**2 + (z-L/2)**2)
term2 = (z+L/2)/sp.sqrt(R**2 + (z+L/2)**2)
B_z = (Br*R*L)*(term2 - term1)

# Calculate the derivative of dB(z)/dz
dB_dz = sp.diff(B_z, z)

# Change the derivative of dB/dz into numerical function 
dB_dz_func = sp.lambdify((z, L, R), dB_dz, modules='numpy')

# Parameters for damped-oscillation
m = 0.35  # mass of magnet (kg)
k = 5     # spring constant (N/m)
g = 9.81  # gravitational acceleration (m/s^2)
b_underdamped = 0.14  # under damping coefficient
C1 = 0.1
C2 = 0.2
b_critical = 2*np.sqrt(k*m)  # critical damping coefficient 
b_overdamped = 4.0  # over damping coefficient

# A function to calculate z(t) and dz(t)/dt for underdamped, critically damped, and over-damped
def z_underdamped(t, C1, C2, b, m, k, g):
    omega_d = np.sqrt(k/m - (b/(2*m))**2)
    return np.exp(-b/(2*m)*t)*(C1*np.cos(omega_d*t) + C2*np.sin(omega_d*t))

def dz_underdamped(t, C1, C2, b, m, k, g):
    omega_d = np.sqrt(k/m - (b/(2*m))**2)
    exp_term = np.exp(-b/(2*m)*t)
    return exp_term*(-b/(2*m)*(C1*np.cos(omega_d*t) + C2*np.sin(omega_d*t)) + omega_d*(C2*np.cos(omega_d*t) - C1*np.sin(omega_d*t)))

def z_critical(t, C1, C2, b, m, k, g):
    return (C1 + C2*t) * np.exp(-b/(2*m)*t) + (m*g/k)

def dz_critical(t, C1, C2, b, m, k, g):
    return (C2 - b/(2*m) * (C1 + C2 * t)) * np.exp(-b/(2*m) * t)

def z_overdamped(t, C1, C2, b, m, k, g):
    r1 = (-b/m + np.sqrt((b/m)**2 - 4 * k/m)) / 2
    r2 = (-b/m - np.sqrt((b/m)**2 - 4 * k/m)) / 2
    return C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t) + (m * g / k)

def dz_overdamped(t, C1, C2, b, m, k, g):
    r1 = (-b/m + np.sqrt((b/m)**2 - 4 * k/m)) / 2
    r2 = (-b/m - np.sqrt((b/m)**2 - 4 * k/m)) / 2
    return C1 * r1 * np.exp(r1 * t) + C2 * r2 * np.exp(r2 * t)

# Time interval and position z(t) (from -L/2 ke L/2)
t = np.linspace(0, 10, 5000)
z_valss = z_underdamped(t, C1, C2, b_underdamped, m, k, g)
z_critical_valss = z_critical(t, C1, C2, b_critical, m, k, g)
z_overdamped_valss = z_overdamped(t, C1, C2, b_overdamped, m, k, g)

# Limits from -L/2 to L/2
z_vals = np.clip(z_valss, -2*L_val, 2*L_val) #for default
zcritical = np.clip(z_critical_valss, -2*L_val, 2*L_val) #for default
zover = np.clip(z_overdamped_valss, -2*L_val, 2*L_val) #for default
#z_vals = np.clip(z_valss, -L_val/2, L_val/2) for default

# Calculate dz(t)/dt for each time t
dz_dt_vals = dz_underdamped(t, C1, C2, b_underdamped, m, k, g)
dz_critical_vals = dz_critical(t, C1, C2, b_critical, m, k, g)
dz_overdamped_vals = dz_overdamped(t, C1, C2, b_overdamped, m, k, g)

# Calculate dB/dz for each z
dB_dz_vals = dB_dz_func(z_vals, L_val, R_val) #under-damped
dBcritical = dB_dz_func(zcritical, L_val, R_val) #critical-damped
dBover = dB_dz_func(zover, L_val, R_val) #over-damped

# Dot product dB/dz*dz/dt
result_vals = (N*A)*(dB_dz_vals*dz_dt_vals)

#result for critically damped
critical = dBcritical*dz_critical_vals
#result for over damped
over = dBover*dz_overdamped_vals

# Plot the result with respect to position z
plt.figure(figsize=(12, 8))
#plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(z_vals, result_vals, color='b', linewidth=2)
plt.xlabel('z position (m)', fontsize=17)
plt.ylabel(r'Induced e.m.f (Volt)', fontsize=17)
plt.title('The induced e.m.f as function of position along the axial axis', fontsize=17)
plt.legend()
plt.xlim([-(3*L_val)/2, (3*L_val)/2])  
#plt.ylim([min(B_vals), max(B_vals)])
#plt.grid(True)

# Set the display of the plot
plt.tick_params(axis='both', which='major', labelsize=13, direction='in', length=8)  
plt.tick_params(axis='both', which='minor', length=4)
plt.xticks(fontsize=16)  
plt.yticks(fontsize=16)  

# Set the linewidth
plt.gca().spines['top'].set_linewidth(1.5)    
plt.gca().spines['bottom'].set_linewidth(1.5) 
plt.gca().spines['left'].set_linewidth(1.5)   
plt.gca().spines['right'].set_linewidth(1.5)

formatter = ScalarFormatter()
formatter.set_powerlimits((-3, 3))  # set the limit for scientific notation
plt.gca().yaxis.set_major_formatter(formatter)

plt.subplot(2, 1, 2)
plt.plot(t, result_vals, color='b', linewidth=2)
plt.xlabel('Time (s)', fontsize=18)
plt.ylabel(r'Induced e.m.f (Volt)', fontsize=18)
plt.title('The induced e.m.f as function of time', fontsize=17)
plt.legend()
plt.xlim([0, 10])  
#plt.ylim([min(B_vals), max(B_vals)])
#plt.grid(True)

# Set the display
plt.tick_params(axis='both', which='major', labelsize=13, direction='in', length=8)  
plt.tick_params(axis='both', which='minor', length=4)
plt.xticks(fontsize=16)  
plt.yticks(fontsize=16)  

# Set the linewidth
plt.gca().spines['top'].set_linewidth(1.5)    
plt.gca().spines['bottom'].set_linewidth(1.5) 
plt.gca().spines['left'].set_linewidth(1.5)   
plt.gca().spines['right'].set_linewidth(1.5)

formatter = ScalarFormatter()
formatter.set_powerlimits((-3, 3))  
plt.gca().yaxis.set_major_formatter(formatter)

plt.tight_layout()
#plt.savefig('emf.pdf')  
plt.show()