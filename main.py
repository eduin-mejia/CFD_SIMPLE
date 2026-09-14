import numpy as np
import matplotlib.pyplot as plt
from solve_x_momentum import solve_x_momentum
from solve_y_momentum import solve_y_momentum
from solve_P import solve_p
from correct_ux import correct_ux
from correct_uy import correct_uy
import os 

#***************************************************************
# Pre-processing
#***************************************************************
imax = 300
jmax = 30
ux = np.zeros((imax+1, jmax))
uy = np.zeros((imax, jmax+1))
ux_star = np.zeros_like(ux)
uy_star = np.zeros_like(uy)
P = np.zeros((imax, jmax))
Delta_P = np.zeros_like(P)

axo = np.zeros_like(ux)
axw = np.zeros_like(ux)
axe = np.zeros_like(ux)
axs = np.zeros_like(ux)
axn = np.zeros_like(ux)

ayo = np.zeros_like(uy)
ayw = np.zeros_like(uy)
aye = np.zeros_like(uy)
ays = np.zeros_like(uy)
ayn = np.zeros_like(uy)

ape = np.zeros_like(P)
aps = np.zeros_like(P)
apn = np.zeros_like(P)
apo = np.zeros_like(P)

#***************************************************************
# Physical parameters
#***************************************************************
dx = 2e-3
dy = 2e-3
mu = 1e-1
rho = 1e4
left_velocity = 0.01

h = dy * jmax
ux[:, :] = left_velocity

#***************************************************************
# Solver parameters
#***************************************************************

alpha = 1.
relax_u = 0.7
relax_P =0.3
conv = 1e-4
conv_P = 1e-2
it_max = 2000
Gaus_it_max = 10
graph_error = np.zeros(it_max)
erreur = 1.

#***************************************************************
# Reynolds Number Output
#***************************************************************
Re = dy * jmax * rho * left_velocity / mu
print(f"Le nombre de Reynolds = {Re:.2f}")

#***************************************************************
# Main SIMPLE loop
#***************************************************************
os.makedirs("./Results", exist_ok=True)
it = 1
while it < it_max and erreur > conv:
    print(f"Itération {it}")

    ux_star = solve_x_momentum(ux, uy, P, axo, axw, axe, axs, axn, rho, mu, dx, dy, Gaus_it_max, conv_P, alpha, left_velocity)
    uy_star = solve_y_momentum(ux, uy, P, ayo, ayw, aye, ays, ayn,rho, mu, dx, dy, Gaus_it_max, conv_P, alpha)
    Delta_P, apo, ape, aps, apn = solve_p(ux_star, uy_star, axo, ayo, imax, jmax, dx, dy, Gaus_it_max, conv_P, left_velocity)

    ux = ux*(1-relax_u)+relax_u*correct_ux(ux, ux_star, Delta_P, axo, dy, relax_P, left_velocity)
    uy = uy*(1-relax_u)+relax_u*correct_uy(uy, uy_star, Delta_P, ayo, dx, relax_P)

    P += relax_P * Delta_P
    erreur = np.max(np.abs(Delta_P)) / max(1e-12, np.max(np.abs(P)))
    graph_error[it] = erreur

    if it % 100 == 0:
        y_vals = (np.arange(1, jmax+1) - 0.5) * dy
        plt.figure(1)
        plt.plot(4*1.5*left_velocity*y_vals/h*(1 - y_vals/h),y_vals, ls='--',c='black',label='Theo')
        plt.plot(ux[imax-5, :], y_vals,'+',c='red',label='Simu')
        #plt.title('X velocity vs theoretical')
        plt.xlabel('ux (m/s)')
        plt.ylabel('y (m)')
        plt.legend()  
        plt.savefig(f'./Results/Poiseuille_profile_{it}.pdf')

        plt.figure(2)
        plt.contourf(np.abs(uy[:, :jmax].T) + np.abs(ux[:imax, :].T),cmap='cividis')
        plt.title(f'Velocity field it= {it}')
        plt.savefig(f'./Results/Velocity_field_{it}.pdf')
        

        plt.figure(3)
        plt.contourf(P.T,levels=20, cmap='cividis')
        plt.title(f'Pressure field it= {it}')
        plt.savefig(f'./Results/Pressure_field_{it}.pdf')

        plt.figure(4)
        x_vals = np.arange(1, imax+1) * dx
        plt.plot(x_vals[::5], P[::5, jmax//2],'+',c='red',label='Simu')
        plt.plot(x_vals, 8 * mu * 1.5 * left_velocity / (dy * jmax)**2 * (imax*dx - (np.arange(1, imax+1)-1)*dx), ls='--', c='black', label='Theo')
        #plt.title('Pressure(x) vs theoretical')
        plt.xlabel('x (m)')
        plt.ylabel('Pressure (Pa)')
        plt.legend()
        plt.title(f'Pressure profile it= {it}')
        plt.savefig(f'./Results/Pressure_profile_{it}.pdf')

        plt.figure(5)
        plt.plot(np.log10(graph_error[1:it]),'+',c='black')
        plt.title('error estimation')
        plt.xlabel('iteration')
        plt.ylabel('Log(error)')
        plt.savefig('./Results/Error.pdf')
        plt.show()

    it += 1

if it == it_max:
    print("L'algorithme n'a pas convergé")
else:
    print(f"L'algorithme a convergé après {it} itérations")

# Affichage final
plt.show()
