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
dx =2e-3
dy = 2e-3
mu = 1e-1
rho = 1e4
Re = 10
left_velocity = Re * mu /(dy*jmax*rho)
#left_velocity = 0.25000

h = dy * jmax
ux[:, :] = left_velocity

#***************************************************************
# Solver parameters
#***************************************************************

alpha = 0.3
relax_u = 0.5
relax_P =0.1
conv = 1e-5
conv_P = 1e-6
it_max = 50000
Gaus_it_max = 10
graph_error = np.zeros(it_max)
erreur = 1.

#***************************************************************
# Reynolds Number Output
#***************************************************************
#Re = dy * jmax * rho * left_velocity / mu
print(f"Le nombre de Reynolds = {Re:.2f}")

#***************************************************************
# Main SIMPLE loop
#***************************************************************

with open("Log.txt" ,"a+") as f:
    f.write(f"imax = {imax}, jmax = {jmax}\n")
    f.write(f"dx = {dx} , dy = {dy} \n") 
    f.write(f"length = {dx*imax}, width = {dy*jmax}\n" )
    f.write(f"left_velocity = {left_velocity}\n") 
    f.write(f"Le nombre de Reynolds = {Re}\n")
    f.close()

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
    it += 1

    if  not (it < it_max and erreur > conv):
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
        ux_theo = 4*1.5*left_velocity*y_vals/h*(1 - y_vals/h)
        plt.plot(ux_theo, ls='--', c='black', label='Theo')
        plt.title('Pressure(x) vs theoretical')
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


error_L2_ux = 100*np.sqrt(np.sum((ux[imax-5,:] - ux_theo)**2)*dy)/np.sqrt(np.sum(ux_theo**2)*dy)
if it == it_max:
    print("L'algorithme n'a pas convergé")
else:
    print(f"L'algorithme a convergé après {it} itérations") 
with open("Log.txt", "a+") as f: 
    f.write(f"Erreur L2 = {error_L2_ux}\n")
    f.write(f"Numero de iteraciones = {it}\n")
    f.write(f"erreur = {erreur}\n")
    f.write("\n")
print(f"El error L2 = {error_L2_ux}")

# Affichage final
#plt.show()
