import numpy as np

def solve_x_momentum(ux, uy, P, axo, axw, axe, axs, axn, rho, mu, dx, dy, Gaus_it_max, conv_P, alpha, left_velocity):
    """
    Solve the x-momentum equation using Gauss-Seidel or Jacobi.
    """
    imax, jmax = P.shape
    ux_star = ux.copy()

    # ---------------------------------------
    # Compute link coefficients
    # ---------------------------------------
    
    #Advice : This can be done either in a loop or with a vectorized form
    
    #Bulk
    uxw = 0.5 * (ux[:-2, 1:-1] + ux[1:-1, 1:-1])
    uxe = 0.5 * (ux[1:-1, 1:-1] + ux[2:, 1:-1])
    uyn = 0.5 * (uy[:-1, 2:-1] + uy[1:, 2:-1])
    uys = 0.5 * (uy[:-1, 1:-2] + uy[1:, 1:-2])

    axo[1:-1, 1:-1] = rho*dy*((np.abs(uxw)-uxw)/2 + (np.abs(uxe)+uxe)/2) + rho*dx*((np.abs(uys)-uys)/2 + (np.abs(uyn)+uyn)/2) + 2*mu*dy/dx + 2*mu*dx/dy
    axw[1:-1, 1:-1] = -rho*dy*((np.abs(uxw)+uxw)/2) - mu*dy/dx
    axe[1:-1, 1:-1] = -rho*dy*((np.abs(uxe)-uxe)/2) - mu*dy/dx
    axs[1:-1, 1:-1] = -rho*dx*((np.abs(uys)+uys)/2) - mu*dx/dy
    axn[1:-1, 1:-1] = -rho*dx*((np.abs(uyn)-uyn)/2) - mu*dx/dy

    # Bottom boundary
    uxw = 0.5 * (ux[:-2, 0] + ux[1:-1, 0])
    uxe = 0.5 * (ux[1:-1, 0] + ux[2:, 0])
    uyn = 0.5 * (uy[:-1, 1] + uy[1:, 1])

    axo[1:-1, 0] = rho*dy*((np.abs(uxw)-uxw)/2 + (np.abs(uxe)+uxe)/2) + rho*dx*((np.abs(uyn)+uyn)/2) + 2*mu*dy/dx + 3*mu*dx/dy
    axw[1:-1, 0] = -rho*dy*((np.abs(uxw)+uxw)/2) - mu*dy/dx
    axe[1:-1, 0] = -rho*dy*((np.abs(uxe)-uxe)/2) - mu*dy/dx
    axs[1:-1, 0] = 0
    axn[1:-1, 0] = -rho*dx*((np.abs(uyn)-uyn)/2) - mu*dx/dy

    # Top boundary
    uxw = 0.5 * (ux[:-2, -1] + ux[1:-1, -1])
    uxe = 0.5 * (ux[1:-1, -1] + ux[2:, -1])
    uys = 0.5 * (uy[:-1, -2] + uy[1:, -2])

    axo[1:-1, -1] = rho*dy*((np.abs(uxw)-uxw)/2 + (np.abs(uxe)+uxe)/2) + rho*dx*((np.abs(uys)-uys)/2) + 2*mu*dy/dx + 3*mu*dx/dy
    axw[1:-1, -1] = -rho*dy*((np.abs(uxw)+uxw)/2) - mu*dy/dx
    axe[1:-1, -1] = -rho*dy*((np.abs(uxe)-uxe)/2) - mu*dy/dx
    axs[1:-1, -1] = -rho*dx*((np.abs(uys)+uys)/2) - mu*dx/dy
    axn[1:-1, -1] = 0

    # ---------------------------------------
    # Solver loop
    # ---------------------------------------
    Gaus_it = 1
    ux_star_b = ux_star.copy()
    erreur = conv_P * 3

    while (Gaus_it < Gaus_it_max and erreur > conv_P) or Gaus_it < 10:
            #Bulk
            ux_star_b[1:-1, 1:-1] = ((P[:-1, 1:-1]-P[1:, 1:-1])*dy - axw[1:-1, 1:-1]*ux_star[:-2, 1:-1] - axe[1:-1, 1:-1]*ux_star[2:, 1:-1] - axs[1:-1, 1:-1]*ux_star[1:-1, :-2] - axn[1:-1, 1:-1]*ux_star[1:-1, 2:])/axo[1:-1, 1:-1]

            #Bottom
            ux_star_b[1:-1, 0] = ((P[:-1, 0]-P[1:, 0])*dy - axw[1:-1, 0]*ux_star[:-2, 0] - axe[1:-1, 0]*ux_star[2:, 0] - axn[1:-1, 0]*ux_star[1:-1, 1])/axo[1:-1, 0]

            #Tops
            ux_star_b[1:-1, -1] = ((P[:-1, -1]-P[1:, -1])*dy - axw[1:-1, -1]*ux_star[:-2, -1] - axe[1:-1, -1]*ux_star[2:, -1] - axs[1:-1, -1]*ux_star[1:-1, -2])/axo[1:-1, -1]

            #Left
            ux_star_b[0, :] = left_velocity

            #Right
            ux_star_b[imax, :] = ux_star_b[imax-1, :]

            erreur = np.max(np.abs(ux_star_b - ux_star)) / max(np.max(np.abs(ux_star)), 1e-14)
            ux_star = ux_star_b.copy()
            Gaus_it += 1

    print(f"The ux error converged in {Gaus_it} iterations")

    # Final relaxation
    ux_star = alpha * ux_star + (1. - alpha) * ux
    return ux_star


