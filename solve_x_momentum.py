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
    print(ux_star)
    return ux_star


if __name__ == "__main__":

    # -------------------------
    # TEST
    # -------------------------
    imax = 5
    jmax = 6

    rho = 1.0
    mu = 0.01
    dx = 0.1
    dy = 0.1

    Gaus_it_max = 100
    conv_P = 1e-6
    alpha = 0.7
    left_velocity = 1.0

    # Campo de presión
# Campo de presión
    P = np.zeros((imax, jmax))

# Test inicial simple
    ux = np.ones((imax+1, jmax))
    uy = np.ones((imax, jmax+1))

# Boundary conditions
    ux[0, :] = left_velocity
    ux[-1, :] = ux[-2, :]
    uy[:, 0] = 0
    uy[:, -1] = 0

# Coeficientes
    axo = np.zeros_like(ux)
    axw = np.zeros_like(ux)
    axe = np.zeros_like(ux)
    axs = np.zeros_like(ux)
    axn = np.zeros_like(ux)
    solve_x_momentum(
        ux, uy, P,
        axo, axw, axe, axs, axn,
        rho, mu, dx, dy,
        Gaus_it_max, conv_P,
        alpha, left_velocity
    )