import numpy as np

def solve_y_momentum(ux, uy, P, ayo, ayw, aye, ays, ayn,
                     rho, mu, dx, dy, Gaus_it_max, conv_P, alpha):
    """
    Résolution de l'équation de quantité de mouvement en y (uy_star).
    """
    imax, jmax = P.shape
    uy_star = uy.copy()
    Gaus_it = 1
    # ----------------------------------------------------
    #Link coefficients
    # ----------------------------------------------------

    #Bulk
    uxw = 0.5 * (ux[1:-2, :-1] + ux[1:-2, 1:])
    uxe = 0.5 * (ux[2:-1, :-1] + ux[2:-1, 1:])
    uyn = 0.5 * (uy[1:-1, 1:-1] + uy[1:-1, 2:])
    uys = 0.5 * (uy[1:-1, :-2] + uy[1:-1, 1:-1])

    ayo[1:-1, 1:-1] = rho*dy*((np.abs(uxw)-uxw)/2 + (np.abs(uxe)+uxe)/2) + rho*dx*((np.abs(uys)-uys)/2 + (np.abs(uyn)+uyn)/2) + 2*mu*dy/dx + 2*mu*dx/dy

    ayw[1:-1, 1:-1] = -rho*dy*((np.abs(uxw)+uxw)/2) - mu*dy/dx
    aye[1:-1, 1:-1] = -rho*dy*((np.abs(uxe)-uxe)/2) - mu*dy/dx
    ays[1:-1, 1:-1] = -rho*dx*((np.abs(uys)+uys)/2) - mu*dx/dy
    ayn[1:-1, 1:-1] = -rho*dx*((np.abs(uyn)-uyn)/2) - mu*dx/dy

    #Left boundary
    uxw = 0.5 * (ux[0, :-1] + ux[0, 1:])
    uxe = 0.5 * (ux[1, :-1] + ux[1, 1:])
    uyn = 0.5 * (uy[0, 1:-1] + uy[0, 2:])
    uys = 0.5 * (uy[0, :-2] + uy[0, 1:-1])

    ayo[0, 1:-1] = rho*dy*((np.abs(uxw)-uxw)/2 + (np.abs(uxe)+uxe)/2) + rho*dx*((np.abs(uys)-uys)/2 + (np.abs(uyn)+uyn)/2) + 3*mu*dy/dx + 2*mu*dx/dy

    ayw[0, 1:-1] = 0
    aye[0, 1:-1] = -rho*dy*((np.abs(uxe)-uxe)/2) - mu*dy/dx
    ays[0, 1:-1] = -rho*dx*((np.abs(uys)+uys)/2) - mu*dx/dy
    ayn[0, 1:-1] = -rho*dx*((np.abs(uyn)-uyn)/2) - mu*dx/dy
 
    # ----------------------------------------------------
    # Solver
    # ----------------------------------------------------

    uy_star_b = uy_star.copy()
    erreur = conv_P * 3
    while (Gaus_it < Gaus_it_max and erreur > conv_P) or Gaus_it < 10:
            #Bulk
            uy_star_b[1:-1, 1:-1] = ((P[1:-1, :-1]-P[1:-1, 1:])*dx - ayw[1:-1, 1:-1]*uy_star[:-2, 1:-1] - aye[1:-1, 1:-1]*uy_star[2:, 1:-1] - ays[1:-1, 1:-1]*uy_star[1:-1, :-2] - ayn[1:-1, 1:-1]*uy_star[1:-1, 2:])/ayo[1:-1, 1:-1]

            #Left
            uy_star_b[0, 1:-1] = ((P[0, :-1]-P[0, 1:])*dx - aye[0, 1:-1]*uy_star[1, 1:-1] - ays[0, 1:-1]*uy_star[0, :-2] - ayn[0, 1:-1]*uy_star[0, 2:])/ayo[0, 1:-1]

            #Bottom
            uy_star_b[:, 0] = 0

            #Top
            uy_star_b[:, -1] = 0

            #Right
            uy_star_b[-1, :] = uy_star_b[-2, :]

            erreur = np.max(np.abs(uy_star_b - uy_star)) / max(np.max(np.abs(uy_star)), 1e-10)
            uy_star = uy_star_b.copy()
            Gaus_it += 1

    print(f"The uy error converged in {Gaus_it} iterations")

    # ----------------------------------------------------
    # Relaxation finale
    # ----------------------------------------------------
    uy_star = alpha * uy_star + (1 - alpha) * uy

    return uy_star


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

    # Pressure
    P = np.zeros((imax, jmax))

    # Staggered velocities
    ux = np.ones((imax+1, jmax))
    uy = np.zeros((imax, jmax+1))

    # Boundary conditions for uy
    uy[:, 0] = 0
    uy[:, -1] = 0

    # Coefficients
    ayo = np.zeros_like(uy)
    ayw = np.zeros_like(uy)
    aye = np.zeros_like(uy)
    ays = np.zeros_like(uy)
    ayn = np.zeros_like(uy)

    uy_star = solve_y_momentum(
        ux, uy, P,
        ayo, ayw, aye, ays, ayn,
        rho, mu, dx, dy,
        Gaus_it_max, conv_P, alpha
    )

    print("Shape P :", P.shape)
    print("Shape ux:", ux.shape)
    print("Shape uy:", uy.shape)

    print("\nuy initial:")
    print(uy)

    print("\nuy_star:")
    print(uy_star)

    print("\nMaximum uy =", np.max(np.abs(uy_star)))

    assert uy_star.shape == uy.shape
    assert np.all(np.isfinite(uy_star))
    assert np.allclose(uy_star[:, 0], 0)
    assert np.allclose(uy_star[:, -1], 0)
    assert np.max(np.abs(uy_star)) < 1e-10

    print("\nTEST PASSED")