import numpy as np

def solve_p(ux_star, uy_star, axo, ayo, imax, jmax, dx, dy, Gaus_it_max, conv_P, left_velocity):
    

    # Initialisation
    Delta_P = np.zeros((imax, jmax))
    Delta_P_b = np.zeros_like(Delta_P)

    # Link coeff
    apw = np.zeros_like(Delta_P)
    ape = np.zeros_like(Delta_P)
    aps = np.zeros_like(Delta_P)
    apn = np.zeros_like(Delta_P)
    apo = np.zeros_like(Delta_P)

    #Bulk
    apo[1:-1, 1:-1] = dy**2*(1/axo[1:-2, 1:-1] + 1/axo[2:-1, 1:-1]) + dx**2*(1/ayo[1:-1, 1:-2] + 1/ayo[1:-1, 2:-1])
    apw[1:-1, 1:-1] = -dy**2/axo[1:-2, 1:-1]
    ape[1:-1, 1:-1] = -dy**2/axo[2:-1, 1:-1]
    aps[1:-1, 1:-1] = -dx**2/ayo[1:-1, 1:-2]
    apn[1:-1, 1:-1] = -dx**2/ayo[1:-1, 2:-1]

    # Bottom
    apo[1:-1, 0] = dy**2*(1/axo[1:-2, 0] + 1/axo[2:-1, 0]) + dx**2/ayo[1:-1, 1]
    apw[1:-1, 0] = -dy**2/axo[1:-2, 0]
    ape[1:-1, 0] = -dy**2/axo[2:-1, 0]
    aps[1:-1, 0] = 0
    apn[1:-1, 0] = -dx**2/ayo[1:-1, 1]

    #Top
    apo[1:-1, -1] = dy**2*(1/axo[1:-2, -1] + 1/axo[2:-1, -1]) + dx**2/ayo[1:-1, -2]
    apw[1:-1, -1] = -dy**2/axo[1:-2, -1]
    ape[1:-1, -1] = -dy**2/axo[2:-1, -1]
    aps[1:-1, -1] = -dx**2/ayo[1:-1, -2]
    apn[1:-1, -1] = 0

    #Left
    apo[0, 1:-1] = dy**2/axo[1, 1:-1] + dx**2*(1/ayo[0, 1:-2] + 1/ayo[0, 2:-1])
    apw[0, 1:-1] = 0
    ape[0, 1:-1] = -dy**2/axo[1, 1:-1]
    aps[0, 1:-1] = -dx**2/ayo[0, 1:-2]
    apn[0, 1:-1] = -dx**2/ayo[0, 2:-1]

    #Bottom-left corner
    apo[0, 0] = dy**2/axo[1, 0] + dx**2/ayo[0, 1]
    apw[0, 0] = 0
    ape[0, 0] = -dy**2/axo[1, 0]
    aps[0, 0] = 0
    apn[0, 0] = -dx**2/ayo[0, 1]

    #Top-left corner
    apo[0, -1] = dy**2/axo[1, -1] + dx**2/ayo[0, -2]
    apw[0, -1] = 0
    ape[0, -1] = -dy**2/axo[1, -1]
    aps[0, -1] = -dx**2/ayo[0, -2]
    apn[0, -1] = 0

    # Solver

    erreur_P = 1.0
    inner_it = 1
    Delta_P_b = np.zeros_like(Delta_P)

    while (inner_it < Gaus_it_max and erreur_P > conv_P) or inner_it < 10:
        
            #Bulk
            Delta_P_b[1:-1, 1:-1] = (-(ux_star[2:-1, 1:-1]-ux_star[1:-2, 1:-1])*dy - (uy_star[1:-1, 2:-1]-uy_star[1:-1, 1:-2])*dx - apw[1:-1, 1:-1]*Delta_P[:-2, 1:-1] - ape[1:-1, 1:-1]*Delta_P[2:, 1:-1] - aps[1:-1, 1:-1]*Delta_P[1:-1, :-2] - apn[1:-1, 1:-1]*Delta_P[1:-1, 2:])/apo[1:-1, 1:-1]

            #Bottom
            Delta_P_b[1:-1, 0] = (-(ux_star[2:-1, 0]-ux_star[1:-2, 0])*dy - (uy_star[1:-1, 1]-uy_star[1:-1, 0])*dx - apw[1:-1, 0]*Delta_P[:-2, 0] - ape[1:-1, 0]*Delta_P[2:, 0] - apn[1:-1, 0]*Delta_P[1:-1, 1])/apo[1:-1, 0]

            #Top
            Delta_P_b[1:-1, -1] = (-(ux_star[2:-1, -1]-ux_star[1:-2, -1])*dy - (uy_star[1:-1, -1]-uy_star[1:-1, -2])*dx - apw[1:-1, -1]*Delta_P[:-2, -1] - ape[1:-1, -1]*Delta_P[2:, -1] - aps[1:-1, -1]*Delta_P[1:-1, -2])/apo[1:-1, -1]

            #Left
            Delta_P_b[0, 1:-1] = (-(ux_star[1, 1:-1]-left_velocity)*dy - (uy_star[0, 2:-1]-uy_star[0, 1:-2])*dx - ape[0, 1:-1]*Delta_P[1, 1:-1] - aps[0, 1:-1]*Delta_P[0, :-2] - apn[0, 1:-1]*Delta_P[0, 2:])/apo[0, 1:-1]

            #Bottom-left corner
            Delta_P_b[0, 0] = (-(ux_star[1, 0]-left_velocity)*dy - (uy_star[0, 1]-uy_star[0, 0])*dx - ape[0, 0]*Delta_P[1, 0] - apn[0, 0]*Delta_P[0, 1])/apo[0, 0]

            #Top-left corner
            Delta_P_b[0, -1] = (-(ux_star[1, -1]-left_velocity)*dy - (uy_star[0, -1]-uy_star[0, -2])*dx - ape[0, -1]*Delta_P[1, -1] - aps[0, -1]*Delta_P[0, -2])/apo[0, -1]

            #Right
            Delta_P_b[-1, :] = 0
   

            # Calcul erreur relative
            diff = np.abs(Delta_P_b - Delta_P)
            max_denominator = np.maximum(np.abs(Delta_P_b), 1e-10)
            erreur_P = np.max(diff / max_denominator)

            Delta_P = Delta_P_b.copy()
            inner_it += 1

    print(f"The P error is {erreur_P:.4e}, converged in {inner_it} iterations, max delta_P= {np.max(np.max(max_denominator)):.4e} Pa")
    return Delta_P, apo, ape, aps, apn


if __name__ == "__main__":

    # -------------------------
    # TEST
    # -------------------------
    imax = 5
    jmax = 6

    dx = 0.1
    dy = 0.1

    Gaus_it_max = 100
    conv_P = 1e-6
    left_velocity = 1.0

    # Staggered velocities
    ux_star = np.ones((imax+1, jmax))
    uy_star = np.zeros((imax, jmax+1))

    # Momentum diagonal coefficients
    axo = np.ones_like(ux_star)
    ayo = np.ones_like(uy_star)

    Delta_P, apo, ape, aps, apn = solve_p(
        ux_star, uy_star,
        axo, ayo,
        imax, jmax,
        dx, dy,
        Gaus_it_max, conv_P,
        left_velocity
    )

    print("\nShape ux_star:", ux_star.shape)
    print("Shape uy_star:", uy_star.shape)
    print("Shape Delta_P:", Delta_P.shape)

    print("\nDelta_P:")
    print(Delta_P)

    print("\nMaximum Delta_P =", np.max(np.abs(Delta_P)))

    assert Delta_P.shape == (imax, jmax)
    assert np.all(np.isfinite(Delta_P))
    assert np.max(np.abs(Delta_P)) < 1e-10
    assert np.allclose(Delta_P[-1, :], 0)

    print("\nTEST PASSED")