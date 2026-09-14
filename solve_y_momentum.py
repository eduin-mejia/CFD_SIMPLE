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
 
    # ----------------------------------------------------
    # Solver
    # ----------------------------------------------------

    uy_star_b = uy_star.copy()
    erreur = conv_P * 3
    while (Gaus_it < Gaus_it_max and erreur > conv_P) or Gaus_it < 10:
            #Bulk
            #Bottom
            #Top
            #Right


            erreur = np.max(np.abs(uy_star_b - uy_star)) / max(np.max(np.abs(uy_star)), 1e-10)
            uy_star = uy_star_b.copy()
            Gaus_it += 1

    print(f"The uy error converged in {Gaus_it} iterations")

    # ----------------------------------------------------
    # Relaxation finale
    # ----------------------------------------------------
    uy_star = alpha * uy_star + (1 - alpha) * uy

    return uy_star
