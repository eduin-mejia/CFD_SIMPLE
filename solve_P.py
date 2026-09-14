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
    # Bottom
    #Top
    #Left

    # Solver

    erreur_P = 1.0
    inner_it = 1
    Delta_P_b = np.zeros_like(Delta_P)

    while (inner_it < Gaus_it_max and erreur_P > conv_P) or inner_it < 10:
        
            #Bulk
            #Bottom
            #Top
            #Left
   

            # Calcul erreur relative
            diff = np.abs(Delta_P_b - Delta_P)
            max_denominator = np.maximum(np.abs(Delta_P_b), 1e-10)
            erreur_P = np.max(diff / max_denominator)

            Delta_P = Delta_P_b.copy()
            inner_it += 1

    print(f"The P error is {erreur_P:.4e}, converged in {inner_it} iterations, max delta_P= {np.max(np.max(max_denominator)):.4e} Pa")
    return Delta_P, apo, ape, aps, apn
