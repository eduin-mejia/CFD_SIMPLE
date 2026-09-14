import numpy as np

def correct_uy(uy, uy_star, Delta_P, ayo, dx, relax_p):
    """
    Correct the y-velocity field using the pressure correction Delta_P.
    """

    imax, jmax = Delta_P.shape

    #Bulk
    uy[:, 1:-1] = uy_star[:, 1:-1] - relax_p*dx*(Delta_P[:, 1:] - Delta_P[:, :-1])/ayo[:, 1:-1]

    #BC
    uy[:, 0] = 0
    uy[:, -1] = 0
    uy[-1, :] = uy[-2, :]

    return uy