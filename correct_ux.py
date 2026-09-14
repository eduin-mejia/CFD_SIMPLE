import numpy as np

def correct_ux(ux, ux_star, Delta_P, axo, dy, relax_p, left_velocity):
    """
    Correct the x-velocity field using the pressure correction Delta_P.
    """

    #Bulk
    ux[1:-1, :] = ux_star[1:-1, :] - relax_p*dy*(Delta_P[1:, :] - Delta_P[:-1, :])/axo[1:-1, :]

    #BC
    ux[0, :] = left_velocity
    ux[-1, :] = ux[-2, :]

    return ux