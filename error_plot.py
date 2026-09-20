import numpy as np
import matplotlib.pyplot as plt

Re = np.array([1, 10, 100, 200, 300, 400, 550, 555])

error_L2 = np.array([0.055729059700848356,
                     0.059580993211297284,
                     0.12928738960126546,
                     0.6935441533080101,
                     2.016055244399122,
                     3.551608980935338,
                     5.751637314222507,
                     5.8201042343576335])

iterations = np.array([3224,
                       3441,
                       4039,
                       5107,
                       5431,
                       7646,
                       15042,
                       15360])

Re_diverged = 565

#***************************************************************
# L2 error vs Reynolds
#***************************************************************

plt.figure(1)
plt.semilogx(Re, error_L2, 'o-', c='black')
plt.axvline(Re_diverged, ls='--', c='red', label='Divergence')
plt.xlabel('Reynolds number')
plt.ylabel('Relative L2 error (%)')
plt.title('Relative L2 velocity error vs Reynolds number')
plt.grid(True, which='both')
plt.legend()
plt.savefig('./Results/L2_vs_Reynolds.pdf')

#***************************************************************
# Number of iterations vs Reynolds
#***************************************************************

plt.figure(2)
plt.semilogx(Re, iterations, 'o-', c='black')
plt.axvline(Re_diverged, ls='--', c='red', label='Divergence')
plt.xlabel('Reynolds number')
plt.ylabel('Number of SIMPLE iterations')
plt.title('Number of iterations vs Reynolds number')
plt.grid(True, which='both')
plt.legend()
plt.savefig('./Results/Iterations_vs_Reynolds.pdf')

plt.show()