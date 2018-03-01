import numpy as np
from scipy.special import j0, j1, jn_zeros
from scipy.integrate import quad
from matplotlib import rc
import matplotlib.pyplot as plt

# Use LaTeX throughout the figure for consistency
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 16})
rc('text', usetex=True)

# Acceleration due to gravity, m.s-2
g = 9.81
# Chain length, m
L = 1
# Vertical axis (m)
N = 201
z = np.linspace(0, L, N)
# Scaled vertical axis
u = 2 * np.sqrt(z/g)

# The exact function of z, x(z,0): linear from (0,d) to (L,0).
d, c = 0.05, L
p, q = -d/c, d
def f(u):
    z = u**2 * g / 4
    return p*z + q

nmax = 10
# jn_zeros calculates the first n zeros of the zero-th order Bessel function
# of the first kind
w = jn_zeros(0, nmax)
b = 2 * np.sqrt(L/g)

def func(u, n):
    """The integrand for the Fourier-Bessel coefficient A_n."""
    return u * f(u) * j0(w[n-1] / b * u)

# Calculate the Fourier-Bessel coefficients using numerical integration
A = []
for n in range(1, nmax+1):
    An = quad(func, 0, b, args=(n,))[0] * 2 / (b * j1(w[n-1]))**2
    A.append(An)

# Create a labelled plot comparing f(z) with the Fourier-Bessel series
# approximation truncated at 2, 3 and 5 terms.
plt.plot(f(u), z, 'k', lw=2, label='$f(z)$')
for nmax in (2,3,5):
    ffit = np.zeros(N)
    for n in range(nmax):
        ffit += A[n-1] * j0(w[n-1] / b * u)
    plt.plot(ffit, z, label=r'$n_\mathrm{max} = '+'{}$'.format(nmax))
plt.xlim(0,d)
plt.xlabel(r'$x$')
plt.ylabel(r'$z$')
plt.legend()

plt.savefig('fourier_bessel.png')
plt.show()
