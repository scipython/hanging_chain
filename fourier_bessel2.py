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

# The exact function of z, x(z,0): linear from (0,d) to (c,0) then zero
# from (c,0) to (L,0).
d, c = 0.05, L/4
p, q = -d/c, d
def f(u):
    z = u**2 * g / 4
    # We have to cater z as a scalar and as an array since u could be passed as
    # either and we want to use boolean indexing to set h = 0 for z > c.
    if np.isscalar(z):
        if z > c:
            return 0
        return p*z + q
    h = p*z + q
    h[z>c] = 0
    return h

# jn_zeros calculates the first n zeros of the zero-th order Bessel function
# of the first kind
max_zeros = 10
w = jn_zeros(0, max_zeros)
b = 2 * np.sqrt(L/g)

def func(u, n):
    """The integrand for the Fourier-Bessel coefficient A_n."""
    return u * f(u) * j0(w[n-1] / b * u)

# Calculate the Fourier-Bessel coefficients using numerical integration
A = []
for n in range(1, max_zeros+1):
    An = quad(func, 0, b, args=(n,))[0] * 2 / (b * j1(w[n-1]))**2
    A.append(An)

# Create a labelled plot comparing f(z) with the Fourier-Bessel series
# approximation truncated at 2, 3 and 5 terms.
plt.plot(f(u), z, 'k', lw=2, label='$f(z)$')
for nmax in (3,5,10):
    ffit = np.zeros(N)
    for n in range(nmax):
        ffit += A[n-1] * j0(w[n-1] / b * u)
    plt.plot(ffit, z, label=r'$n_\mathrm{max} = '+'{}$'.format(nmax))
plt.xlim(-d/5,d)
plt.xlabel(r'$x$')
plt.ylabel(r'$z$')
plt.legend()

plt.savefig('fourier_bessel2.png')
plt.show()
