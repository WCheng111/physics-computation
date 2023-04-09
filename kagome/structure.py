from __future__ import division  # so that 1/2 == 0.5, and not 0
from math import pi, sqrt, tanh

import kwant

# For computing eigenvalues
import scipy.sparse.linalg as sla

# For plotting
from matplotlib import pyplot


# Define the graphene lattice
sin_30, cos_30 = (1 / 2, sqrt(3) / 2)
kagome = kwant.lattice.general([(1, 0), (sin_30, cos_30)],
                                 [(0, 0), (1/4, sqrt(3)/4), (-1/4, sqrt(3)/4)])
a, b, c= kagome.sublattices
def make_system(r=5, w=2.0, pot=2):

    #### Define the scattering region. ####
    # circular scattering region
    def circle(pos):
        x, y = pos
        return x ** 2 < r ** 2 and y**2<r**2

    sys = kwant.Builder()

    # w: width and pot: potential maximum of the p-n junction
    def potential(site):
        (x, y) = site.pos
        d = y * cos_30 + x * sin_30
        return pot * tanh(d / w)

    sys[kagome.shape(circle, (0, 0))] = potential

    # specify the hoppings of the graphene lattice in the
    # format expected by builder.HoppingKind
    hoppings = (((0, 0), a, b), ((0,0),a,c),((0, 1), a, b), ((-1, 1), a, c),((0,0),b,c),((-1, 0), b, c))
    sys[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings]] = -1


    sym0 = kwant.TranslationalSymmetry(kagome.vec((-1, 0)))
    lead0 = kwant.Builder(sym0)
    lead0[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings]] = -1

    return sys,lead0
def plot_bandstructure(flead, momenta):
    bands = kwant.physics.Bands(flead)
    energies = [bands(k) for k in momenta]

    pyplot.figure()
    pyplot.plot(momenta, energies)
    pyplot.xlabel("momentum [(lattice constant)^-1]")
    pyplot.ylabel("energy [t]")
    pyplot.show()

def main():
    pot = 2
    sys = make_system(pot=pot)
    def family_colors(site):
         if site.family == a:
             return 0
         if site.family ==b:
             return 1/2
         if site.family ==c:
              return 1

    syst,lead0=make_system()
    kwant.plot(syst, site_color=family_colors, site_lw=0.01, site_size=0.2,colorbar=False)

    # lead0=lead0.finalized()
    # kwant.plotter.bands(lead0, show=False)
    # pyplot.xlabel("momentum [(lattice constant)^-1]")
    # pyplot.ylabel("energy [t]")
    # pyplot.show()
    
    lead0 = lead0.finalized()
    momenta = [-pi + 0.02 * pi * i for i in range(101)]
    plot_bandstructure(lead0, momenta)









if __name__ == '__main__':
    main()