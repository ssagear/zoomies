
def calc_jz(gaia_table, write=False, fname=None):

    """
    Calculates Jz using Gala for a star with a row of Gaia data.

    Parameters
    ----------
    gaia_table:
        A row of Gaia data.
    write: boolean
        Save Gaia data table with Jz column?
    fname: str
        If write = True, filename to save data.
    
    """

    import astropy.coordinates as coord
    import astropy.table as at
    import astropy.units as u
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import binned_statistic_2d

    # gala
    import gala.coordinates as gc
    import gala.dynamics as gd
    import gala.potential as gp
    from gala.units import galactic
    from gala.dynamics.actionangle.tests.staeckel_helpers import galpy_find_actions_staeckel
    from pyia import GaiaData


    g = GaiaData(gaia_table)
    mw = gp.MilkyWayPotential()
    c = g.get_skycoord()
    galcen = c.transform_to(coord.Galactocentric(galcen_v_sun=[8, 254, 8] * u.km / u.s, galcen_distance=8.275 * u.kpc))
    w = gd.PhaseSpacePosition(galcen.data)
    aaf = galpy_find_actions_staeckel(mw, w)
    Jz = aaf['actions'][:, 2]
    Jphi = aaf['actions'][:, 1]
    Jtheta = aaf['actions'][:, 0]

    gaia_table['Jz'] = Jz
    gaia_table['Jphi'] = Jphi
    gaia_table['Jtheta'] = Jtheta

    if write==False:
        return gaia_table

    elif write==True:
        gaia_table.write(fname, overwrite=True)