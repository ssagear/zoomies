
def calc_jz(gaia_table, method="agama", mwmodel="2022", write=False, fname=None):

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

    print('Starting Jz calculation...')

    import astropy.coordinates as coord
    import astropy.table as at
    import astropy.units as u
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    # gala
    import gala.coordinates as gc
    import gala.dynamics as gd
    import gala.potential as gp
    from gala.units import galactic
    # from gala.dynamics.actionangle.tests.staeckel_helpers import galpy_find_actions_staeckel
    from pyia import GaiaData

    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='True'


    g = GaiaData(gaia_table)

    if mwmodel == '2022':
        mw = gp.MilkyWayPotential2022()

    else:
        print('Custom potential')
        #mw = gp.MilkyWayPotential()
        mw = mwmodel
    
    c = g.get_skycoord()
    galcen = c.transform_to(coord.Galactocentric(galcen_v_sun=[8, 254, 8] * u.km / u.s, galcen_distance=8.275 * u.kpc))
    w = gd.PhaseSpacePosition(galcen.data)

    if method=='galpy':

        print('Calculating actions with galpy...')

        aaf = galpy_find_actions_staeckel(mw, w)
        Jz = aaf['actions'][:, 2]
        Jphi = aaf['actions'][:, 1]
        Jr = aaf['actions'][:, 0]

    elif method=='agama':

        print('Calculating actions with agama...')

        # import agama

        # agama.setUnits(mass=u.Msun, length=u.kpc, time=u.Myr)
        # agama_pot = mw.as_interop("agama")
        # af = agama.ActionFinder(agama_pot)
        # Jr, Jz, Jphi = af(w.w(galactic).T).T * 1000 # agama units are different from galpy
        Jr, Jz, Jphi = 1,2,3
        
    else:
        print('you have to pick galpy or agama')

    gaia_table['Jz'] = Jz
    gaia_table['Jphi'] = Jphi
    gaia_table['Jr'] = Jr

    if write==False:
        return gaia_table

    elif write==True:
        gaia_table.write(fname, overwrite=True)
