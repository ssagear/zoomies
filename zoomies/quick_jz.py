def galpy_find_actions_staeckel(potential, w, mean=True, delta=None, ro=None, vo=None):
    """
    Function from a previous version of Gala! pre-2023

    Compute approximate actions, angles, and frequencies using the Staeckel
    Fudge as implemented in Galpy. If you use this function, please also cite
    Galpy in your work (Bovy 2015).

    Parameters
    ----------
    potential : potential-like
        A Gala potential instances.
    w : `~gala.dynamics.PhaseSpacePosition` or `~gala.dynamics.Orbit`
        Either a set of initial conditions / phase-space positions, or a set of
        orbits computed in the input potential.
    mean : bool (optional)
        If an `~gala.dynamics.Orbit` is passed in, take the mean over actions
        and frequencies.
    delta : numeric, array-like (optional)
        The focal length parameter, ∆, used by the Staeckel fudge. This is
        computed if not provided.
    ro : quantity-like (optional)
    vo : quantity-like (optional)

    Returns
    -------
    aaf : `astropy.table.QTable`
        An Astropy table containing the actions, angles, and frequencies for
        each input phase-space position or orbit.

    """

    import astropy.table as at
    import astropy.units as u
    from collections.abc import Iterable
    from gala.dynamics.actionangle import get_staeckel_fudge_delta
    from galpy.actionAngle import actionAngleStaeckel
    from gala.dynamics import Orbit


    delta = get_staeckel_fudge_delta(potential, w)
    galpy_potential = potential.as_interop("galpy")

    if isinstance(galpy_potential, list):
        ro = galpy_potential[0]._ro * u.kpc
        vo = galpy_potential[0]._vo * u.km / u.s
    else:
        ro = galpy_potential._ro * u.kpc
        vo = galpy_potential._vo * u.km / u.s


    if not isinstance(w, Orbit):
        w = Orbit(w.pos[None], w.vel[None], t=[0.0] * potential.units["time"])


        if w.norbits == 1:
            iter_ = [w]
        else:
            iter_ = w.orbit_gen()


    if isinstance(delta, u.Quantity):
        delta = np.atleast_1d(delta)

    if not isinstance(delta, Iterable):
        delta = [delta] * w.norbits

    if len(delta) != w.norbits:
        raise ValueError(
            "Input delta must have same shape as the inputted number of orbits"
        )
    
    rows = []

    for w_, delta_ in zip(iter_, delta):
        o = w_.to_galpy_orbit(ro, vo)
        aAS = actionAngleStaeckel(pot=galpy_potential, delta=delta_)

        aaf = aAS.actionsFreqsAngles(o)
        aaf = {
            "actions": np.array(aaf[:3]).T * ro * vo,
            "freqs": np.array(aaf[3:6]).T * vo / ro,
            "angles": coord.Angle(np.array(aaf[6:]).T * u.rad),
        }
        if mean:
            aaf["actions"] = np.nanmean(aaf["actions"], axis=0)
            aaf["freqs"] = np.nanmean(aaf["freqs"], axis=0)
            aaf["angles"] = aaf["angles"][0]
        rows.append(aaf)
    
    return at.QTable(rows=rows)


def calc_jz(gaia_table, method="galpy", mwmodel="2022", write=False, fname=None):

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
    import astropy.units as u

    # gala
    import gala.dynamics as gd
    import gala.potential as gp
    from gala.units import galactic
    from pyia import GaiaData

    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='True'


    g = GaiaData(gaia_table)
    

    if mwmodel == '2022':
        mw = gp.MilkyWayPotential2022()

    else:
        print('Custom potential')
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

        raise NotImplementedError('Action calculation with agama is not currently supported due to install issues.')

        import agama

        agama.setUnits(mass=u.Msun, length=u.kpc, time=u.Myr)
        agama_pot = mw.as_interop("agama")
        af = agama.ActionFinder(agama_pot)
        Jr, Jz, Jphi = af(w.w(galactic).T).T * 1000 # agama units are different from galpy
        
    else:
        print('you have to pick galpy or agama')

    gaia_table['Jz'] = Jz
    gaia_table['Jphi'] = Jphi
    gaia_table['Jr'] = Jr

    if write==False:
        return gaia_table

    elif write==True:
        gaia_table.write(fname, overwrite=True)