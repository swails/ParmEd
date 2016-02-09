"""
This module contains a handful of extra custom integrator subclasses for OpenMM
simulations
"""
from __future__ import division, print_function, absolute_import

import math
from parmed.constants import GAS_CONSTANT
import parmed.unit as u
# Play some gymnastics to make this module importable when OpenMM is not
# available. It obviously won't run, though (and would be quite useless, anyway)
try:
    import simtk.openmm as mm
except ImportError:
    CustomIntegrator = object # pragma: no cover
else:
    CustomIntegrator = mm.CustomIntegrator

class SINRMTSIntegrator(CustomIntegrator):
    """ A Stochastic-Isokinetic Nose-Hoover chain RESPA integrator

    Parameters
    ----------
    dt : float
        Size of the inner time step (XO-RESPA)
    nrespa : int
        Number of time steps to run the fast forces each time step. A value of 1
        is *not* a multiple time-stepping integrator and will simply implement a
        stochastic isokinetic Nose-Hoover chain integrator
    temperature : float or Quantity
        The target temperature *of the configuration space sampling*.
    slowgroup : int
        The group number representing the slow forces evaluated only on the
        outer time steps
    fastgroup : int
        The group number representing the fast forces evaluated every step
    shortgroup : int
        The group number representing the short-range *nonbonded* forces only.
        Since the long-range nonbonded forces will include the short-range
        forces as well, this needs to be *subtracted* from the long-range
        forces before the RESPA impulse is added
    nauxvar : int, optional, default=1
        Number of auxiliary thermostat variables to use. Default is 1
    friction : float or Quantity, optional, default=5.0/picosecond
        The friction coefficient (in 1/ps) for the stochastic temperature
        control. Default is 5/ps
    tau : float, optional, default=24.0
        Time constant for defining mass of thermostat variables
    tolerance : float, optional, default=1.0e-5
        Threshhold below which the cosh and sinh terms are expanded as a
        4th-order Taylor series expansion rather than being computed outright.
        You should rarely (if ever) need to change this.
    calc_total_KE : bool, optional, default=False
        If True, the total kinetic energy, including the fictitious integrator
        particles, is computed. This is useful as an implementation check, since
        the constraints should keep this quantity exactly conserved.

    Notes
    -----
    The size of the outer time step becomes ``dt*nrespa``. If nrespa=1, the
    values of slowgroup, fastgroup, and shortgroup become irrelevant and are
    ignored.

    Note that the coordinates will come from a canonical distribution at this
    temperature, but the velocities (and therefore apparent temperature) will
    not be equal to this temperature. The temperature is guaranteed to not
    exceed the target temperature, and will vary based on the number of
    auxiliary thermostat variables you have. It is normal for the temperature to
    approach roughly 1/2 of the target temperature for a single auxiliary
    variable. For more information, and a more thorough explanation, see the
    citation:

    Leimkuhler, B.; Margul, D.; Tuckerman, M.; Mol. Phys., 2013, 111 p3579
    """

    NRES = 2 # currently hard-coded in a couple places. DO NOT CHANGE
    NSY = 3  # currently hard-coded in a couple places. DO NOT CHANGE

    def __init__(self, dt, nrespa, temperature, slowgroup, fastgroup,
                 shortgroup, nauxvar=1, friction=5.0/u.picosecond, tau=24.0,
                 tolerance=1e-5, calc_total_KE=False):
        if nrespa <= 0:
            raise ValueError('nrespa must be > 0. 1 means no MTS')
        if slowgroup < 0 or slowgroup > 31:
            raise ValueError('slowgroup must be between 0 and 31')
        if fastgroup < 0 or fastgroup > 31:
            raise ValueError('fastgroup must be between 0 and 31')
        self.nrespa = nrespa
        self.slowgroup = slowgroup
        self.fastgroup = fastgroup
        self.shortgroup = shortgroup
        self.L = nauxvar
        # Convert input to unitless numbers
        if u.is_quantity(dt):
            dt = dt.value_in_unit(u.picosecond)
        if u.is_quantity(friction):
            friction = friction.value_in_unit(u.picosecond**-1)
        if u.is_quantity(temperature):
            temperature = temperature.value_in_unit(u.kelvin)
        # Initialize our superclass
        CustomIntegrator.__init__(self, dt)
        # Add integrator global variables
        errtol = tolerance
        kbT = GAS_CONSTANT.value_in_unit(u.kilojoule_per_mole/u.kelvin) * \
                temperature
        Lbeta = self.L * kbT
        LLp1 = self.L/(self.L+1)
        Q2 = Q1 = kbT * tau * tau
        tQ2 = 2 * Q2
        sigma = math.sqrt(2*KB*temperature*friction / Q2)
        egt = math.exp(-1*friction*dt)
        sigsqe2gt = sigma * math.sqrt((1-math.exp(-2*friction*dt))/(2*friction))
        # Now add the integrator variables
        w13 = 1.0 / (2.0 - 2.0**(1/3))
        w12 = 1.0 - 2*w13
        fac = dt/(2.0*SINRMTSIntegrator.NRES)
        self.addGlobalVariable('SINR_w1', w13)
        self.addGlobalVariable('SINR_w2', w2)
        self.addGlobalVariable('SINR_w3', w13)
        self.addGlobalVariable('SINR_Lbeta', Lbeta)
        self.addGlobalVariable('SINR_LLp1', LLp1)
        self.addGlobalVariable('SINR_sigsqe2gt', sigsqe2gt)
        self.addGlobalVariable('SINR_egt', egt)
        self.addGlobalVariable('SINR_Q1', Q1)
        self.addGlobalVariable('SINR_Q2', Q2)
        self.addGlobalVariable('SINR_kbT', kbT)
        # Add the integrator DOFs
        for i in range(self.L):
            self.addPerDofVariable('SINR_%dv1' % i, 0)
            self.addPerDofVariable('SINR_%dv2' % i, 0)
        # Temporary variables
        for i in range(self.L):
            self.addGlobalVariable('SINR_v2kw%d' % i)
        self.addGlobalVariable('SINR_q1v1sq', 0)
        self.addGlobalVariable('SINR_Htnres', 0)
        self.addPerDofVariable('SINR_a', 0)
        self.addPerDofVariable('SINR_b', 0)
        self.addPerDofVariable('SINR_sb', 0)
        self.addPerDofVariable('SINR_s', 0)
        self.addPerDofVariable('SINR_sdot', 0)
        self.addPerDofVariable('SINR_f', 0) # backup forces

        self.addUpdateContextState() # Start out allowing context to be updated
        if nrespa > 1:
            self._create_mts_steps()
        else:
            self._create_regular_steps()
        # Compute the kinetic energy

    def _create_mts_steps(self):
        """ Create integration steps with MTS """
        for i in range(self.nrespa-1):
            self._iLndt()
            self._iLvdt()
            self._iLoudt()
            self.addComputePerDof('SINR_f', 'f%d + f%d' % (self.fastgroup,
                self.shortgroup))
            self._iLvdt()
            self._iLndt()
            # We just updated positions -- add the constraints now
            self.addConstrainPositions()
        self._iLndt()
        self._iLvdt()
        self._iLoudt()
        # Do the respa stuff now. Since the short group is actually included as
        # part of the slow group, we need to subtract that out from the slow
        # forces before multiplying by nrespa
        self.addComputePerDof('SINR_f', 'f%d + f%d + %d*(f%d-f%d)'
                % (self.fastgroup, self.shortgroup, self.nrespa,
                   self.slowgroup, self.shortgroup))
        self._iLvdt()
        self._iLndt()
        # Constraints
        self.addConstrainPositions()
        self.addConstrainVelocities()

    def _create_regular_steps(self):
        """ Create the integration step with *no* RESPA """
        self._iLndt()
        self._iLvdt()
        self._iLoudt()
        self.addComputePerDof('SINR_f', 'f')
        self._iLvdt()
        self._iLndt()
        # Constraints
        self.addConstrainPositions()
        self.addConstrainVelocities()

    def _iLndt(self):
        """ Applies the iL_N Liouville operator on the integrator variables """
        repl = dict()
        for i in range(SINRMTSIntegrator.NSY):
            repl['i'] = i
            for j in range(SINRMTSIntegrator.NRES):
                repl['j'] = j
                for k in range(self.L):
                    repl['k'] = k
                    self.addComputePerDof('SINR_%(k)dv2' % repl,
                            'SINR_%(k)dv2 + 0.25*SINR_w%(i)d*(SINR_Q1*'
                            'SINR_%(k)dv1**2-SINR_kbT)/SINR_Q2' % repl)
                self.addComputeGlobal('SINR_q1v1sq', '0')
                for k in range(self.L):
                    repl['k'] = k
                    self.addComputeGlobal('SINR_v2kw%(k)d' % repl, 'exp(-0.5*'
                            'SINR_%(k)dv2*SINR_w%(i)d)' % repl)
                    self.addComputeGlobal('SINR_q1v1sq', 'SINR_q1v1sq + '
                            'SINR_Q1*SINR_%(k)dv1**2*SINR_v2kw%(k)d**2' % repl)
                self.addComputeGlobal('SINR_Htnres', 'sqrt(SINR_Lbeta/'
                        '(m*v**2+SINR_LLp1*SINR_q1v1sq))')
                self.addComputePerDof('v', 'v*SINR_Htnres')
                for k in range(self.L):
                    repl['k'] = k
                    self.addComputePerDof('SINR_%(k)dv1' % repl,
                            'SINR_%(k)dv1*SINR_Htnres*SINR_v2kw%(k)d' % repl)
                for k in range(self.L):
                    repl['k'] = k
                    self.addComputePerDof('SINR_%(k)dv2' % repl,
                            'SINR_%(k)dv2 + 0.25*SINR_w%(i)d*(SINR_Q1*'
                            'SINR_%(k)dv1**2-SINR_kbT)/SINR_Q2' % repl)

    def _iLvdt(self):
        """ Applies the iL_V Liouville operator on the integrator variables """
        self.addComputePerDof('SINR_a', 'SINR_f*v/SINR_Lbeta')
        self.addComputePerDof('SINR_b', 'SINR_f**2/(m*SINR_Lbeta)')
        self.addComputePerDof('SINR_sb', 'sqrt(SINR_b)')
        self.beginIfBlock('SINR_sb*dt*0.5 < %s' % self.errtol) # BEGIN IF
        self.addComputePerDof('SINR_s', '((((SINR_b*SINR_a/24)*dt2 + SINR_b/6)*'
                'dt2 + 0.5*SINR_a)*dt2 + 1)*dt2; dt2 = dt*0.5')
        self.addComputePerDof('SINR_sdot', '(((SINR_b*SINR_a/6)*dt2 +'
                'SINR_b*0.5)*dt2 + SINR_a)*dt2 + 1; dt2 = dt*0.5')
        self.endIfBlock() # ELSE
        self.beginIfBlock('SINR_sb*dt*0.5 >= %s' % self.errtol)
        self.addComputePerDof('SINR_s', '(1/SINR_sb)*sinhbt + (SINR_a/SINR_b)*'
                '(coshbt-1); sinhbt=sinh(SINR_sb*dt*0.5); '
                'coshbt=cosh(SINR_sb*dt*0.5)')
        self.addComputePerDof('SINR_sdot', '(SINR_a/SINR_sb)*sinhbt + coshbt; '
                'sinhbt=sinh(SINR_sb*dt*0.5); coshbt=cosh(SINR_sb*dt*0.5)')
        self.endIfBlock() # END IF
        self.addComputePerDof('v', '(v + (SINR_f/m)*SINR_s)/SINR_sdot')
        # Update the thermostat variables
        for i in range(self.L):
            self.addComputePerDof('SINR_%dv1' % i, 'SINR_%dv1/SINR_sdot' % i)

    def _iLoudt(self):
        """
        Applies the iL_OU Liouville operator on the integrator variables.
        This is the only operator that actually propagates positions
        """
        self.addComputePerDof('x', 'x + v*dt')
        for i in range(self.L):
            self.addComputePerDof('SINR_%dv2' % i,
                    'SINR_%dv2*SINR_egt + gaussian*SINR_sigsqe2gt' % i)

    @staticmethod
    def modify_system(system, short_cutoff, long_cutoff,
                      fastgroup=0, slowgroup=1):
