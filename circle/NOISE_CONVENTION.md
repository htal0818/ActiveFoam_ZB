# Tension-noise convention

The user supplied the original tension equation:

```
tau_T * dT_ij/dt = -(T_ij - T0_ij) + delta_T * zeta_ij(t)
```

Here zeta is Gaussian white noise. This has the same algebraic form as the current
interior-contact implementation. The previous suggestion that a factor
`sqrt(2/tau_T)` necessarily belongs in the noise coefficient was unwarranted: that
factor belongs to a parameterization in which the amplitude is defined as the
stationary RMS fluctuation. The supplied equation does not itself make that definition.

## Mapping and normalization

In code units (tension in T0 and time in tau_R), assume

```
E[zeta(t) zeta(t')] = delta(t - t')
zeta(t) dt = dW(t),  E[dW**2] = dt
```

Dividing the supplied equation by tau_T gives

```
dT = -(T - T_fixed)/tauT * dt + (delta_T/T0)/tauT * dW
```

Thus **mu = delta_T/T0 for interior contacts under this noise convention**.
Gaussian white noise alone specifies neither its covariance prefactor nor its time
units. If the covariance in these time units is `C*delta(t-t')`, the mapping is
instead `mu = sqrt(C)*(delta_T/T0)`. The user supplied the equation but not an explicit
covariance, so equivalence of the algebraic form is established while the numerical
mapping remains conditional on this normalization.

## Verified implementation

`circle/disc.py` and `activefoam/foam_full.py` both implement

```
T_next = T - (dt/tauT)*(T - T_fixed)
         + (c/2)*(mu/tauT)*sqrt(dt)*xi
xi ~ N(0, 1), independently per film and step
c = 2 for cell-cell contacts, 1 for cell-space films
```

The factor `c/2` is one for interior contacts. Free films receive half the noise
coefficient; this separate boundary convention is inherited from the planar code
and is not established by the supplied interior-contact equation alone.

For a persistent film with fixed type and fixed mean tension, the stationary raw
OU variance is `(c/2)^2 * mu^2/(2*tauT)`. The discrete update has exact stationary
variance `(c/2)^2 * mu^2/(2*tauT - dt)` when `0 < dt/tauT < 2`.
These describe stored tension, before forces use `max(T, 0)`; topology changes and
transients can alter measured tension statistics.

At tauT=10 and dt=0.005, an interior film's continuum standard deviation is
0.223607*mu and its discrete standard deviation is 0.223635*mu. This is a derived
property of the supplied equation with unit white noise, not a replacement ODE or
a reason to reinterpret delta_T as the RMS fluctuation.

The local MATLAB acceleration copy uses the same noise coefficient as Python.
The directory for the original MATLAB archive contains only a provenance record
in this checkout. Online paper access was denied by the environment's domain policy
on 2026-11-14; the original equation above was provided directly by the user.
No stochastic update, parameter value or simulation result was changed in this audit.
