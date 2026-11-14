# Tension-noise convention: source verification pending

The circular-wall sweep records `mu`, the implemented noise input. Its identification
with the paper's `Delta T/T0` has **not been verified against the original stochastic
equation and noise covariance**. Earlier plot labels in this repository are not
independent evidence for that identification. The accompanying numerical sweeps are
results for the implemented process below, not yet calibrated paper-activity values.

## Verified implementation

In code units (tension in T0 and time in tau_R), `circle/disc.py` and
`activefoam/foam_full.py` both implement

```
T_next = T - (dt/tauT)*(T - T_fixed)
         + (c/2)*(mu/tauT)*sqrt(dt)*xi
xi ~ N(0, 1), independently per film and step
c = 2 for cell-cell contacts, 1 for cell-space films
```

The continuous-time SDE corresponding to this Euler–Maruyama update is

```
dT = -(T - T_fixed)/tauT * dt + (c/2)*(mu/tauT) * dW
E[dW**2] = dt
```

For a persistent film with fixed type and fixed mean tension, its stationary raw
OU variance is `(c/2)^2 * mu^2/(2*tauT)`. The discrete update instead has exact
stationary variance `(c/2)^2 * mu^2/(2*tauT - dt)` when `0 < dt/tauT < 2`.
These are properties of the stored tension, before the force uses `max(T, 0)`;
topology changes and transients can alter observed tension statistics.

At tauT=10 and dt=0.005, an interior film's continuum standard deviation is
0.223607*mu and its discrete standard deviation is 0.223635*mu. A free film has
half those values. This calculation does not establish the paper's convention.

## What remains to be checked

Read the original equation together with the covariance of its white noise and its
choice of time units. A factor of tauT can sit in either the written noise amplitude
or the covariance, so the equation alone may not establish a conversion.

**Conditional example only:** if the paper defines Delta T as the stationary RMS
fluctuation of an interior contact, then in these code units the corresponding SDE
uses `(Delta T/T0)*sqrt(2/tauT)*dW`. Matching the current implementation requires
`mu=sqrt(2*tauT)*(Delta T/T0)`, rather than equality. This is not asserted to be the
paper's definition without reading its noise covariance.

The local MATLAB acceleration copy uses the same noise coefficient as the Python
code. The directory for the original MATLAB archive contains only a provenance
record in this checkout, so the untouched original update has not been reverified
in this audit. Access to the paper at pmc.ncbi.nlm.nih.gov was denied by the running
environment's domain policy on 2026-11-14. No stochastic update or simulation result
has been changed on the basis of an assumed convention.
