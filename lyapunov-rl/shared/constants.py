"""Physical constants and unit conversions for orbital mechanics.

All internal computation uses SI units (meters, seconds, kg).
AU/day conversions provided for interfacing with ephemeris data and REBOUND.
"""

import jax.numpy as jnp

# --- Gravitational parameters (GM, m^3/s^2) ---
# Source: IAU 2015 nominal values + JPL DE440
MU_SUN = 1.32712440018e20
MU_EARTH = 3.986004418e14
MU_MARS = 4.282837e13
MU_VENUS = 3.24859e14
MU_JUPITER = 1.26686534e17

# --- Masses (kg) ---
M_SUN = 1.98892e30
M_EARTH = 5.97217e24
M_MARS = 6.4171e23
M_VENUS = 4.8675e24
M_JUPITER = 1.89813e27

# --- Orbital semi-major axes (m) ---
A_EARTH = 1.496e11  # 1 AU
A_MARS = 2.279e11
A_VENUS = 1.082e11
A_JUPITER = 7.785e11

# --- Unit conversions ---
AU = 1.49597870700e11  # meters per AU
DAY = 86400.0  # seconds per day
YEAR = 365.25 * DAY

# --- Spacecraft defaults ---
DEFAULT_MAX_THRUST = 0.5  # Newtons (low-thrust ion engine scale)
DEFAULT_SC_MASS = 1000.0  # kg (dry mass)
DEFAULT_ISP = 3000.0  # seconds (ion engine specific impulse)
G0 = 9.80665  # m/s^2, standard gravity for Isp conversion

# --- Target orbital energies (specific, J/kg) ---
# epsilon = -mu / (2a) for circular orbit
EPSILON_EARTH = -MU_SUN / (2.0 * A_EARTH)
EPSILON_MARS = -MU_SUN / (2.0 * A_MARS)
EPSILON_VENUS = -MU_SUN / (2.0 * A_VENUS)
EPSILON_JUPITER = -MU_SUN / (2.0 * A_JUPITER)

# --- Normalization scales ---
# Used to keep observation values in ~[-1, 1] range for neural nets
NORM_POS = AU  # position normalization (1 AU)
NORM_VEL = 3e4  # velocity normalization (~30 km/s, Earth orbital speed)
NORM_ENERGY = abs(EPSILON_EARTH)  # energy normalization

# --- Simulation defaults ---
DEFAULT_DT = 3600.0  # 1 hour timestep for leapfrog integrator
DEFAULT_EPISODE_DAYS = 400  # ~13 months, enough for Earth-Mars Hohmann + margin
DEFAULT_EPISODE_STEPS = int(DEFAULT_EPISODE_DAYS * DAY / DEFAULT_DT)

# --- Planet indices (for observation vector) ---
# Maps planet name to index in the planet array
PLANET_NAMES = ("mercury", "venus", "earth", "mars", "jupiter")
PLANET_MU = jnp.array([2.2032e13, MU_VENUS, MU_EARTH, MU_MARS, MU_JUPITER])
PLANET_A = jnp.array([5.791e10, A_VENUS, A_EARTH, A_MARS, A_JUPITER])
