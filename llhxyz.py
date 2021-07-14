"""Convert between (latitude, longitude) and Cartesian coordinates.
Computes distances between two geographic points on the WGS84 ellipsoid.
"""

import numpy as np

R_A = 6378137.0  # semi-major axis
R_E2 = 0.0066943799901499996  # eccentricity of earth ellipsoid

# pylint: disable=invalid-name


def llh2xyz(lat, lon, alt):
  """Convert latitude, longitude to Cartesian coordinates."""
  deg2rad = np.pi / 180.0
  r_re = R_A / np.sqrt(1.0 - R_E2 * np.sin(lat * deg2rad)**2)
  x = (r_re + alt) * np.cos(lat * deg2rad) * np.cos(lon * deg2rad)
  y = (r_re + alt) * np.cos(lat * deg2rad) * np.sin(lon * deg2rad)
  z = (r_re * (1.0 - R_E2) + alt) * np.sin(lat * deg2rad)
  return x, y, z


def xyz2llh(x, y, z):
  """Convert Cartesian coordinates to latitude, longitude."""
  rad2deg = 180.0 / np.pi
  r_q2 = 1.0 / (1.0 - R_E2)
  r_q = np.sqrt(r_q2)
  r_q3 = r_q2 - 1.0
  r_b = R_A * np.sqrt(1.0 - R_E2)
  lon = np.arctan2(y, x)
  r_p = np.sqrt(x**2 + y**2)
  r_tant = (z / r_p) * r_q
  r_theta = np.arctan(r_tant)
  r_tant_num = (z + r_q3 * r_b * np.sin(r_theta)**3)
  r_tant_denom = (r_p - R_E2 * R_A * np.cos(r_theta)**3)
  r_tant = r_tant_num / r_tant_denom
  lat = np.arctan(r_tant)
  r_re = R_A / np.sqrt(1.0 - R_E2 * np.sin(lat)**2)
  alt = r_p / np.cos(lat) - r_re
  return rad2deg * lat, rad2deg * lon, alt
