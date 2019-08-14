import numpy as np
cimport numpy as np
from libc.math cimport sqrt, pi, atan2, sin, cos, acos
from tqdm import tqdm
import pygame


#############
# Constants #
#############

CW = 1
CCW = -1

x_axis = np.array([1, 0]).astype(float)
y_axis = np.array([0, 1]).astype(float)


###############
# c functions #
###############

cdef np.ndarray[double, ndim=1] rotate(np.ndarray[double, ndim=1] vec,
                                       double angle):
    # Returns vec rotated by angle (ccw)
    cdef np.ndarray[double, ndim=2] rot_matrix = np.array([[cos(angle), -sin(angle)],
                                                           [sin(angle),  cos(angle)]]).astype(np.float64)
    return np.dot(rot_matrix, vec)


cdef int c_clockwise(np.ndarray[double, ndim=1] v1,
                     np.ndarray[double, ndim=1] v2):
    if v1[1]*v2[0] >= v1[0]*v2[1]:
        return CW
    else:
        return CCW


cdef double dot(np.ndarray[double, ndim=1] v1,
                np.ndarray[double, ndim=1] v2):
    # Returns the dot product of vectors v1 and v2
    return v1[0]*v2[0] + v1[1]*v2[1]


cdef double c_angle_between(np.ndarray[double, ndim=1] v1,
                            np.ndarray[double, ndim=1] v2):
    # Returns angle between vectors v1 and v2
    if norm(v1) == 0 or norm(v2) == 0:
        return 0.0
    else:
        return acos(dot(v1, v2) / (norm(v1)*norm(v2)))


cdef double c_angle_to_xaxis(np.ndarray[double, ndim=1] vec):
    # Return SIGNED angle between vec and x-axis
    return atan2(vec[1], vec[0])

cdef np.ndarray[double, ndim=1] get_vec(double R,
                                        double t):
    # Return a vector of norm R and angle t to x-axis
    cdef np.ndarray[double, ndim=1] vec = np.zeros(2).astype(np.float64)
    vec[0] = R * cos(t)
    vec[1] = R * sin(t)
    return vec


cdef double cross(np.ndarray[double, ndim=1] v1,
                  np.ndarray[double, ndim=1] v2):
    return v1[0]*v2[1] - v1[1]*v2[0] 


cdef double rad2deg(double angle):
    return angle * 180/pi


cdef double deg2rad(double angle):
    return angle * pi/180


cdef double norm(np.ndarray[double, ndim=1] vec):
    return sqrt(dot(vec, vec))


cdef np.ndarray[double, ndim=1] c_normalize(np.ndarray[double, ndim=1] vec):
    cdef double N = norm(vec)
    if N != 0:
        return vec_scale(vec, 1/N)
    else:
        return np.zeros(2).astype(np.float64)


cdef np.ndarray[double, ndim=1] vec_add(np.ndarray[double, ndim=1] v1,
                                        np.ndarray[double, ndim=1] v2):
    cdef np.ndarray[double, ndim=1] v_return = np.zeros(2).astype(np.float64)
    v_return[0] = v1[0] + v2[0]
    v_return[1] = v1[1] + v2[1]
    return v_return


cdef np.ndarray[double, ndim=1] vec_sub(np.ndarray[double, ndim=1] v1,
                                        np.ndarray[double, ndim=1] v2):
    cdef np.ndarray[double, ndim=1] v_return = np.zeros(2).astype(np.float64)
    v_return[0] = v1[0] - v2[0]
    v_return[1] = v1[1] - v2[1]
    return v_return


cdef np.ndarray[double, ndim=1] vec_scale(np.ndarray[double, ndim=1] vec,
                                          double scale):
    cdef np.ndarray[double, ndim=1] v_return = np.zeros(2).astype(np.float64)
    v_return[0] = scale * vec[0]
    v_return[1] = scale * vec[1]
    return v_return


cdef vec_sum(np.ndarray[double, ndim=1] v1,
             np.ndarray[double, ndim=1] v2,
             np.ndarray[double, ndim=1] v3):
    return vec_add(vec_add(v1, v2), v3)


cdef np.ndarray[double, ndim=1] gravity_acc(np.ndarray[double, ndim=1] pos1,
                                            np.ndarray[double, ndim=1] pos2,
                                            double mass2,
                                            double G):
    # Force directed from 1 to 2
    cdef np.ndarray[double, ndim=1] norm_dist_vec = c_normalize(vec_sub(pos2, pos1))
    cdef double acc_magnitude = G * mass2 / dist2(pos1, pos2)
    return vec_scale(norm_dist_vec, acc_magnitude)

cdef double dist2(np.ndarray[double, ndim=1] v1,
                  np.ndarray[double, ndim=1] v2):
    return (v1[0]-v2[0])**2 + (v1[1]-v2[1])**2

cdef double c_dist(np.ndarray[double, ndim=1] v1,
                   np.ndarray[double, ndim=1] v2):
    return sqrt(dist2(v1, v2))

cdef double kinetic_energy(np.ndarray[double, ndim=1] v,
                           double m):
    return 0.5 * m * dot(v, v)


cdef np.ndarray[double, ndim=1] c_orbital_params(np.ndarray[double, ndim=1] planet_pos,
                                                 np.ndarray[double, ndim=1] star_pos,
                                                 np.ndarray[double, ndim=1] planet_vel,
                                                 np.ndarray[double, ndim=1] star_vel,
                                                 double m, double M, double G):
    # distance
    cdef double r = c_dist(planet_pos, star_pos)
    cdef np.ndarray[double, ndim=1] r_vec = planet_pos - star_pos

    # Velocity^2
    cdef np.ndarray[double, ndim=1] v_vec = planet_vel - star_vel
    cdef double v2 = dot(v_vec, v_vec)

    # Reduced mass
    cdef double mu = G*(m+M)

    # Specific orbital energy
    cdef double E = 0.5*v2 - mu/r

    # Standard gravitational parameter
    cdef double sgp = G * M

    # Specific relative angular momentum
    cdef double h = cross(r_vec, planet_vel)

    # Eccentricity
    cdef double e = sqrt(1 + 2*E*h**2/mu**2)

    # Semi major axis
    cdef double a = -mu/(2*E)

    # Semi minor axis
    cdef double b = a * sqrt(1-e**2)

    # Return
    cdef np.ndarray[double, ndim=1] returned_vec = np.zeros(3).astype(np.float64)
    returned_vec[0] = e
    returned_vec[1] = a
    returned_vec[2] = b
    return returned_vec


cdef double c_get_ellipse_angle(np.ndarray[double, ndim=1] planet_pos,
                                np.ndarray[double, ndim=1] star_pos,
                                double a, double e):
    # Distance and angle from planet to star
    cdef double r = c_dist(planet_pos, star_pos)
    cdef double phi = c_angle_to_xaxis(star_pos - planet_pos)

    # Distance between planet and f2
    cdef double d = 2*a - r

    # Angle between planet-star line and planet-f2 line
    cdef double theta = acos((r**2 + d**2 - (2*a*e)**2)/(2*r*d))

    # Position of f2
    cdef np.ndarray[double, ndim=1] d_vec = get_vec(d, phi+theta)
    cdef np.ndarray[double, ndim=1] f2 = planet_pos + d_vec

    # Get angle of f2-star line
    cdef double angle_f2_star = c_angle_between(x_axis, star_pos - f2)

    return angle_f2_star


cdef np.ndarray[double, ndim=2] c_get_ellipse(np.ndarray[double, ndim=1] center,
                                              double a, double b, double e,
                                              double angle, int num_points):
    cdef np.ndarray[double, ndim=1] ts = np.linspace(0, 2*pi, num_points).astype(np.float64)
    cdef np.ndarray[double, ndim=2] points = np.zeros(shape=(num_points, 2))
    cdef np.ndarray[double, ndim=1] f1_transformed = center - a*e*np.array([cos(angle), sin(angle)]).astype(np.float64)
    cdef int i
    for i in range(num_points):
        points[i][0] = a * cos(ts[i])
        points[i][1] = b * sin(ts[i])
        points[i] = rotate(points[i], angle) + f1_transformed
    return points


############################
# Python funation wrappers #
############################

def normalize(vec):
    return c_normalize(vec)

def dist(v1, v2):
    return c_dist(v1, v2)

def angle_between(v1, v2):
    return c_angle_between(v1, v2)

def clockwise(v1, v2):
    return c_clockwise(v1, v2)

def make_norm(vec, scale):
    vec /= norm(vec)
    vec *= scale

def py_rotate(vec, angle):
    return rotate(vec, angle)

def orbital_params(planet, star, G):
    return c_orbital_params(planet.pos, star.pos,
                            planet.vel, star.vel,
                            planet.mass, star.mass,
                            G)

def get_ellipse_angle(planet_pos, star_pos, a, e):
    return c_get_ellipse_angle(planet_pos, star_pos, a, e)

def get_ellipse(center, a, b, e, angle, num_points=100):
    return c_get_ellipse(center, a, b, e, angle, num_points)
