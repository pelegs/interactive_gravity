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


###############
# c functions #
###############

cdef int c_clockwise(np.ndarray[double, ndim=1] v1,
                     np.ndarray[double, ndim=1] v2):
    if v1[1]*v2[0] >= v1[0]*v2[1]:
        return CW
    else:
        return CCW


cdef double angle_between(np.ndarray[double, ndim=1] v1,
                          np.ndarray[double, ndim=1] v2):
    return acos(dot(v1, v2) / (norm(v1)*norm(v2)))


cdef double dot(np.ndarray[double, ndim=1] v1,
                np.ndarray[double, ndim=1] v2):
    return v1[0]*v2[0] + v1[1]*v2[1]


cdef double cross(np.ndarray[double, ndim=1] v1,
                  np.ndarray[double, ndim=1] v2):
    cdef double theta = angle_between(v1, v2)
    return norm(v1) * norm(v2) * sin(theta)


cdef double get_xy_angle(np.ndarray[double, ndim=1] vec):
    angle = atan2(vec[1], vec[0])
    if angle < 0:
        angle += 2*pi
    return angle


cdef double rad2deg(double angle):
    return angle * 180/pi


cdef double deg2rad(double angle):
    return angle * pi/180


cdef np.ndarray[double, ndim=1] mat_vec_dot(np.ndarray[double, ndim=2] matrix,
                                            np.ndarray[double, ndim=1] vec):
    cdef np.ndarray[double, ndim=1] v_new = np.zeros(2).astype(np.float64)
    v_new[0] = dot(matrix[0], vec)
    v_new[1] = dot(matrix[1], vec)
    return v_new


cdef np.ndarray[double, ndim=1] rotate(np.ndarray[double, ndim=1] vec,
                                       double angle):
    cdef double s = sin(angle)
    cdef double c = cos(angle)
    cdef np.ndarray[double, ndim=2] R_matrix = np.array([[c, -s],
                                                         [+s, c]])
    return mat_vec_dot(R_matrix, vec)


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


cdef double c_eccentricity(np.ndarray[double, ndim=1] small_pos,
                           np.ndarray[double, ndim=1] large_pos,
                           np.ndarray[double, ndim=1] small_vel,
                           np.ndarray[double, ndim=1] large_vel,
                           double M, double m, double G):
    # distance
    cdef double r = c_dist(small_pos, large_pos)
    cdef np.ndarray[double, ndim=1] r_vec = small_pos - large_pos

    # Velocity^2
    cdef np.ndarray[double, ndim=1] v_vec = small_vel - large_vel
    cdef double v2 = dot(v_vec, v_vec)

    # Reduced mass
    cdef double mu = G*(m+M)

    # Specific orbital energy
    cdef double E = 0.5*v2 - mu/r

    # Standard gravitational parameter
    cdef double sgp = G * M

    # Specific relative angular momentum
    cdef double h = cross(r_vec, small_vel)

    return sqrt(1 + 2*E*h**2/mu**2)


cdef double c_a(np.ndarray[double, ndim=1] small_pos,
                np.ndarray[double, ndim=1] large_pos,
                np.ndarray[double, ndim=1] small_vel,
                np.ndarray[double, ndim=1] large_vel,
                double M, double m, double G):
    # distance
    cdef double r = c_dist(small_pos, large_pos)

    # Velocity^2
    cdef np.ndarray[double, ndim=1] v_vec = small_vel - large_vel
    cdef double v2 = dot(v_vec, v_vec)

    # Reduced mass
    cdef double mu = G*(m+M)

    # Specific orbital energy
    cdef double E = 0.5*v2 - mu/r

    return -mu/(2*E)

cdef double c_b(np.ndarray[double, ndim=1] small_pos,
                 np.ndarray[double, ndim=1] large_pos,
                 np.ndarray[double, ndim=1] small_vel,
                 np.ndarray[double, ndim=1] large_vel,
                 double M, double m, double G):
    cdef double e = c_eccentricity(small_pos, large_pos, small_vel, large_vel, M, m, G)
    cdef double a = 0.5 * c_a(small_pos, large_pos, small_vel, large_vel, M, m, G)
    return a * sqrt(1-e**2)


cdef np.ndarray[double, ndim=2] c_get_ellipse(np.ndarray[double, ndim=1] center,
                                              double a, double b, double angle,
                                              int num_points):
    cdef np.ndarray[double, ndim=1] ts = np.linspace(0, 2*pi, num_points).astype(np.float64)
    cdef np.ndarray[double, ndim=2] points = np.zeros(shape=(num_points, 2))
    cdef double r
    cdef int i
    for i in range(num_points):
        c = cos(ts[i])
        s = sin(ts[i])
        if (2*b*c)**2+(a*s)**2 != 0:
            r = 2*a*b/sqrt((2*b*c)**2+(a*s)**2)
        else:
            r = 0.0
        points[i][0] = r * c
        points[i][1] = r * s
        points[i] = rotate(points[i], angle) + center
    return points


############################
# Python funation wrappers #
############################

def normalize(vec):
    return c_normalize(vec)

def dist(v1, v2):
    return c_dist(v1, v2)

def ellipse_axes(small, large, G):
    a = c_a(small.pos, large.pos, small.vel, large.vel, 1.0, large.mass, G)
    b = c_b(small.pos, large.pos, small.vel, large.vel, 1.0, large.mass, G)
    return a, b

def eccentricity(small, large, G):
    return c_eccentricity(small.pos, large.pos, small.vel, large.vel, 1.0, large.mass, G)

def get_angle(vec):
    x_axis = np.array([1, 0]).astype(np.float64)
    return angle_between(vec, x_axis)

def py_angle_between(v1, v2):
    return angle_between(v1, v2)

def clockwise(v1, v2):
    return c_clockwise(v1, v2)

def make_norm(vec, scale):
    vec /= norm(vec)
    vec *= scale

def py_rotate(vec, angle):
    return rotate(vec, angle)

def get_ellipse(center, a, b, angle, num_points=100):
    return c_get_ellipse(center, a, b, angle, num_points)
