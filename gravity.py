#!/usr/bin/env python3.6
# -*- coding: iso-8859-15 -*-


from sys import exit, argv
import numpy as np
import pygame


###################
# Maths functions #
###################

def normalize(vec):
    """
    Returns normalized vector
    """
    L = np.linalg.norm(vec)
    if L != 0:
        return vec/L
    else:
        return vec*0.0

def scale_vec(vec, size):
    """
    Returns scaled to size
    """
    new_vec = normalize(vec)
    return new_vec * size

def get_angle(vec):
    angle = np.arctan2(vec[1], vec[0])
    return np.degrees(angle)

def get_angle_between(v1, v2):
    return get_angle(v1) - get_angle(v2)

def rotate(vec, angle):
    """
    returns vector rotated by angle
    """
    c = np.cos(angle)
    s = np.sin(angle)
    mat = np.array([[c, -s],
                    [s,  c]])
    return np.dot(mat, vec)

def intersection(x1, x2, x3, x4,
                 y1, y2, y3, y4):
    """
    returns the intersection
    point of two line segments.
    (used for particle-wall interaction)
    """
    a = ((y3-y4)*(x1-x3) + (x4-x3)*(y1-y3))
    b = ((y1-y2)*(x1-x3) + (x2-x1)*(y1-y3))
    c = ((x4-x3)*(y1-y2) - (x1-x2)*(y4-y3))
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    if c != 0.0:
        """
        c = 0 means that the
        intersection point exists.
        """
        return a/c, b/c, p1 + (p2-p1)*(a/c)
    else:
        return 0, 0, np.zeros(2)

def dist(p1, p2):
    """
    returns the Euclidean distance
    between two points p1, p2
    """
    return np.linalg.norm(p2-p1)

def cross(v1, v2):
    return (v1[0]*v2[1] - v1[1]*v2[0])

def orbital_params(x, X, v, m, M):
    # Distance
    dr = x - X
    r = dist(x, X)

    # Velocity squared
    v2 = np.dot(v, v)

    # Reduced mass
    mu = G*M

    # Specific orbital energy
    E = v2/2 - mu/r

    # Specific relative angular momentum
    h = cross(dr, v)

    # Eccentricity
    e = np.sqrt(1 + 2*E*h**2/mu**2)

    # Major axis
    a = mu*r / (2*mu - r*v2)

    # Minor axis
    b = a * np.sqrt(1-e**2)

    # Position of second focus
    perp_vec = rotate(v, np.pi/2)
    da = get_angle_between(dr, perp_vec)
    r2 = rotate(dr, 2*da)
    r2 = scale_vec(r2, 2*a-r)
    f2 = x + r2

    # Angle of major axis
    angle = get_angle(f2 - X)

    return e, a, b, angle


def get_ellipse(x, X, v, m, M,
                center, num_points=1000):
    e, a, b = orbital_params(x, X, v, m, M)
    angle = get_angle(x - X)
    points = np.zeros((num_points, 2))
    ts = np.linspace(0, 2*np.pi, num_points)
    for i in range(num_points):
        c = np.cos(ts[i])
        s = np.sin(ts[i])
        r = 2*a*b/np.sqrt((2*b*c)**2+(a*s)**2)
        points[i][0] = r * c
        points[i][1] = r * s
        points[i] = rotate(points[i], -angle) + center

    return points


###########
# Classes #
###########

class TextOnScreen:
    """
    A class to display a text
    on screen, at a chosen location,
    font, size, etc.
    """
    def __init__(self, pos=(0,0), color=(0, 200, 0),
                       font='Cabin', size=15, text='',
                       centered=False):
        self.pos = pos
        self.color = color
        self.font = pygame.font.SysFont(font, size)
        self.text = text
        self.centered = centered

    def set_text(self, text):
        self.text = text

    def display(self, surface):
        render = self.font.render(self.text,
                                  False,
                                  self.color)
        if self.centered:
            text_rect = render.get_rect(center=(w/2, h/2))
            surface.blit(render, text_rect)
        else:
            surface.blit(render, self.pos)


class Body:
    """
    A star/planet with position, velocity, mass
    and a radius. Moves according to Newton's
    laws of motion, including gravity.
    """
    def __init__(self, id=-1,
                 pos=np.zeros(2),
                 vel=np.zeros(2),
                 mass=1,
                 radius=5,
                 active=True,
                 atmo_radius=0,
                 color='blue'):

        self.id = id

        self.pos = pos
        self.vel = vel
        self.speed = np.linalg.norm(self.vel)
        self.mass = mass
        self.radius = radius
        self.atmo_radius = atmo_radius
        self.total_radius = self.radius + self.atmo_radius
        self.active = active

        self.color = color
        self.atmo_color = (np.array(self.color) * 0.05).astype(int)

        self.cell = (-1, -1)
        self.neighbors = []

        self.ellipse_surf = None

    def create_ellipse(self, star):
        e, a, b, angle = orbital_params(self.pos, star.pos,
                                        self.vel,
                                        self.mass, star.mass)
        e = e
        a = a
        b = b
        orbit_angle = np.degrees(angle)

        self.focus = (star.pos[0]-a*(1+e), star.pos[1]-b)
        new_surf = pygame.Surface((2*a, 2*b))
        pygame.draw.ellipse(new_surf, [255, 255, 255],
                            (0, 0, 2*a, 2*b), 3)
        self.ellipse_surf = pygame.transform.rotate(new_surf, orbit_angle)

    def set_cell(self, x, y):
        self.cell = (x, y)

    def set_neighors(self, grid):
        # Reset current neighbors list
        self.neighbors = []

        # Create new list
        x, y = self.cell
        Nx, Ny = grid.Nx, grid.Ny
        neighbors = [grid.objects[i][j] for i in range(x-1, x+2) if 0 <= i < Nx
                                        for j in range(y-1, y+2) if 0 <= j < Ny]
        self.neighbors = [object for sublist in neighbors
                                 for object in sublist
                                 if object is not self]

    def flip_selection_status(self):
        if self.selected:
            self.unselect()
        else:
            self.select()

    def set_atmo_radius(self, atmo_radius):
        self.atmo_radius = atmo_radius
        self.total_radius = self.radius + self.atmo_radius

    def gravity(self, p2, dt):
        if self.active and p2.active:
            dr = p2.pos - self.pos
            r2 = np.linalg.norm(dr)**2
            r_norm = normalize(dr)
            g_acc = G * p2.mass / r2 * r_norm
            self.add_acceleration(g_acc, dt)

    def in_atmosphere(self, p2):
        d = dist(self.pos, p2.pos)
        if p2.radius <= d <= p2.total_radius:
            percent_atmosphere = (d-p2.radius) / p2.atmo_radius
            A = np.exp(-.02 / percent_atmosphere)
            self.vel = self.vel * A

    def add_acceleration(self, a, dt):
        self.vel += a*dt

    def move(self, dt):
        if self.active:
            self.pos += self.vel * dt
            self.speed = np.linalg.norm(self.vel)

    def in_bounds(self):
        xin = min_x <= self.pos[0] <= max_x
        yin = min_y <= self.pos[1] <= max_y
        if xin and yin:
            return True
        else:
            return False

    def get_potential_grav_energy(self, yref):
        r2 = (yref - self.pos[1])**2
        return G * self.mass / r2

    def get_kinetic_energy(self):
        return 0.5*self.mass*np.linalg.norm(self.vel)**2

    def set_kinetic_energy(self, energy):
        self.vel = scale_vec(self.vel, np.sqrt(2*energy/self.mass))

    def draw(self, surface):
        if self.ellipse_surf:
            surface.blit(self.ellipse_surf, self.focus)

        pygame.draw.circle(surface, self.atmo_color,
                           self.pos.astype(int), self.radius + self.atmo_radius)
        pygame.draw.circle(surface, self.color,
                           self.pos.astype(int), self.radius)



##################
# Draw mouse pos #
##################

def draw_mouse(surface,
               old_mouse_pos=None,
               width=2):
    if mouse_status == SET_VELOCITY:
        pygame.draw.circle(surface, [0, 200, 255],
                           mouse_pos, planet_radius)
    if mouse_status == PLACE_PLANET:
        pygame.draw.line(surface,
                         [255, 0, 0],
                         old_mouse_pos,
                         mouse_pos,
                         width)


####################
# Status variables #
####################

PLACE_PLANET = 0
SET_VELOCITY = 1
mouse_status = -1

planet_radius = 10

SUN_ATMOSHPHERE = False


######################
# Physics parameters #
######################

dt = 0.01
G = 0.5E4


#####################
# Initialize pygame #
#####################

w, h = 1200, 1000
center = np.array([w/2, h/2])
pygame.display.init()
screen = pygame.display.set_mode((w, h))

pygame.font.init()


#####################
# Stars and planets #
#####################

star = Body(id = 0,
            pos = center,
            mass = 1E3,
            radius = 100,
            atmo_radius = 0,
            color = [255, 160, 0])

planets = []


#############
# Main loop #
#############

run = True
mpos = []
frame_num = 0
old_mouse_pos = pygame.mouse.get_pos()
while run:
    # Input events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                run = False
            if event.key == pygame.K_a:
                SUN_ATMOSHPHERE = not (SUN_ATMOSHPHERE)
                if SUN_ATMOSHPHERE:
                    star.set_atmo_radius(100)
                else:
                    star.set_atmo_radius(0)

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            # Cycle between 3 possible mouse states
            mouse_status = (mouse_status + 1) % 2

            if mouse_status == PLACE_PLANET:
                new_planet = Body(pos = np.array(mouse_pos).astype(float),
                                  mass = 0.01,
                                  radius = planet_radius,
                                  color = [0, 200, 255],
                                  active = False)
                planets.append(new_planet)
                old_mouse_pos = mouse_pos

            if mouse_status == SET_VELOCITY:
                new_planet.vel = np.array(mouse_pos) - new_planet.pos
                new_planet.active = True
                new_planet.create_ellipse(star)

        if event.type == pygame.MOUSEBUTTONDOWN:
            if 2 <= planet_radius <= 20:
                if event.button == 4:
                    planet_radius = planet_radius + 1
                if event.button == 5:
                    planet_radius = planet_radius - 1


    # Mouse position
    mouse_pos = pygame.mouse.get_pos()

    # Physics
    for p in planets:
        p.gravity(star, dt)
        p.in_atmosphere(star)
    #star.gravity(p1, dt)

    # Move
    for p in planets:
        p.move(dt)
        if dist(p.pos, star.pos) <= p.radius + star.radius:
            planets.remove(p)
    #star.move(dt)

    # Draw
    screen.fill(3*[0])
    for p in planets:
        p.draw(screen)
    draw_mouse(screen,
               old_mouse_pos=old_mouse_pos,
               width=2)
    star.draw(screen)

    # Update screen
    pygame.display.update()


# Stop pygame
pygame.quit()

# Close program
exit()
