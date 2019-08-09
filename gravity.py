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

def rotate(vec, angle):
    """
    returns vector rotated by angle
    """
    c = cos(angle)
    s = sin(angle)
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

def distance_point_wall(p, wall):
    AP = p - wall.start
    u = wall.dir
    return np.abs(cross(AP, u))


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
        self.active = active

        self.color = color

        self.cell = (-1, -1)
        self.neighbors = []

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

    def gravity(self, p2, dt):
        if self.active and p2.active:
            dr = p2.pos - self.pos
            r2 = np.linalg.norm(dr)**2
            r_norm = normalize(dr)
            g_acc = G * p2.mass / r2 * r_norm
            self.add_acceleration(g_acc, dt)

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
        pygame.draw.circle(surface, self.color,
                           self.pos.astype(int), self.radius)

    def wall_collision(self, w, dt):
        if w.active:
            next_pos = self.pos + self.vel*dt
            a, b, _ =  intersection(self.pos[0], next_pos[0], w.start[0], w.end[0],
                                    self.pos[1], next_pos[1], w.start[1], w.end[1])
            if 0 <= a <= 1 and 0 <= b <= 1:
                if distance_point_wall(next_pos, w) <= self.radius:
                    # Change velocity vector according to equation
                    self.vel = self.vel - 2 * (np.dot(self.vel, w.normal)) * w.normal
                    return True
        return False




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
    #star.gravity(p1, dt)

    # Move
    for p in planets:
        p.move(dt)
    #star.move(dt)

    # Draw
    screen.fill(3*[0])
    star.draw(screen)
    for p in planets:
        p.draw(screen)
    draw_mouse(screen,
               old_mouse_pos=old_mouse_pos,
               width=2)

    # Update screen
    pygame.display.update()


# Stop pygame
pygame.quit()

# Close program
exit()
