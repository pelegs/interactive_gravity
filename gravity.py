#!/usr/bin/env python3.6
# -*- coding: iso-8859-15 -*-


from sys import exit, argv
import numpy as np
from numpy import sqrt, sin, cos, pi
import pygame
from libgrav import *


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

        self.points = np.ones((2, 2))*-1
        self.e, self.a, self.b = [-1]*3
        self.ellipse_surface = None

    def set_orbital_params(self, star):
        self.star = star
        self.e, self.a, self.b = orbital_params(self, star, G)
        self.ellipse_surface = None

    def create_ellipse(self):
        r1 = self.pos - star.pos
        perp_vec = py_rotate(self.vel, np.pi/2)
        r1_angle = np.arctan2(r1[1], r1[0])

        #c = clockwise(r1, perp_vec)
        da = -py_angle_between(r1, perp_vec)
        r2 = -py_rotate(r1, 2*da)
        make_norm(r2, 2*self.a - np.linalg.norm(r1))

        second_center = self.pos + r2
        self.r12_angle_rad = get_angle(second_center - star.pos)
        self.r12_angle_deg = np.degrees(self.r12_angle_rad)

        if 0 <= self.a <= w and 0 <= self.b <= h:
            self.ellipse_surface = pygame.Surface((2*self.a, 2*self.b))
            self.ellipse_surface.fill([0, 0, 0])
            self.ellipse_surface.set_colorkey([0, 0, 0])

            pygame.draw.circle(self.ellipse_surface,
                               [0, 255, 150],
                               np.array([self.a*(1+self.e), self.b]).astype(int),
                               5)
            pygame.draw.circle(self.ellipse_surface,
                               [255, 0, 0],
                               np.array([self.a*(1-self.e), self.b]).astype(int),
                               5)

            # Avoid radius being smaller than width
            if self.a <= 1 or self.b <= 1:
                pygame.draw.line(self.ellipse_surface,
                                 [255, 255, 255],
                                 (0, self.b),
                                 (2*self.a, self.b),
                                 1)
            # Normal operation
            else:
                pygame.draw.ellipse(self.ellipse_surface,
                                    [255, 255, 255],
                                    (0, 0, 2*self.a, 2*self.b),
                                    1)

            # Transform to fit shape
            self.ellipse_surface = pygame.transform.rotate(self.ellipse_surface, -self.r12_angle_deg)

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
        # Draw trajectory
        if self.ellipse_surface is not None:
            edge_correction = np.array(self.ellipse_surface.get_size()) * 0.5
            angle = pi - self.r12_angle_rad
            focus_correction = np.array([cos(-angle), sin(-angle)]) * self.a*self.e
            pos = self.star.pos - edge_correction - focus_correction
            surface.blit(self.ellipse_surface, pos.astype(int))

        # Draw atmosphere
        if self.atmo_radius >= 0.0:
            pygame.draw.circle(surface, self.atmo_color,
                               self.pos.astype(int), self.radius + self.atmo_radius)

        # Draw self
        pygame.draw.circle(surface, self.color,
                           self.pos.astype(int), self.radius)



##################
# Draw mouse pos #
##################

def draw_mouse(surface,
               original_pos=None,
               width=2):
    if mouse_status == SET_VELOCITY:
        pygame.draw.circle(surface, [0, 200, 255],
                           mouse_pos, planet_radius)
    if mouse_status == PLACE_PLANET:
        pygame.draw.line(surface,
                         [255, 0, 0],
                         original_pos,
                         mouse_pos,
                         width)


#####################
# Initialize pygame #
#####################

w, h = 1200, 1000
center = np.array([w/2, h/2])
pygame.display.init()
screen = pygame.display.set_mode((w, h))

pygame.font.init()


###################
# On-screen texts #
###################

eccentricity_text = TextOnScreen(pos=(10, 0),
                                 color=(0, 250, 120),
                                 text='eccentricity: -')


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
mouse_pos = (0,0)
original_pos = (0,0)
eccentricity_val = -1
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
            if event.key == pygame.K_d:
                planets = []

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
                original_pos = new_planet.pos

            if mouse_status == SET_VELOCITY:
                new_planet.active = True

        if event.type == pygame.MOUSEBUTTONDOWN:
            if 2 <= planet_radius <= 20:
                if event.button == 4:
                    planet_radius = planet_radius + 1
                if event.button == 5:
                    planet_radius = planet_radius - 1


    # Mouse position
    old_mouse_pos = mouse_pos
    mouse_pos = pygame.mouse.get_pos()

    # Show trajectory
    if mouse_status == PLACE_PLANET and mouse_pos != old_mouse_pos:
        new_planet.vel = mouse_pos - new_planet.pos
        new_planet.set_orbital_params(star)
        new_planet.create_ellipse()
        eccentricity_val = new_planet.e

    # Physics
    for p in planets:
        p.gravity(star, dt)
        p.in_atmosphere(star)

    # Move
    for p in planets:
        p.move(dt)
        if dist(p.pos, star.pos) <= p.radius + star.radius:
            planets.remove(p)

    # Draw
    screen.fill(3*[0])
    star.draw(screen)
    for p in planets:
        p.draw(screen)
    draw_mouse(screen,
               original_pos=original_pos,
               width=2)

    # Update text on screen
    eccentricity_text.set_text('eccentricity: {:0.2f}'.format(eccentricity_val))
    eccentricity_text.display(screen)

    # Update screen
    pygame.display.update()


# Stop pygame
pygame.quit()

# Close program
exit()
