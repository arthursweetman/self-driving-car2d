import pandas as pd
import numpy as np
import pymunk as pm
import pygame as pg
import math

import pymunk.pygame_util as util


width, height = 690, 600
screen = pg.display.set_mode((width, height))

# Declare Pymunk Space
space = pm.Space()
space.gravity = 0, 0

SIGHT_LENGTH = 1000

class Car():
    def __init__(self):
        self.body = pm.Body()
        self.body.position = 100,100
        self.body.velocity = 0,0
        self.body.angle = 0
        self.sights = {}

        w, h = 30, 20
        self.shape = pm.Poly.create_box(self.body, (w, h))
        self.shape.mass = 10
        self.shape.collision_type = 1
        space.add(self.body, self.shape)

    def reset(self):
        self.body.position = 100, 100
        self.body.velocity = 0, 0
        self.body.angle = 0
        self.body._set_angular_velocity(0)

    def accelerate(self, accel_const):
        # Accelerate in the direction the car is facing
        angle = self.body.angle
        self.body.velocity += accel_const*math.cos(angle), accel_const*math.sin(angle)

    def decelerate(self, accel_const):
        # Decelerate in the direction of the velocity vector
        x = self.body.velocity[0]
        y = self.body.velocity[1]
        angle = math.atan2(y, x)  # angle of the direction of velocity
        self.body.velocity += accel_const*math.cos(angle), accel_const*math.sin(angle)

    def get_velo(self):
        x=self.body.velocity[0]
        y=self.body.velocity[1]
        magnitude = math.sqrt(x**2 + y**2)

        return magnitude

    def set_velocity(self, x, y):
        self.body.velocity = x, y

    def turn(self, direction, turn_speed=math.pi/36):
        self.body._set_angular_velocity(0)
        self.body.angle = (self.body.angle + turn_speed*direction)

    def update_sight(self, wall):

        pi = math.pi

        directions = [0, pi/4, 2*pi/4, 3*pi/4, pi, 5*pi/4, 6*pi/4, 7*pi/4]

        for d in directions:

            angle = self.body.angle + d  # direction relative to directly in fron tof the car

            segment_start = self.body.position
            segment_end = tuple(map(sum, zip(self.body.position, SIGHT_LENGTH*np.array([math.cos(angle), math.sin(angle)]))))

            info = wall.shape.segment_query(segment_start, segment_end)
            self.sights['{0:.2f}'.format(d)] = (info.point if info.shape is not None else None)

            print(self.sights)

        return self.sights


class Wall():
    def __init__(self):
        self.body = pm.Body(body_type = pm.Body.STATIC)

        self.shape = pm.Segment(self.body, (50,50), (50,200),5)
        self.shape.collision_type = 2
        space.add(self.body, self.shape)


def collide(arbiter, space, data):
    print("Collision!")
    return True

def main():

    pg.init()
    clock = pg.time.Clock()
    draw_options = util.DrawOptions(screen)

    car = Car()
    wall = Wall()

    acceleration = 15
    friction = 10
    max_speed = 60

    handler = space.add_collision_handler(1, 2)
    handler.begin = collide

    running = True
    # Run the game
    while running:
        for event in pg.event.get():
            if(
                event.type == pg.QUIT
                or event.type == pg.KEYDOWN
                and (event.key in [pg.K_ESCAPE, pg.K_q])
            ):
                running = False

        keys = pg.key.get_pressed()
        if keys[pg.K_r]:
            car.reset()

        velo = car.get_velo()
        if keys[pg.K_UP]:
            car.accelerate(acceleration)
        elif keys[pg.K_DOWN]:
            car.accelerate(-acceleration)
        else:
            if velo > 0:
                car.decelerate(-friction)
            if (velo > 0) & (velo < 10):  # Prevent persistent "drifting" of the car
                car.set_velocity(0, 0)

        if keys[pg.K_LEFT]:
            car.turn(-1)
        elif keys[pg.K_RIGHT]:
            car.turn(1)

        car.update_sight(wall)
        
        screen.fill(pg.Color("white"))
        space.debug_draw(draw_options)
        pg.display.flip()
        
        fps = 60
        dt = 1.0 / fps
        space.step(dt)

        clock.tick(fps)

    pg.quit()
    

if __name__ == "__main__":
    main()