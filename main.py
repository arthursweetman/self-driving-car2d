import time

import pandas as pd
import numpy as np
import pymunk as pm
import pygame as pg
import math

import pymunk.pygame_util as util
from pymunk.vec2d import Vec2d


width, height = 690, 600
screen = pg.display.set_mode((width, height))

# Declare Pymunk Space
space = pm.Space()
space.gravity = 0, 0

SIGHT_LENGTH = 1000

class Car():
    def __init__(self):
        self.body = pm.Body()
        self.body.position = 300,500
        self.body.velocity = 0,0
        self.body.angle = 3*math.pi/2
        self.sights = {}

        w, h = 30, 20
        self.shape = pm.Poly.create_box(self.body, (w, h))
        self.shape.mass = 10
        self.shape.collision_type = 1
        self.shape.filter = pm.ShapeFilter(group = 1)
        space.add(self.body, self.shape)

        self.start_time = time.time()
        self.end_time = None
        self.finish_time = None

    def reset(self, hard: bool):
        self.body.position = 300, 500
        self.body.velocity = 0, 0
        self.body.angle = 3 * math.pi / 2
        self.body._set_angular_velocity(0)

        if hard:
            self.start_time = time.time()
            self.end_time = None
            self.finish_time = None

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

    def update_sight(self, surface):

        directions = np.linspace(0, 2*math.pi, 32)  # Generate 32 diffrent angles of sight evenly distributed around the car

        for d in directions:

            angle = self.body.angle + d  # direction relative to directly in front of the car

            segment_start = (self.body.position[0], self.body.position[1])
            segment_end = self.body.position + (SIGHT_LENGTH*math.cos(angle), SIGHT_LENGTH*math.sin(angle))
            segment_end = (segment_end[0], segment_end[1])

            filter = pm.ShapeFilter(group=1)
            info = space.segment_query_first(segment_start, segment_end, radius=1, shape_filter=filter)
            self.sights['{0:.2f}'.format(d)] = (info.point if info is not None else None)

            pg.draw.line(surface, (255,0,0,75), segment_start, segment_end if info is None else info.point)

        return self.sights

    def finish(self, arbiter, space, data):
        self.end_time = time.time()
        self.finish_time = self.end_time - self.start_time
        print(f"Completed course in {self.finish_time} seconds.")
        self.reset(hard=True)
        return True

    def wall_collision(self, arbiter, space, data):
        self.reset(hard=False)
        return True


class Wall():
    def __init__(self, a, b, radius=5, collision_type=2, group = 2):
        self.body = pm.Body(body_type = pm.Body.STATIC)

        self.shape = pm.Segment(self.body, a, b,radius)
        self.shape.collision_type = collision_type
        self.shape.filter = pm.ShapeFilter(group = group)
        space.add(self.body, self.shape)


def main():

    pg.init()
    clock = pg.time.Clock()
    draw_options = util.DrawOptions(screen)

    car = Car()

    segments1 = [(250, 500),(250, 100)]
    wall1 = Wall(*segments1)
    segments2 = [(350, 500),(350, 100)]
    wall2 = Wall(*segments2)
    finishLine = [(250, 100),(350,100)]
    finish = Wall(*finishLine, radius=10, collision_type=3, group=3)

    acceleration = 15
    friction = 10
    max_speed = 60

    hit_wall = space.add_collision_handler(1, 2)
    hit_wall.begin = car.wall_collision

    cross_finish = space.add_collision_handler(1, 3)
    cross_finish.begin = car.finish

    start_time = time.time()
    running = True
    # Run the game
    while running:
        screen.fill(pg.Color("white"))
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

        car.update_sight(screen)

        space.debug_draw(draw_options)
        pg.display.flip()
        
        fps = 60
        dt = 1.0 / fps
        space.step(dt)

        clock.tick(fps)

    pg.quit()
    

if __name__ == "__main__":
    main()