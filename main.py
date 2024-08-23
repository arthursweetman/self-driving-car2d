import time

import pandas as pd
import numpy as np
import pymunk as pm
import pygame as pg
import math

import pymunk.pygame_util as util
from pymunk.vec2d import Vec2d

import model

SCREEN_WIDTH, SCREEN_HEIGHT = 800, 800

ACCEL = 15
FRICTION = 10

class Car():
    def __init__(self, space):
        self.body = pm.Body()
        self.body.position = SCREEN_WIDTH/2, SCREEN_HEIGHT/2
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

        self.observation = []
        self.outputs = []

    def turn(self, direction, turn_speed=math.pi/36):
        self.body._set_angular_velocity(0)
        self.body.angle = (self.body.angle + turn_speed*direction)

    def update_sight(self, space, finish_wall, SIGHT_LENGTH):

        # self.observation are reset in this method

        directions = np.linspace(0, 31*math.pi/16, 32)  # Generate 32 diffrent angles of sight evenly distributed around the car
        self.observation = []
        sensor_distances = []
        wall_types = []

        for d in directions:

            angle = self.body.angle + d  # direction relative to directly in front of the car

            segment_start = (self.body.position[0], self.body.position[1])
            segment_end = self.body.position + (SIGHT_LENGTH*math.cos(angle), SIGHT_LENGTH*math.sin(angle))
            segment_end = (segment_end[0], segment_end[1])

            filter = pm.ShapeFilter(group=1)
            info = space.segment_query_first(segment_start, segment_end, radius=1, shape_filter=filter)
            self.sights['{0:.2f}'.format(d)] = (info.point if info is not None else None)

            if info is not None:
                vec = info.point - self.body.position
                sensor_distances.append(abs(vec))
                wall_types.append(1 if info.shape == finish_wall.shape else -1)
            else:
                sensor_distances.append(0)
                wall_types.append(0)

        self.observation = sensor_distances + wall_types

        return self.sights

    def draw_sight(self, surface, sight_length):
        directions = np.linspace(0, 31 * math.pi / 16,
                                 32)  # Generate 32 diffrent angles of sight evenly distributed around the car
        for d in directions:
            angle = self.body.angle + d  # direction relative to directly in front of the car

            segment_start = (self.body.position[0], self.body.position[1])
            segment_end = self.body.position + (sight_length*math.cos(angle), sight_length*math.sin(angle))
            segment_end = (segment_end[0], segment_end[1])
            poc = self.sights['{0:.2f}'.format(d)]

            pg.draw.line(surface, (255, 0, 0, 75), segment_start, segment_end if poc is None else poc)

    def update_info(self):
        self.observation = self.observation + list(self.body.velocity)
        self.observation.append(self.body.velocity.angle)
        self.observation.append(self.body.angle)

    def drive_frame(self):
        action = model.step(self.observation)
        return action


class Wall():
    def __init__(self, center, length, angle, space, radius=2, collision_type=2, group = 2):
        self.body = pm.Body(body_type = pm.Body.KINEMATIC)
        self.body.position = center
        self.body.angle = angle

        self.shape = pm.Segment(self.body, (-length/2, 0), (length/2, 0), radius)
        self.shape.collision_type = collision_type
        self.shape.filter = pm.ShapeFilter(group = group)

        space.add(self.body, self.shape)


def game():
    screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    gamestate = GameState()

    pg.init()
    pg.display.set_caption("Car")
    clock = pg.time.Clock()
    draw_options = util.DrawOptions(screen)

    running = True
    terminal_state = False
    # Run the game
    while running and not terminal_state:
        screen.fill(pg.Color("white"))
        for event in pg.event.get():
            if(
                event.type == pg.QUIT
                or event.type == pg.KEYDOWN
                and (event.key in [pg.K_ESCAPE, pg.K_q])
            ):
                running = False

        gamestate.game_draw(screen)  # I believe the drawn sights will be a frame behind since this happens before game_step

        keys = pg.key.get_pressed()
        action=[keys[pg.K_UP], keys[pg.K_DOWN], keys[pg.K_LEFT], keys[pg.K_RIGHT]]

        terminal_state = gamestate.game_step(action, pygame_draw_options=draw_options)

        fps = 60
        clock.tick(fps)

    pg.quit()


scroll = [0,0]
class GameState():
    def __init__(self):
        # Initialize the space and the car
        self.terminal_state = False
        self.space = pm.Space()
        self.car = Car(self.space)
        self.sight_length = 1000

        # Initialize the walls and the finish line
        self.wall1 = Wall((250, 300), 400, math.pi/2, space=self.space)
        self.wall2 = Wall((350, 300), 400, math.pi/2, space=self.space)
        self.finish_line = Wall((300, 100), 100, 0, space=self.space, radius=2, collision_type=3, group=3)

        # Collision handlers in the space
        self.hit_wall = self.space.add_collision_handler(1, 2)
        self.hit_wall.begin = self.wall_collision

        self.cross_finish = self.space.add_collision_handler(1, 3)
        self.cross_finish.begin = self.finish_collision

        # Record the start time
        self.start_time = time.time()

        self.car.update_sight(self.space, self.finish_line, self.sight_length)
        self.car.update_info()

        self.acceleration = 15
        self.friction = 10

    def game_step(self, action, pygame_draw_options = None):
        if action is None:
            action = self.car.drive_frame()
        print(action)

        self.car.update_sight(self.space, self.finish_line, self.sight_length)
        self.car.update_info()

        velo = abs(self.wall1.body.velocity)
        if action[0]:
            # Accelerate in the direction of the car
            accel_vec = (ACCEL*math.cos(self.car.body.angle), ACCEL*math.sin(self.car.body.angle))
            self.wall1.body.velocity -= accel_vec
            self.wall2.body.velocity -= accel_vec
            self.finish_line.body.velocity -= accel_vec
        elif action[1]:
            accel_vec = (ACCEL*math.cos(self.car.body.angle), ACCEL*math.sin(self.car.body.angle))
            self.wall1.body.velocity += accel_vec
            self.wall2.body.velocity += accel_vec
            self.finish_line.body.velocity += accel_vec
        else:
            if velo > 0:
                # Decelerate in direction of velocity vector
                angle = self.wall1.body.velocity.angle
                decel_vec = (FRICTION*math.cos(angle), FRICTION*math.sin(angle))
                self.wall1.body.velocity -= decel_vec
                self.wall2.body.velocity -= decel_vec
                self.finish_line.body.velocity -= decel_vec
            if (velo > 0) & (velo < 10):  # Prevent persistent "drifting" of the car
                self.wall1.body.velocity = 0, 0
                self.wall2.body.velocity = 0, 0
                self.finish_line.body.velocity = 0, 0

        if action[2]:
            self.car.turn(-1)
        elif action[3]:
            self.car.turn(1)

        if pygame_draw_options is not None:
            self.space.debug_draw(pygame_draw_options)
            pg.display.flip()

        fps = 60
        dt = 1.0 / fps
        self.space.step(dt)

        return self.terminal_state

    def game_draw(self, surface):
        self.car.draw_sight(surface, self.sight_length)

    def wall_collision(self, arbiter, space, data):
        self.terminal_state = True
        return True

    def finish_collision(self, arbiter, space, data):
        self.terminal_state = True
        return True


if __name__ == "__main__":
    game()