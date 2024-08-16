import pandas as pd
import numpy as np
import pymunk as pm
import pygame as pg

import pymunk.pygame_util as util


width, height = 690, 600
screen = pg.display.set_mode((width, height))

# Declare Pymunk Space
space = pm.Space()
space.gravity = 0, 0

class Car():
    def __init__(self):
        self.body = pm.Body()
        self.body.position = 100,100
        self.body.velocity = 0,0

        w, h = 20, 30
        self.shape = pm.Poly.create_box(self.body, (w, h))
        self.shape.mass = 10
        space.add(self.body, self.shape)

    def update_velo(self, x, y):
        self.body.velocity += x, y

    def get_velo(self):
        return self.body.velocity

class Wall():
    def __init__(self):
        self.body = pm.Body(body_type = pm.Body.STATIC)

        self.shape = pm.Segment(self.body, (50,50), (50,200),5)
        space.add(self.body, self.shape)

def main():

    pg.init()
    clock = pg.time.Clock()
    draw_options = util.DrawOptions(screen)

    car = Car()
    wall = Wall()

    acceleration = 15
    friction = 5
    max_speed = 60

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
        if keys[pg.K_UP]:
            car.update_velo(0, -acceleration)
        elif keys[pg.K_DOWN]:
            car.update_velo(0, acceleration)
        else:
            velo_x = car.get_velo()[1]
            if velo_x > 0:
                car.update_velo(0, -friction)
            elif velo_x < 0:
                car.update_velo(0, friction)

        if keys[pg.K_LEFT]:
            car.update_velo(-acceleration, 0)
        elif keys[pg.K_RIGHT]:
            car.update_velo(acceleration, 0)
        else:
            velo_y = car.get_velo()[0]
            if velo_y > 0:
                car.update_velo(-friction, 0)
            elif velo_y < 0:
                car.update_velo(friction, 0)

        
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