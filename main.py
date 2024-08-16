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

    def update_pos(self, x, y):
        self.body.position += x,y


def main():

    pg.init()
    clock = pg.time.Clock()
    draw_options = util.DrawOptions(screen)

    car = Car()

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
            car.update_pos(0,-5)
        if keys[pg.K_DOWN]:
            car.update_pos(0,5)
        if keys[pg.K_LEFT]:
            car.update_pos(-5,0)
        if keys[pg.K_RIGHT]:
            car.update_pos(5,0)
        
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