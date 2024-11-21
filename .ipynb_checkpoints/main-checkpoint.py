import pandas as pd
import numpy as np
import pymunk as pm
import pygame as pg

import pymunk.pygame_util as util


width, height = 690, 600

def main():
    pg.init()
    screen = pg.display.set_mode((width, height))
    clock = pg.time.Clock()
    running = True
    # font = pg.font.SysFont("Arial", 16)

    # Physics
    space = pm.Space()
    space.gravity = 0,0
    draw_options = util.DrawOptions(screen)
    
    body = pm.Body()
    body.position = 100,100
    body.velocity = 0,0

    w, h = 20, 30
    poly = pm.Poly.create_box(body, (w, h))
    poly.mass = 10
    space.add(body, poly)

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
            body.position += 0,-5
        if keys[pg.K_DOWN]:
            body.position += 0,5
        if keys[pg.K_LEFT]:
            body.position += -5,0
        if keys[pg.K_RIGHT]:
            body.position += 5,0
        
        screen.fill(pg.Color("black"))
        space.debug_draw(draw_options)

        pg.display.flip()
        
        fps = 60
        dt = 1.0 / fps
        space.step(dt)

        clock.tick(fps)

    pg.quit()
    

if __name__ == "__main__":
    main()