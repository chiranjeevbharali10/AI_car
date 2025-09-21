# main.py
import pygame
import math
from car import Car

pygame.init()
screen = pygame.display.set_mode((1000, 800))
font = pygame.font.SysFont("Arial", 30)

# button
button_rect = pygame.Rect(200, 700, 700, 100)
text_surface = font.render("START", True, (0, 0, 0))
text_rect = text_surface.get_rect(center=(200 + 700//2, 700 + 100//2))

# road + car
road_surface = pygame.Surface((1000, 800), pygame.SRCALPHA)
car = None  # will be created after first click

state = "setup"
running = True
last_mouse = None
first_click = False

while running:
    screen.fill((144, 238, 144))
    screen.blit(road_surface, (0, 0))
    pygame.draw.rect(screen, (211, 211, 211), button_rect)
    screen.blit(text_surface, text_rect)

    mouse_pos = pygame.mouse.get_pos()
    mouse_click = pygame.mouse.get_pressed()

    if state == "GO":
        if car:
            car.draw(screen)
            car.move_forward(step=2)  # <-- here AI or NN will later control movement

    if state == "setup":
        if mouse_click[0]:
            x, y = mouse_pos
            pygame.draw.circle(road_surface, (0, 0, 0), (x, y), 50)

            if last_mouse is not None:
                dx, dy = x - last_mouse[0], y - last_mouse[1]
                if car:
                    car.rotate_towards(dx, dy)
            last_mouse = (x, y)

            if not first_click:
                car = Car(x, y)  # create car at first brush click
                first_click = True

        if button_rect.collidepoint(mouse_pos) and mouse_click[0]:
            state = "GO"

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.update()

pygame.quit()







