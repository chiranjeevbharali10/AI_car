# main.py
import pygame
import math
import os
import json
from car import Car
from train import run_training

pygame.init()
screen = pygame.display.set_mode((1000, 800))
font = pygame.font.SysFont("Arial", 30)

# button
button_rect = pygame.Rect(200, 700, 600, 80)
text_surface = font.render("START TRAINING", True, (0, 0, 0))
text_rect = text_surface.get_rect(center=(button_rect.centerx, button_rect.centery))

# road + car
road_surface = pygame.Surface((1000, 800), pygame.SRCALPHA)
car = None  # will be created after first click
state = "setup"
running = True
last_mouse = None
first_click = False
start_training = False
goal_pos = None

# Instructions
instruction_font = pygame.font.SysFont("Arial", 20)
instructions = [
    "LEFT CLICK & DRAG: Draw the road",
    "RIGHT CLICK: Set goal point (red circle)",
    "Click START TRAINING when ready"
]

while running:
    screen.fill((144, 238, 144))
    screen.blit(road_surface, (0, 0))
    
    # Draw button
    button_color = (100, 200, 100) if button_rect.collidepoint(pygame.mouse.get_pos()) else (211, 211, 211)
    pygame.draw.rect(screen, button_color, button_rect)
    pygame.draw.rect(screen, (0, 0, 0), button_rect, 3)
    screen.blit(text_surface, text_rect)
    
    # Draw instructions
    for i, instruction in enumerate(instructions):
        inst_surface = instruction_font.render(instruction, True, (255, 255, 255))
        inst_bg = pygame.Surface((inst_surface.get_width() + 10, inst_surface.get_height() + 5))
        inst_bg.fill((0, 0, 0))
        inst_bg.set_alpha(180)
        screen.blit(inst_bg, (10, 10 + i * 30))
        screen.blit(inst_surface, (15, 12 + i * 30))
    
    # Draw goal if set
    if goal_pos:
        pygame.draw.circle(screen, (255, 0, 0), goal_pos, 30, 3)
        goal_text = instruction_font.render("GOAL", True, (255, 0, 0))
        screen.blit(goal_text, (goal_pos[0] - 20, goal_pos[1] - 50))
    
    mouse_pos = pygame.mouse.get_pos()
    mouse_click = pygame.mouse.get_pressed()
    
    if state == "setup":
        # Left click to draw road
        if mouse_click[0] and not button_rect.collidepoint(mouse_pos):
            x, y = mouse_pos
            pygame.draw.circle(road_surface, (0, 0, 0), (x, y), 50)
            
            if last_mouse is not None:
                # Draw line between points for smooth road
                pygame.draw.line(road_surface, (0, 0, 0), last_mouse, (x, y), 100)
                dx, dy = x - last_mouse[0], y - last_mouse[1]
                if car:
                    car.rotate_towards(dx, dy)
            
            last_mouse = (x, y)
            
            if not first_click:
                car = Car(x, y)  # create car at first brush click
                first_click = True
        else:
            last_mouse = None
        
        # Right-click to set goal point
        if mouse_click[2]:
            gx, gy = mouse_pos
            goal_pos = (gx, gy)
            print(f"Goal set at: ({gx}, {gy})")
        
        # Start training button
        if button_rect.collidepoint(mouse_pos) and mouse_click[0]:
            if not car:
                print("ERROR: Please draw a road and create a starting point first!")
            elif not goal_pos:
                print("ERROR: Please set a goal point (right-click)!")
            else:
                # Save the drawn road as an image for training
                base_dir = os.path.dirname(__file__)
                track_path = os.path.join(base_dir, "track.png")
                
                try:
                    pygame.image.save(road_surface, track_path)
                    print(f"Saved track to: {track_path}")
                    
                    # Save starting pose metadata
                    meta_path = os.path.join(base_dir, "track_meta.json")
                    start_meta = {
                        "start_x": car.x,
                        "start_y": car.y,
                        "start_angle": car.angle,
                        "goal_x": goal_pos[0],
                        "goal_y": goal_pos[1]
                    }
                    with open(meta_path, "w") as f:
                        json.dump(start_meta, f)
                    print(f"Saved start pose to: {meta_path}")
                    
                    start_training = True
                    running = False
                except Exception as e:
                    print(f"Failed to save track at {track_path}: {e}")
    
    # Draw car if it exists
    if car:
        car.draw(screen)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    pygame.display.update()

pygame.quit()

if start_training:
    print("\n" + "="*60)
    print("Starting NEAT Training...")
    print("="*60 + "\n")
    
    # Run NEAT training with 50 generations
    run_training(generations=50)
