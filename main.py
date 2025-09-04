import pygame  
import math

pygame.init() 

##SCREEN SIZE AND RES
screen = pygame.display.set_mode((1000,800))

#mouse 

mouse_2 = pygame.mouse.get_pos()
mouse_click1 = pygame.mouse.get_pressed() 
#FONT 
font = pygame.font.SysFont("Arial" , 30)
text_surface = font.render("START", True , (0,0,0))
text_rect = text_surface.get_rect(center=(200 + 700//2, 700 + 100//2))


#LOADING THE CAR IMAGE
car_img = pygame.image.load('car_image.png').convert_alpha() 
car_small = pygame.transform.scale(car_img ,(100,100))
road_surface = pygame.Surface((1000, 800), pygame.SRCALPHA)
#CAR SIZE 

car_width, car_height = car_small.get_size() 



button_rect = pygame.Rect(200, 700, 700, 100 )    
state = "setup"

first_click = False 
car_x , car_y = 0,0 
last_mouse = None 
running = True 
x =  0
y = 0 
while running : 
    screen.fill((144, 238, 144))
    screen.blit(road_surface, (0,0))
    pygame.draw.rect(screen,(211, 211, 211),button_rect) 
    screen.blit(text_surface , text_rect) 
    
   
       
    ##MOUSE POSITION HERE
    mouse_pos = pygame.mouse.get_pos()
    mouse_click = pygame.mouse.get_pressed()

       
        ##FONT HERE
    if state == "GO":
        box = pygame.Surface((30,60),pygame.SRCALPHA)
        pygame.draw.rect(box, (255, 0, 0), (0, 0, 30, 30))   # right half = red (front)
        pygame.draw.rect(box, (0, 0, 255), (0, 30, 30, 30))
        rotated_car = pygame.transform.rotate(box, -car_angle)  # negative because pygame’s y axis is inverted
        rect = rotated_car.get_rect(center=(car_x, car_y))
        screen.blit(rotated_car, rect.topleft)
              
        ##THIS IS WHERE IT STARTS  
    if state == "setup": 
        if mouse_click[0]:
            x,y = mouse_pos
            pygame.draw.circle(road_surface,(0,0,0),(x,y),50)

            #Rotates the car towards where the road is headed 
            if last_mouse is not None:
                dx, dy = x - last_mouse[0], y - last_mouse[1]
                if dx != 0 or dy != 0:
                    car_angle = math.degrees(math.atan2(-dy, dx))  # compute angle
            last_mouse = (x, y)

            #first cordinate of the Brush 
            if not first_click:
                car_x , car_y = mouse_pos 
                first_click = True
        if button_rect.collidepoint(mouse_pos) and mouse_click[0] :
            state ="GO"
    

    for event in pygame.event.get():
        if event.type == pygame.QUIT: 
            running = False 


    pygame.display.update()
pygame.quit()



















