import pygame
import os
import json

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import random

import numpy as np

from collections import deque

from car import Car


######## this whole thing is just completely defining the NN 
class CarNN(nn.Module):
    def __init__(self , input_size = 7 , hidden_size = 64 , output_size =3): 

        super(CarNN , self).__init__()
        self.fc1 = nn.Linear(input_size , hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)

    def forward(self , x): 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
# -------- REPLAY BUFFER ---- ### 
# -- Its basically gonna store the prev values and delete the first value and add new ones and so on 


class ReplayBuffer : 
    def __init__(self , capacity = 1000): 
        self.buffer = deque(maxlen = capacity) 

    def push(self , state , action , reward , next_state , done): 
        self.buffer.append((state , action , reward , next_state , done))

    def sample(self , batch_size):
        batch = random.sample(self.buffer , batch_size)

        s , a , r , ns , d  = zip(*batch)

        return np.array(s), np.array(a) , np.array(r) ,np.array(ns) , np.array(d)

    def __len__(self): 
        return len(self.buffer)


#--- DNQ Agent ----- 


class DQNAgent: 

    def __init__(self , input_size = 7 , lr = 0.001 , gamma = 0.99 , epsilon = 1.0 , epsilon_decay = 0.995 , epsilon_min = 0.01):
        self.model = CarNN(input_size=input_size)
        self.target_model = CarNN(input_size=input_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.replay_buffer = ReplayBuffer()
        self.batch_size = 64



    def select_action(self , state): 
        state_tensor = torch.tensor(state, dtype = torch.float32)
        if random.random() < self.epsilon: 
            return random.randint(0,2)
        with torch.no_grad(): 
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item() 


    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return  
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Compute current Q values
        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())







def run_episode(agent, render=False):
    pygame.init()
    # Load drawn track if available; fallback to blank surface
    try:
        base_dir = os.path.dirname(__file__)
        track_path = os.path.join(base_dir, "track.png")
        # Load first (no convert), get size, set display, then convert with alpha
        loaded_image = pygame.image.load(track_path)
        width, height = loaded_image.get_width(), loaded_image.get_height()
        # Use double buffering and vsync to reduce flicker
        screen = pygame.display.set_mode((width, height), pygame.DOUBLEBUF | pygame.SCALED, vsync=1)
        road_image = loaded_image.convert_alpha()
        road = pygame.Surface((width, height), pygame.SRCALPHA)
        road.blit(road_image, (0, 0))
        print(f"Loaded track from: {track_path} ({width}x{height})")
        # Load starting pose if available
        start_x, start_y, start_angle = width // 2, height // 2, 0
        goal_x, goal_y = None, None
        meta_path = os.path.join(base_dir, "track_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                start_x = meta.get("start_x", start_x)
                start_y = meta.get("start_y", start_y)
                start_angle = meta.get("start_angle", start_angle)
                goal_x = meta.get("goal_x", None)
                goal_y = meta.get("goal_y", None)
                print(f"Loaded start pose from meta: x={start_x}, y={start_y}, angle={start_angle}")
            except Exception as me:
                print(f"Failed to read track_meta.json: {me}")
    except Exception as e:
        print(f"Could not load track.png, using blank surface. Error: {e}")
        width, height = 800, 600
        screen = pygame.display.set_mode((width, height), pygame.DOUBLEBUF | pygame.SCALED, vsync=1)
        road = pygame.Surface((width, height), pygame.SRCALPHA)
        start_x, start_y, start_angle = width // 2, height // 2, 0
    clock = pygame.time.Clock()

    # If start is on a wall (black), nudge to nearest free pixel
    def find_safe_start(x, y, max_radius=50):
        road_color = (0, 0, 0, 255)
        ix, iy = int(x), int(y)
        # Prefer starting ON road; if already on road, keep
        if 0 <= ix < width and 0 <= iy < height and road.get_at((ix, iy)) == road_color:
            return x, y
        for r in range(1, max_radius + 1):
            for dx in (-r, 0, r):
                for dy in (-r, 0, r):
                    nx, ny = ix + dx, iy + dy
                    if 0 <= nx < width and 0 <= ny < height and road.get_at((nx, ny)) == road_color:
                        return float(nx), float(ny)
        return x, y

    safe_x, safe_y = find_safe_start(start_x, start_y)
    if (safe_x, safe_y) != (start_x, start_y):
        print(f"Adjusted start to road: x={safe_x}, y={safe_y}")

    car = Car(safe_x, safe_y)
    car.angle = start_angle
    total_reward = 0
    steps = 0
    done = False
    debug_overlay = True
    goal = (goal_x, goal_y) if goal_x is not None and goal_y is not None else None

    while not done and steps < 500:
        state = car.get_state(road)
        action = agent.select_action(state)

        # Apply action
        if action == 0: car.move_forward(2)
        elif action == 1: car.angle += 5
        elif action == 2: car.angle -= 5

        # Reward: on-road bonus + progress toward goal, off-road penalty
        color = road.get_at((int(car.x), int(car.y)))
        if color == (0,0,0,255):  # on-road
            reward = 0.05
            if goal:
                # dense shaping: reduce distance to goal
                dx = goal[0] - car.x
                dy = goal[1] - car.y
                dist = (dx*dx + dy*dy) ** 0.5
                # compute next distance for shaping
                ndx = goal[0] - (car.x)
                ndy = goal[1] - (car.y)
                next_dist = (ndx*ndx + ndy*ndy) ** 0.5
                progress = (dist - next_dist) * 0.01
                reward += progress
                # success condition
                if dist < 20:
                    reward += 5.0
                    done = True
        else:
            reward = -1
            done = True

        total_reward += reward

        # next state
        next_state = car.get_state(road)

        agent.replay_buffer.push(state, action, reward, next_state, done)
        agent.train_step()

        # update screen
        if render:
            # Handle events to keep window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_d:
                        debug_overlay = not debug_overlay
            screen.fill((144, 238, 144))
            screen.blit(road,(0,0))
            car.draw(screen)
            if debug_overlay:
                # draw sensors and collision dot
                car.draw_sensors(screen, road)
                cx, cy = int(car.x), int(car.y)
                if 0 <= cx < width and 0 <= cy < height:
                    pygame.draw.circle(screen, (0, 255, 0), (cx, cy), 3)
                if goal:
                    pygame.draw.circle(screen, (255, 0, 0), (int(goal[0]), int(goal[1])), 6)
            pygame.display.flip()
            clock.tick(60)

        steps += 1

    pygame.quit()
    return total_reward

# ----- Training Loop -----
if __name__ == "__main__":
    agent = DQNAgent()

    for episode in range(1000):
        reward = run_episode(agent, render=False)
        if episode % 10 == 0:
            agent.update_target()
            print(f"Episode {episode}, Reward: {reward:.2f}, Epsilon: {agent.epsilon:.2f}")
        


















'''

import torch.optim as optim 

model = CarNN() 

optimizer = optim.Adam(model.parameters() , lr = 0.001)
loss_fn = nn.MSELOSS() ##or USE RL loss 

def select_action(state_tensor): 
    wiht torch.no_grad(): 
        output = model(state_tensor)
        action = torch.argmax(output).item()

    return action

for episode in range(100):
    car = Car(400, 300)
    road = pygame.Surface((800, 600), pygame.SRCALPHA)
    
    for step in range(500):
        state = car.get_state(road)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action = select_action(state_tensor)

        # Apply action
        if action == 0: car.move_forward(2)
        elif action == 1: car.angle += 5
        elif action == 2: car.angle -= 5
'''
