# 🚗 NEAT Self-Driving Car Simulator

An AI-powered car simulation where neural networks learn to navigate custom tracks using **NEAT (NeuroEvolution of Augmenting Topologies)**. Draw your own track, set a goal, and watch 50 cars evolve and learn to drive!

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.5.0-green.svg)
![NEAT](https://img.shields.io/badge/NEAT--Python-0.92-orange.svg)

```
Generation 1:  🚗💥💥💥💥  (Most cars crash immediately)
Generation 10: 🚗→🚗→💥🚗  (Some survive longer)
Generation 30: 🚗→→→→🚗🎯  (Cars reach the goal!)
```

## 📋 Requirements

- Python 3.9 or higher
- Pygame 2.5.0+
- NEAT-Python 0.92+


```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
neat-car-simulator/
│
├── main.py              # Main entry point - track drawing interface
├── train.py             # NEAT training logic and parallel simulation
├── car.py               # Car class with sensors and movement
├── neat_config.txt      # NEAT algorithm configuration
├── requirements.txt     # Python dependencies
├── README.md           # This file
│
└── Generated Files (after first run):
    ├── track.png        # Your drawn track image
    ├── track_meta.json  # Starting position and goal metadata
    └── best_genome.pkl  # Saved best-performing neural network
```

## 🎮 How to Use

### Step 1: Draw Your Track

```bash
python main.py
```

1. **Left-click and drag** to draw the road (black)
2. **Right-click** to set the goal point (red circle)
3. Click **"START TRAINING"** button when ready

### Step 2: Watch the Evolution

- The training window will open automatically
- 50 cars spawn at the starting position
- Cars evolve over 50 generations
- Progress is shown on screen in real-time

### Step 3: View Results

After training completes:
- The best-performing car will be demonstrated
- Network saved to `best_genome.pkl`
- Press ESC or close window to exit

## 🧠 How It Works

### Neural Network Inputs (13 total)

```python
Inputs = [
    x_position,          # Car's X coordinate (normalized)
    y_position,          # Car's Y coordinate (normalized)
    angle,               # Car's current rotation angle
    speed,               # Car's current speed
    angle_to_goal,       # Relative angle to goal (-180° to 180°)
    distance_to_goal,    # Euclidean distance to goal
    sensor_-90°,         # Left wall distance
    sensor_-45°,         # Front-left wall distance
    sensor_-22.5°,       # Slight left wall distance
    sensor_0°,           # Forward wall distance
    sensor_22.5°,        # Slight right wall distance
    sensor_45°,          # Front-right wall distance
    sensor_90°           # Right wall distance
]
```

### Neural Network Outputs (2 total)

```python
Outputs = [
    steering,      # -1 (turn left) to +1 (turn right)
    acceleration   # -1 (brake) to +1 (accelerate)
]
```

### Fitness Function

```python
fitness = survival_bonus + goal_proximity + goal_bonus + movement_bonus

Where:
  - survival_bonus = steps_alive × 0.1          (0-100 points)
  - goal_proximity = max(0, 1000 - distance)    (0-1000 points)
  - goal_bonus = 5000 if reached else 0         (0 or 5000 points)
  - movement_bonus = distance_traveled × 0.05   (rewards movement)
```

## ⚙️ Configuration

Edit `neat_config.txt` to customize evolution:

### Key Parameters

```ini
pop_size = 50              # Number of cars per generation
num_inputs = 13            # Input neurons (must match car sensors)
num_outputs = 2            # Output neurons (steering, acceleration)
num_hidden = 2             # Initial hidden layer neurons
fitness_threshold = 5000   # Stop training if fitness reached

# Mutation rates
conn_add_prob = 0.5        # Probability of adding connection
node_add_prob = 0.2        # Probability of adding neuron
weight_mutate_rate = 0.8   # Probability of weight mutation
```

## 🎨 Customization

### Change Number of Generations

In `main.py`, modify:
```python
run_training(generations=50)  # Change to desired number
```

### Adjust Car Speed

In `train.py`, modify:
```python
speed = max(0, acceleration * 3)  # Change multiplier (3)
```

### Modify Sensor Range

In `car.py`, modify:
```python
def get_sensor_data(self, road_surface, max_distance=200):  # Change 200
```

### Add More Sensors

In `car.py`, modify:
```python
directions = [-90, -45, -22.5, 0, 22.5, 45, 90]  # Add more angles
```

Then update `neat_config.txt`:
```ini
num_inputs = 13  # Update based on new sensor count
```

## 🐛 Troubleshooting

### Error: "Expected X inputs, got Y"

**Solution:** Your `neat_config.txt` doesn't match your code.

```bash
# Count inputs in car.py get_state() method
# Update neat_config.txt:
num_inputs = <your_count>
```

### Training Too Slow

**Solutions:**
1. Reduce population: `pop_size = 30`
2. Reduce max steps: `max_steps = 500` (in `train.py`)
3. Disable rendering: Set `render=False` in debugging mode

### Pygame Display Error

**Solution:** Ensure display is initialized before `convert_alpha()`:
```python
pygame.init()
screen = pygame.display.set_mode((width, height))
# NOW load images with convert_alpha()
```

## 📊 Performance Tips

### Fast Training
- Use simpler tracks for testing
- Reduce `max_steps` in `eval_genomes()`
- Lower FPS: `self.clock.tick(120)` instead of 60

### Better Results
- Increase population: `pop_size = 100`
- Add more generations: `run_training(generations=100)`
- Fine-tune fitness function weights

## 🤝 Contributing

Contributions are welcome! Here are some ideas:

- [ ] Add checkpoint system for multi-goal tracks
- [ ] Implement lap timing and leaderboards
- [ ] Add obstacles and moving targets
- [ ] Create pre-made tracks library
- [ ] Add replay system for best runs
- [ ] Implement save/load training progress

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.


