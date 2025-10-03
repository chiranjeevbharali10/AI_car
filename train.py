import pygame
import neat
import os
import json
import math
from car import Car

class NEATTrainer:
    def __init__(self, track_path="track.png", meta_path="track_meta.json"):
        """Initialize the NEAT trainer with track and metadata"""
        self.track_path = track_path
        self.meta_path = meta_path
        
        # Load track image
        if not os.path.exists(track_path):
            raise FileNotFoundError(f"Track image not found: {track_path}")
        
        # Load start position and goal
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            self.start_x = meta.get('start_x', 100)
            self.start_y = meta.get('start_y', 100)
            self.start_angle = meta.get('start_angle', 0)
            self.goal_x = meta.get('goal_x', 900)
            self.goal_y = meta.get('goal_y', 700)
        
        # Pygame setup for visualization
        pygame.init()
        
        temp_surface = pygame.image.load(track_path)
        self.width = temp_surface.get_width()
        self.height = temp_surface.get_height()
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("NEAT Car Training - Parallel")
        
        self.road_surface = pygame.image.load(track_path).convert_alpha()
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        
        # Training statistics
        self.generation = 0
        self.best_fitness = 0
        self.best_genome = None
        self.cars_reached_goal = 0
        
        # Calculate initial distance for normalization
        self.initial_distance = math.sqrt((self.start_x - self.goal_x)**2 + 
                                         (self.start_y - self.goal_y)**2)

    def is_on_road(self, x, y):
        """Check if position (x, y) is on the black road"""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        color = self.road_surface.get_at((int(x), int(y)))
        return color == (0, 0, 0, 255)

    def calculate_fitness(self, car, steps, reached_goal, min_distance):
        """
        Ultra-aggressive fitness function that HEAVILY prioritizes reaching the goal.
        Only goal-reaching cars get elite fitness!
        """
        
        # If reached goal, fitness is MASSIVELY higher and based on speed
        if reached_goal:
            # Base goal bonus (huge!)
            base_goal_reward = 500000
            # Speed bonus: fewer steps = even higher fitness
            speed_bonus = max(0, (1500 - steps) * 200)
            return base_goal_reward + speed_bonus
        
        # If NOT reached goal, fitness is MUCH lower with steep gradient near goal
        progress_ratio = (self.initial_distance - min_distance) / self.initial_distance
        
        # Progressive distance reward but capped very low
        distance_component = progress_ratio * 2000  # Max 2000 (vs 500000+ for goal)
        
        # Minimal survival bonus
        survival_component = min(steps * 0.3, 300)
        
        # VERY steep bonus for getting extremely close (creates pressure to reach)
        close_bonus = 0
        if min_distance < 150:
            # Exponential bonus as we get closer
            proximity_factor = (150 - min_distance) / 150
            close_bonus = proximity_factor ** 3 * 3000  # Max ~3000
        
        # Penalty for barely moving
        total_movement = math.sqrt((car.x - self.start_x)**2 + (car.y - self.start_y)**2)
        movement_penalty = -200 if total_movement < 30 else 0
        
        fitness = distance_component + survival_component + close_bonus + movement_penalty
        
        return max(0, fitness)

    def eval_genomes(self, genomes, config):
        """Evaluate ALL genomes in parallel (all cars drive at once!)"""
        self.generation += 1
        
        # Create neural networks and cars for all genomes
        cars_data = []
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            car = Car(self.start_x, self.start_y, self.start_angle)
            cars_data.append({
                'genome_id': genome_id,
                'genome': genome,
                'net': net,
                'car': car,
                'alive': True,
                'steps': 0,
                'reached_goal': False,
                'min_distance': self.initial_distance,
                'stuck_counter': 0,
                'last_x': self.start_x,
                'last_y': self.start_y
            })
        
        max_steps = 2000  # Give more time to reach goal
        current_step = 0
        
        # Run simulation until all cars are done or max steps reached
        while current_step < max_steps:
            # Check for pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            
            # Count alive cars
            alive_count = sum(1 for c in cars_data if c['alive'])
            if alive_count == 0:
                break
            
            # Update each alive car
            for car_data in cars_data:
                if not car_data['alive']:
                    continue
                
                car = car_data['car']
                net = car_data['net']
                
                # Get sensor data
                state = car.get_state(self.road_surface)
                
                # Get neural network output
                output = net.activate(state)
                steering = output[0]
                acceleration = output[1]
                
                # Apply actions
                car.angle += steering * 5
                speed = max(0, acceleration * 3)
                car.move_forward(step=speed)
                
                # Check if car is stuck (not moving much)
                movement = math.sqrt((car.x - car_data['last_x'])**2 + 
                                   (car.y - car_data['last_y'])**2)
                if movement < 0.5:  # Very little movement
                    car_data['stuck_counter'] += 1
                    if car_data['stuck_counter'] > 150:  # Stuck for 150 steps
                        car_data['alive'] = False
                        continue
                else:
                    car_data['stuck_counter'] = 0
                
                car_data['last_x'] = car.x
                car_data['last_y'] = car.y
                
                # Track minimum distance to goal
                dist_to_goal = math.sqrt((car.x - self.goal_x)**2 + (car.y - self.goal_y)**2)
                car_data['min_distance'] = min(car_data['min_distance'], dist_to_goal)
                
                # Check if still on road
                if not self.is_on_road(car.x, car.y):
                    car_data['alive'] = False
                    continue
                
                # Check if reached goal
                if dist_to_goal < 30:
                    car_data['reached_goal'] = True
                    car_data['alive'] = False
                    continue
                
                car_data['steps'] += 1
            
            current_step += 1
            
            # RENDER (every 2 frames for performance)
            if current_step % 2 == 0:
                self.screen.fill((144, 238, 144))
                self.screen.blit(self.road_surface, (0, 0))
                
                # Draw goal
                pygame.draw.circle(self.screen, (255, 0, 0), 
                                 (int(self.goal_x), int(self.goal_y)), 30, 3)
                
                # Draw all alive cars
                for car_data in cars_data:
                    if car_data['alive']:
                        car_data['car'].draw(self.screen)
                
                # Display stats
                successful_cars = sum(1 for c in cars_data if c['reached_goal'])
                best_distance = min(c['min_distance'] for c in cars_data)
                best_progress = ((self.initial_distance - best_distance) / 
                               self.initial_distance) * 100
                
                info_text = [
                    f"Generation: {self.generation}",
                    f"Step: {current_step}/{max_steps}",
                    f"Alive: {alive_count}/{len(cars_data)}",
                    f"Reached goal this gen: {successful_cars}",
                    f"Total goals reached: {self.cars_reached_goal}",
                    f"Best progress: {best_progress:.1f}%",
                    f"Best fitness ever: {self.best_fitness:.0f}"
                ]
                
                y_offset = 10
                for text in info_text:
                    text_surface = self.font.render(text, True, (255, 255, 255))
                    text_bg = pygame.Surface((text_surface.get_width() + 10, 
                                            text_surface.get_height() + 5))
                    text_bg.fill((0, 0, 0))
                    text_bg.set_alpha(180)
                    self.screen.blit(text_bg, (10, y_offset))
                    self.screen.blit(text_surface, (15, y_offset + 2))
                    y_offset += 28
                
                pygame.display.flip()
                self.clock.tick(60)
        
        # Calculate fitness for all genomes
        for car_data in cars_data:
            fitness = self.calculate_fitness(
                car_data['car'], 
                car_data['steps'], 
                car_data['reached_goal'],
                car_data['min_distance']
            )
            car_data['genome'].fitness = fitness
            
            # Track statistics
            if car_data['reached_goal']:
                self.cars_reached_goal += 1
            
            # Track best genome
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_genome = car_data['genome']
                status = "REACHED GOAL" if car_data['reached_goal'] else "getting closer"
                progress = ((self.initial_distance - car_data['min_distance']) / 
                          self.initial_distance) * 100
                print(f"Gen {self.generation}: New best! Fitness: {fitness:.0f} ({status}, {progress:.1f}% progress)")

    def train(self, config_path="neat_config.txt", generations=50):
        """Run NEAT training"""
        # Load NEAT configuration
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        
        # Create population
        population = neat.Population(config)
        
        # Add reporters for statistics
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        
        # Run NEAT
        print(f"Starting NEAT training for UP TO {generations} generations...")
        print("All cars in each generation will run simultaneously!")
        print("Goal reward: 500,000+ | Distance progress: max ~5,000")
        print("Cars MUST reach goal to get high fitness!")
        print("Training will STOP as soon as a car reaches the goal!\n")
        
        # Custom training loop to stop when goal is reached
        for gen in range(generations):
            winner = population.run(self.eval_genomes, 1)
            
            # Check if any car reached the goal this generation
            if self.best_fitness >= 500000:  # Goal reached!
                print(f"\n🎉 SUCCESS! Goal reached in generation {self.generation}!")
                break
        else:
            print(f"\nReached max generations ({generations}) without reaching goal.")
            print("Consider adjusting track difficulty or NEAT parameters.")
        
        # Save the winner
        import pickle
        with open('best_genome.pkl', 'wb') as f:
            pickle.dump(winner, f)
        
        print(f"\nTraining complete! Best fitness: {self.best_fitness:.0f}")
        print(f"Total cars that reached goal: {self.cars_reached_goal}")
        print("Winner genome saved to best_genome.pkl")
        
        # Show winner performance
        print("\nRunning winner genome...")
        self.run_single_car(winner, config)
        
        pygame.quit()
        return winner

    def run_single_car(self, genome, config):
        """Run a single car for demonstration"""
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        car = Car(self.start_x, self.start_y, self.start_angle)
        
        max_steps = 2000
        steps = 0
        reached_goal = False
        min_distance = self.initial_distance
        
        print("\nPress ESC or close window to exit...")
        running = True
        
        while running and steps < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # Get state and action
            state = car.get_state(self.road_surface)
            output = net.activate(state)
            steering = output[0]
            acceleration = output[1]
            
            # Apply actions
            car.angle += steering * 5
            speed = max(0, acceleration * 3)
            car.move_forward(step=speed)
            
            # Check conditions
            if not self.is_on_road(car.x, car.y):
                print("Car went off road!")
                break
            
            dist_to_goal = math.sqrt((car.x - self.goal_x)**2 + (car.y - self.goal_y)**2)
            min_distance = min(min_distance, dist_to_goal)
            
            if dist_to_goal < 30:
                reached_goal = True
                print(f"Car reached the goal in {steps} steps!")
                break
            
            steps += 1
            
            # Render
            self.screen.fill((144, 238, 144))
            self.screen.blit(self.road_surface, (0, 0))
            pygame.draw.circle(self.screen, (255, 0, 0), 
                             (int(self.goal_x), int(self.goal_y)), 30, 3)
            car.draw(self.screen)
            car.draw_sensors(self.screen, self.road_surface)
            
            # Display info
            progress = ((self.initial_distance - min_distance) / self.initial_distance) * 100
            info_text = [
                "WINNER DEMONSTRATION",
                f"Steps: {steps}",
                f"Current distance: {int(dist_to_goal)}",
                f"Best distance: {int(min_distance)}",
                f"Progress: {progress:.1f}%",
                "Press ESC to exit"
            ]
            
            y_offset = 10
            for text in info_text:
                text_surface = self.font.render(text, True, (255, 255, 255))
                text_bg = pygame.Surface((text_surface.get_width() + 10, 
                                        text_surface.get_height() + 5))
                text_bg.fill((0, 0, 0))
                text_bg.set_alpha(180)
                self.screen.blit(text_bg, (10, y_offset))
                self.screen.blit(text_surface, (15, y_offset + 2))
                y_offset += 28
            
            pygame.display.flip()
            self.clock.tick(60)
        
        fitness = self.calculate_fitness(car, steps, reached_goal, min_distance)
        print(f"Final fitness: {fitness:.0f}")
        if reached_goal:
            print(f"SUCCESS! Reached goal in {steps} steps")
        else:
            print(f"Did not reach goal. Progress: {progress:.1f}%")


def run_training(generations=100):
    """Helper function to start training"""
    trainer = NEATTrainer()
    winner = trainer.train(generations=generations)
    return winner


if __name__ == "__main__":
    run_training(generations=100)
