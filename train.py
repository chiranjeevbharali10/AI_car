# train.py
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
        
        # Pygame setup for visualization (MUST initialize display first!)
        pygame.init()
        
        # Load track image AFTER pygame.init() to get dimensions
        temp_surface = pygame.image.load(track_path)
        self.width = temp_surface.get_width()
        self.height = temp_surface.get_height()
        
        # NOW create display
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("NEAT Car Training - Parallel")
        
        # NOW convert_alpha will work
        self.road_surface = pygame.image.load(track_path).convert_alpha()
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        
        # Training statistics
        self.generation = 0
        self.best_fitness = 0
        self.best_genome = None

    def is_on_road(self, x, y):
        """Check if position (x, y) is on the black road"""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        color = self.road_surface.get_at((int(x), int(y)))
        return color == (0, 0, 0, 255)

    def calculate_fitness(self, car, steps, reached_goal):
        """Calculate fitness based on distance traveled, goal proximity, and survival"""
        # Distance to goal
        dist_to_goal = math.sqrt((car.x - self.goal_x)**2 + (car.y - self.goal_y)**2)
        
        # Fitness components
        survival_bonus = steps * 0.1  # Reward for staying alive
        goal_proximity = max(0, 1000 - dist_to_goal)  # Reward for getting close to goal
        goal_bonus = 5000 if reached_goal else 0  # Big bonus for reaching goal
        
        fitness = survival_bonus + goal_proximity + goal_bonus
        return fitness

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
                'reached_goal': False
            })
        
        max_steps = 1000
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
                break  # All cars dead, end generation
            
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
                steering = output[0]  # -1 to 1
                acceleration = output[1]  # -1 to 1
                
                # Apply actions
                car.angle += steering * 5
                speed = max(0, acceleration * 3)
                car.move_forward(step=speed)
                
                # Check if still on road
                if not self.is_on_road(car.x, car.y):
                    car_data['alive'] = False
                    continue
                
                # Check if reached goal
                dist_to_goal = math.sqrt((car.x - self.goal_x)**2 + (car.y - self.goal_y)**2)
                if dist_to_goal < 30:
                    car_data['reached_goal'] = True
                    car_data['alive'] = False  # Done!
                    continue
                
                car_data['steps'] += 1
            
            current_step += 1
            
            # RENDER ALL CARS AT ONCE
            self.screen.fill((144, 238, 144))  # Green background
            self.screen.blit(self.road_surface, (0, 0))
            
            # Draw goal
            pygame.draw.circle(self.screen, (255, 0, 0), 
                             (int(self.goal_x), int(self.goal_y)), 30, 3)
            
            # Draw all alive cars
            for car_data in cars_data:
                if car_data['alive']:
                    car = car_data['car']
                    car.draw(self.screen)
            
            # Display stats
            successful_cars = sum(1 for c in cars_data if c['reached_goal'])
            info_text = [
                f"Generation: {self.generation}",
                f"Step: {current_step}/{max_steps}",
                f"Alive: {alive_count}/{len(cars_data)}",
                f"Reached goal: {successful_cars}",
                f"Best fitness: {self.best_fitness:.1f}"
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
            self.clock.tick(60)  # 60 FPS
        
        # Calculate fitness for all genomes
        for car_data in cars_data:
            fitness = self.calculate_fitness(
                car_data['car'], 
                car_data['steps'], 
                car_data['reached_goal']
            )
            car_data['genome'].fitness = fitness
            
            # Track best genome
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_genome = car_data['genome']
                print(f"Generation {self.generation}: New best fitness: {fitness:.2f}")

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
        print(f"Starting NEAT training for {generations} generations...")
        print("All cars in each generation will run simultaneously!")
        winner = population.run(self.eval_genomes, generations)
        
        # Save the winner
        import pickle
        with open('best_genome.pkl', 'wb') as f:
            pickle.dump(winner, f)
        
        print(f"\nTraining complete! Best fitness: {self.best_fitness:.2f}")
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
        
        max_steps = 1000
        steps = 0
        reached_goal = False
        
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
            if dist_to_goal < 30:
                reached_goal = True
                print("Car reached the goal!")
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
            info_text = [
                "WINNER DEMONSTRATION",
                f"Steps: {steps}",
                f"Distance to goal: {int(dist_to_goal)}",
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
        
        fitness = self.calculate_fitness(car, steps, reached_goal)
        print(f"Final fitness: {fitness:.2f}")


def run_training(generations=50):
    """Helper function to start training"""
    trainer = NEATTrainer()
    winner = trainer.train(generations=generations)
    return winner


if __name__ == "__main__":
    # Run training when script is executed directly
    run_training(generations=50)
