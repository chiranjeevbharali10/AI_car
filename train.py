# train.py
from car import Car

def simulate():
    car = Car(100, 100)
    # Example: let AI/NN control car
    for step in range(10):
        car.move_forward(5)
        print(f"Car position: {car.x}, {car.y}, angle: {car.angle}")

if __name__ == "__main__":
    simulate()
