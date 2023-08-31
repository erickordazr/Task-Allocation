# Allocation task in swarm robotics

## Description

**Allocation task in swarm robotics** is a Python-based platform designed to simulate the behavior of robot swarms. Users can simulate individual or multiple robot behaviors based on parameters such as the number of objects, number of individuals, and various radii for repulsion, orientation, and attraction. The results of the simulation can be visualized and are saved as numpy arrays for further analysis.

## Features

- **Single Simulation Mode**: Run a single instance of the simulation with user-defined parameters.
- **Multiple Simulations Mode**: Execute multiple replicas of the simulation to gather more comprehensive data.
- **Visualization**: Option to visualize the simulation in real-time (if integrated with a visualization tool in the future).
- **Data Saving**: Results are saved as numpy arrays for easy post-processing and analysis.

## Setup and Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/erickordazr/task-allocation.git
    cd task-allocation
    ```

2. **Install Dependencies**:
    Ensure you have Python installed. Then, install the required packages using:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Simulation**:
    Navigate to the `src` directory and run:
    ```bash
    task-allocation.py
    ```

## Usage

1. **Choose Simulation Mode**:
    - After running `task-allocation.py`, you'll be presented with a menu.
    - Choose `1` for a single simulation or `2` for multiple simulations.
    - If you wish to exit, choose `3`.

2. **Input Parameters**:
    - You'll be prompted to input various parameters such as the number of objects, robots, and radii for repulsion, orientation, and attraction.
    - For multiple simulations, you'll also need to specify the number of replicas.

3. **View Results**:
    - If you've chosen to visualize the simulation, watch it in real-time.
    - Once the simulation is complete, the results will be saved in the `data` directory as numpy arrays.

## Contributing

We welcome contributions! If you find a bug or have suggestions for improvements, please open an issue. If you'd like to contribute code, fork the repository, make your changes, and submit a pull request.

## License

This project is open-source and available under the [GNU General Public License](LICENSE).
