# AMoD-Rebalancing
Reinforcement Learning approach for rebalancing Autonomous Mobility-on-Demand (AMoD) systems

# Taxi Fleet Management Reinforcement Learning Project

This project implements a reinforcement learning approach to optimize taxi fleet management using SARSA (State-Action-Reward-State-Action) algorithm. The goal is to intelligently rebalance taxis across stations based on historical trip data from New York City.

## Table of Contents

- [Overview](#overview)
- [Files Included](#files-included)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Usage](#usage)

## Overview

The project simulates a taxi fleet management environment where taxis need to be strategically redistributed across stations (or zones) based on real-world trip data. It uses SARSA agent reinforcement learning to learn optimal policies for rebalancing.

## Files Included

- **src/data_preprocessing.py**: Loads and preprocesses historical taxi trip data, filtering out invalid records and unnecessary columns.
- **src/environment.py**: Defines the simulation environment with a fixed number of states and actions, simulating state transitions and rewards.
- **src/main.py**: Main script to initialize the environment, train the SARSA agent, and evaluate performance metrics.
- **src/rebalancing_logic.py**: Implements logic for choosing actions (rebalancing strategies) based on Q-values derived from SARSA learning.
- **src/sarsa_agent.py**: SARSA agent implementation that learns and updates Q-values based on state transitions and rewards.
- **src/state_action_spaces.py**: Defines the state and action spaces based on taxi data characteristics.

## Dependencies

- numpy==1.26.4
- pandas==2.2.2
- python-dateutil==2.9.0.post0
- pytz==2024.1
- six==1.16.0
- tzdata==2024.1

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/taxi-fleet-management.git
   cd taxi-fleet-management
2. Install dependencies:

  ```bash
  pip install -r requirements.txt
```

3. Usage
  ```bash
  python src/main.py

