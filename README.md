# AMoD-Rebalancing
Reinforcement Learning approach for rebalancing Autonomous Mobility-on-Demand (AMoD) systems

This project implements a reinforcement learning approach to optimize taxi fleet management using SARSA (State-Action-Reward-State-Action) algorithm. The goal is to intelligently rebalance taxis across stations based on historical trip data from New York City.

![image](https://github.com/srimoyee1212/AMoD-Rebalancing/assets/30791239/e75d1b0b-a81b-48ab-80b5-ccde927bfce0)


## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Files Included](#files-included)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Usage](#usage)

## Overview

The project simulates a taxi fleet management environment where taxis need to be strategically redistributed across stations (or zones) based on real-world trip data. It uses SARSA agent reinforcement learning to learn optimal policies for rebalancing.

## Dataset

Context
Dive into the rich dataset of New York City's yellow taxis, a treasure trove for any data scientist. This dataset is a detailed chronicle of urban mobility, capturing intricate details of each taxi trip. From temporal data points like pickup and drop-off times to spatial dimensions involving locations, this data offers a granular view of urban transit patterns. It also includes quantitative metrics like trip distance, fare composition, payment methods, and passenger counts. Provided by tech companies under the TPEP/LPEP initiatives, this data is a goldmine for analyzing urban transportation trends, fare dynamics, and passenger behavior. It's not just about taxis and their routes; it's a window into the lifeblood of the city, offering endless possibilities for data-driven insights and urban plannin

Column Descriptions

- VendorID: Identifier for the TPEP provider supplying the record.
   - 1 = Creative Mobile Technologies, LLC
   - 2 = VeriFone Inc.
- tpep_pickup_datetime: The date and time when the meter was activated.
- tpep_dropoff_datetime: The date and time when the meter was turned off.
- Passenger_count: The number of passengers in the vehicle, as entered by the driver.
- Trip_distance: The distance of the trip in miles, as recorded by the taximeter.
- PULocationID: TLC Taxi Zone where the meter was engaged.
- DOLocationID: TLC Taxi Zone where the meter was disengaged.
- RateCodeID: The applicable rate code at the end of the trip.
   - 1 = Standard rate
   - 2 = JFK
   - 3 = Newark
   - 4 = Nassau or Westchester
   - 5 = Negotiated fare
   - 6 = Group ride
- Store_and_fwd_flag: Indicates if the trip record was stored in the vehicle's memory before transmission to the vendor due to lack of server connection.
   - Y = Store and forward trip
   - N = Not a store and forward trip
- Payment_type: How the passenger paid for the trip, represented by a numeric code.
   - 1 = Credit card
   - 2 = Cash
   - 3 = No charge
   - 4 = Dispute
   - 5 = Unknown
   - 6 = Voided trip
- Fare_amount: The fare as calculated by the meter based on time and distance.
- Extra: Additional charges, currently including only the $0.50 and $1 rush hour and overnight charges.
- MTA_tax: A $0.50 tax automatically added based on the metered rate.
- Improvement_surcharge: A $0.30 surcharge added at the start of the trip, implemented since 2015.
- Tip_amount: Credit card tip amounts. (Note: Cash tips are not recorded here.)
- Tolls_amount: Total tolls paid during the trip.
- Total_amount: The total charge to passengers, excluding cash tips.

Source:
[Kaggle New York Taxi Trip Dataset](https://www.kaggle.com/datasets/microize/newyork-yellow-taxi-trip-data-2020-2019)

[NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

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

