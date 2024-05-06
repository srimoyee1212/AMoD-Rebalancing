import pandas as pd
from data_preprocessing import load_data, preprocess_data
from state_action_spaces import StateActionSpace
from sarsa_agent import SarsaAgent
from rebalancing_logic import RebalancingLogic

def main():
    # Load and preprocess data
    taxi_data = load_data()
    preprocessed_data = preprocess_data(taxi_data)

    # Define state and action spaces
    state_action_space = StateActionSpace(preprocessed_data)

    # Initialize SARSA agent
    sarsa_agent = SarsaAgent(state_action_space)

    # Train SARSA agent
    sarsa_agent.train()

    # Get Q-table from SARSA agent
    q_table = sarsa_agent.get_q_table()

    # Initialize rebalancing logic
    rebalancing_logic = RebalancingLogic(q_table)

    # Use rebalancing logic to choose actions
    states = [0, 1, 2]  # Example states
    actions = rebalancing_logic.choose_actions(states)

    print("Chosen actions:", actions)

if __name__ == "__main__":
    main()
