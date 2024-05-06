import pandas as pd
from data_preprocessing import taxi_data
from state_action_spaces import StateActionSpace
from sarsa_agent import SARSA_Agent
from rebalancing_logic import RebalancingLogic

def main():
    
    preprocessed_data = taxi_data


    state_action_space = StateActionSpace(preprocessed_data)

    # Retrieve dimensions from StateActionSpace object
    state_dim = state_action_space.state_dim
    action_dim = state_action_space.action_dim

   
    sarsa_agent = SARSA_Agent(state_dim, action_dim)

  
    sarsa_agent.train()


    q_table = sarsa_agent.get_q_table()

   
    rebalancing_logic = RebalancingLogic(q_table)

    
    states = [0, 1, 2]  # Sample states
    actions = rebalancing_logic.choose_actions(states)

    print("Chosen actions:", actions)

if __name__ == "__main__":
    main()
