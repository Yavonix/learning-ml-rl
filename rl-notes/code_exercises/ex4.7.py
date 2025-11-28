import math
import numpy as np
from scipy.stats import poisson


np.set_printoptions(precision=1)


print("Precomputing state transition probabilities...")

max_cars = 21
poisson_tail = 50

transition_A = np.zeros((max_cars,max_cars))
transition_B = np.zeros((max_cars,max_cars))
expected_reward_A = np.zeros((max_cars)) # map starting number of cars to expected reward
expected_reward_B = np.zeros((max_cars)) # map starting number of cars to expected reward

pmf_map = {
    3: [poisson.pmf(i, 3) for i in range(poisson_tail)],
    4: [poisson.pmf(i, 4) for i in range(poisson_tail)],
    2: [poisson.pmf(i, 2) for i in range(poisson_tail)]
}

for starting_cars in range(0, max_cars):
    for returned_cars in range(0, poisson_tail):
        for rented_cars in range(0, poisson_tail):
            actual_rented = min(starting_cars, rented_cars)
            final_cars = min(starting_cars + returned_cars - actual_rented, 20)

            prob_A = pmf_map[3][rented_cars] * pmf_map[3][returned_cars]
            prob_B = pmf_map[4][rented_cars] * pmf_map[2][returned_cars]

            transition_A[starting_cars, final_cars] += prob_A
            transition_B[starting_cars, final_cars] += prob_B

            expected_reward_A[starting_cars] += prob_A * 10 * actual_rented
            expected_reward_B[starting_cars] += prob_B * 10 * actual_rented

def apply_action(action, state):
    return tuple(np.clip((state[0] - action, state[1] + action), 0, 20))

def eval_policy(values, policy, states, gamma, threshold):
    while True:

        delta = 0
        for state in states:
            old_value = values[state]
            action = policy[state]
            if action >= 0: # Move A -> B (limited by cars at A)
                actual_move = min(action, state[0])
            else: # Move B -> A (limited by cars at B)
                actual_move = -min(abs(action), state[1])
            reward = -2*abs(actual_move)
            next_direct_state = apply_action(actual_move, state)
            reward += expected_reward_A[next_direct_state[0]] + expected_reward_B[next_direct_state[1]]
            values[state] = reward + gamma * np.sum(np.outer(transition_A[next_direct_state[0], :], transition_B[next_direct_state[1], :]) * values)

            delta = max(delta, abs(old_value-values[state]))

        if delta < threshold:
            return values
        
def improve_policy(values, policy, states, actions, gamma):
    policy_stable = True
    for state in states:
        old_action = policy[state]

        best = (0,-math.inf)
        for action in actions:
            if action >= 0: # Move A -> B (limited by cars at A)
                actual_move = min(action, state[0])
            else: # Move B -> A (limited by cars at B)
                actual_move = -min(abs(action), state[1])
            reward = -2*abs(actual_move)
            next_direct_state = apply_action(actual_move, state)
            reward += expected_reward_A[next_direct_state[0]] + expected_reward_B[next_direct_state[1]]
            predicted_reward = reward + gamma * np.sum(np.outer(transition_A[next_direct_state[0], :], transition_B[next_direct_state[1], :]) * values)
            best = max(best, (action, predicted_reward), key=lambda x: x[1])

        policy[state] = best[0]
        if old_action != policy[state]: policy_stable = False
    
    return policy, policy_stable

states = [(i//max_cars, i%max_cars) for i in range(max_cars*max_cars)]
policy = np.zeros((max_cars,max_cars), dtype=int)
values = np.zeros((max_cars,max_cars))
actions = list(range(-5,6))

gamma = 0.9

stable = False

while not stable:
    values = eval_policy(values, policy, states, gamma, 0.001)
    print("complete eval")
    policy, stable = improve_policy(values, policy, states, actions, gamma)
    print(policy)
    print("complete pol")