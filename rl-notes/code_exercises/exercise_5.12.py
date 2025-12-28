import numpy as np

track_map = np.array([
    [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,3],
    [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3], 
    [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3],
    [0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0], 
    [0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,2,2,2,2,2,0,0,0,0,0,0,0,0,0],
])

start_locations = []
finish_locations = []
for idx, val in np.ndenumerate(track_map):
    if val == 2:
        start_locations.append(idx)
    elif val == 3:
        finish_locations.append(idx)
start_locations = np.array(start_locations)
finish_locations = set(tuple(loc) for loc in finish_locations)

# state is [y, x, vy, vx]

epsilon = 0.1

# y, x, vy, vx, action
Q = np.zeros((*track_map.shape, 11, 11, 9))

# probability of taking each action

action_map = [(-1,1), (0, 1), (1, 1), (-1,0), (0, 0), (1, 0), (-1,-1), (0, -1), (1, -1)]

policy = np.ones_like(Q) / 9

def generate_episode(policy, track_map):
    start_choice = np.random.randint(0, len(start_locations))
    state = np.append(start_locations[start_choice], [5,5])

    log = []

    while True:
        initial_state = tuple(state)

        policy_probabilities = policy[initial_state]
        indices = np.arange(len(policy_probabilities))
        selected_action_index = np.random.choice(indices, p=policy_probabilities)
        action = action_map[selected_action_index]

        state[0:2] += state[2:4] - 5
        state[2:4] += action
        is_valid = (0 <= state[0] < track_map.shape[0]) and (0 <= state[1] < track_map.shape[1])
        state[2:4] = np.clip(state[2:4], 0, 10)
        is_valid = is_valid and track_map[tuple(state[0:2])] != 0

        if not is_valid:
            start_choice = np.random.randint(0, len(start_locations))
            state = np.append(start_locations[start_choice], [5,5])

        log.append((initial_state, selected_action_index, -1))

        if tuple(state[0:2]) in finish_locations:
            break

    return log

episode = generate_episode(policy, track_map)

print(episode[0:8])