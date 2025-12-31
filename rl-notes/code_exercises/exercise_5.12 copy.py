import numpy as np
from tqdm import tqdm

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
        start_locations.append(idx[::-1]) # Store as x, y
    elif val == 3:
        finish_locations.append(idx[::-1])
start_locations = np.array(start_locations)
finish_locations = set(tuple(loc) for loc in finish_locations)

epsilon = 0.5 # Start lower, usually better for on-policy
gamma = 1.0
alpha = 0.1 # Constant learning rate (better than 1/N for this)

# Q-Table: y, x, vy, vx, action
# Note: Dimensions are (19, 17, 5, 5, 9)
Q = np.random.random((*track_map.shape, 5, 5, 9)) - 10 # Init slightly negative to encourage exploration
# Action Map: 0:↙, 1:↓, 2:↘, 3:←, 4:•, 5:→, 6:↖, 7:↑, 8:↗
action_map = np.array([[-1,1], [0, 1], [1, 1], [-1,0], [0, 0], [1, 0], [-1,-1], [0, -1], [1, -1]])

def visualize_policy(Q, track_map, episode_num, epsilon):
    arrows = ['↙', '↓', '↘', '←', '•', '→', '↖', '↑', '↗']
    print(f"\nEpisode: {episode_num}, Epsilon: {epsilon:.4f}")
    for y in range(track_map.shape[0]):
        row_str = ""
        for x in range(track_map.shape[1]):
            if track_map[y, x] == 0:
                row_str += "  "
            elif track_map[y, x] == 3:
                row_str += "F "
            else:
                best_action = np.argmax(Q[y, x, 0, 0])
                row_str += arrows[best_action] + " "
        print(row_str)

def check_collision(old_state, new_state, track_map):
    # Bresenham-like or simple step check to prevent teleporting through walls
    y1, x1 = old_state[1], old_state[0]
    y2, x2 = new_state[1], new_state[0]
    
    # Calculate steps
    steps = max(abs(y2 - y1), abs(x2 - x1))
    if steps == 0: return False # No move
    
    for i in range(1, steps + 1):
        t = i / steps
        y = int(y1 + t * (y2 - y1))
        x = int(x1 + t * (x2 - x1))
        
        # Check bounds
        if not (0 <= y < track_map.shape[0] and 0 <= x < track_map.shape[1]):
            return True # Out of bounds
        
        # Check Wall
        if track_map[y, x] == 0:
            return True # Hit wall
            
        # Maybe check finish line crossing here for perfect precision
            
    return False

def generate_episode(Q, track_map, epsilon):
    start_choice = np.random.randint(0, len(start_locations))
    state = np.append(start_locations[start_choice], [0, 0]) # x, y, vx, vy
    log = []

    for _ in range(5000): # Safety limit
        initial_state = state.copy()

        # Epsilon Greedy
        if np.random.uniform() < epsilon:
            choice = np.random.randint(0, 9)
        else:
            choice = Q[state[1], state[0], state[3], state[2]].argmax()
        
        action = action_map[choice]

        # Velocity Update
        new_vx = np.clip(state[2] + action[0], 0, 4)
        new_vy = np.clip(state[3] + action[1], 0, 4)

        if new_vx == 0 and new_vy == 0:
            # Force agent to keep moving or stick to old velocity?
            # Standard rule: if result is 0,0, velocity doesn't change
            new_vx, new_vy = state[2], state[3]
            # If it was already 0,0 and tries to stay 0,0, that's allowed (but bad strategy)
        
        state[2] = new_vx
        state[3] = new_vy
        
        # Calculate intended new position
        new_x = state[0] + state[2]
        new_y = state[1] - state[3] # Moving UP array
        
        new_state_pos = np.array([new_x, new_y, state[2], state[3]])
        
        # Check Collision
        if check_collision(initial_state, new_state_pos, track_map):
            start_choice = np.random.randint(0, len(start_locations))
            state = np.append(start_locations[start_choice], [0,0])
            log.append((initial_state, choice, -10)) # Heavier penalty for crash helps speed up learning
        else:
            # Valid move
            state[0] = new_x
            state[1] = new_y
            
            # Check Finish
            if tuple(state[0:2]) in finish_locations:
                log.append((initial_state, choice, 0)) # Finished!
                break
            else:
                log.append((initial_state, choice, -1)) # Step cost
                
    return log

for i in tqdm(range(50000)):
    episode = generate_episode(Q, track_map, epsilon)
    
    if i > 0 and i % 100 == 0:
        epsilon = max(0.01, epsilon * 0.99)
        
    G = 0
    for state, action, reward in episode[::-1]:
        G = gamma * G + reward
        idx = (state[1], state[0], state[3], state[2], action)
        
        Q[idx] = Q[idx] + alpha * (G - Q[idx])


print("\nTest Runs (Deterministic):")
for i in range(5):
    episode = generate_episode(Q, track_map, 0)
    print(f"Run {i}: {len(episode)} steps")