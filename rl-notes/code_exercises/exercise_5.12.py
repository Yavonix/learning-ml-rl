import numpy as np
import time
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
        start_locations.append(idx[::-1])
    elif val == 3:
        finish_locations.append(idx[::-1])
start_locations = np.array(start_locations)
finish_locations = set(tuple(loc) for loc in finish_locations)

# state is [x, y, vx, vy]

epsilon = 0.9
gamma = 1

# x, y, vx, vy, action
Q = np.zeros((*track_map.shape, 5, 5, 9))
N = np.zeros_like(Q)

# probability of taking each action

action_map = np.array([[-1,1], [0, 1], [1, 1], [-1,0], [0, 0], [1, 0], [-1,-1], [0, -1], [1, -1]])

def visualize_policy(Q: np.ndarray, N: np.ndarray, track_map, episode_num, epsilon):
    # Actions: 0:↙, 1:↓, 2:↘, 3:←, 4:•, 5:→, 6:↖, 7:↑, 8:↗
    arrows = ['↙', '↓', '↘', '←', '•', '→', '↖', '↑', '↗']
    
    # ANSI escape code to clear screen and move cursor to top left
    print("\033[H\033[J", end="")
    
    print(f"Episode: {episode_num}, Epsilon: {epsilon:.4f}")
    for y in range(track_map.shape[0]):
        row_str = ""
        for x in range(track_map.shape[1]):
            if track_map[y, x] == 0:
                row_str += "  "
            elif track_map[y, x] == 3:
                row_str += "F "
            else:
                # best_action = np.argmax(Q[y, x, 5, 5])
                best_action = np.argmax(Q[y, x].sum(axis=(0,1)))
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
            
        # Optional: Check finish line crossing here if you want perfect precision
            
    return False


def generate_episode(Q: np.ndarray, track_map, epsilon):
    start_choice = np.random.randint(0, len(start_locations))
    # Start with vx=0, vy=0
    state = np.append(start_locations[start_choice], [0, 0]) 

    log = []

    while True:
        initial_state = state.copy()

        # Epsilon-greedy selection
        if np.random.uniform() < epsilon:
            choice = np.random.randint(0, 9)
        else:
            choice = Q[state[1], state[0], state[3], state[2]].argmax()
        
        action = action_map[choice]

        # Calculate potential new velocity
        new_vx = np.clip(state[2] + action[0], 0, 4)
        new_vy = np.clip(state[3] + action[1], 0, 4)

        # Both cannot be zero. If they are, keep old velocity 
        if new_vx == 0 and new_vy == 0:
            pass 
        else:
            state[2] = new_vx
            state[3] = new_vy

        # x increases (moves right toward finish)
        state[0] += state[2] 
        # y DECREASES (moves UP the map toward finish)
        state[1] -= state[3] 

        # NEEDS FIXING
        is_valid = not check_collision(initial_state, state, track_map)

        if tuple(state[0:2]) in finish_locations:
            log.append((initial_state, choice, 0))
            break
            
        if not is_valid:
            start_choice = np.random.randint(0, len(start_locations))
            state = np.append(start_locations[start_choice], [0,0])
            log.append((initial_state, choice, -1))
        else:
            log.append((initial_state, choice, -1))
    
        if len(log) > 5000:
            return log 

    return log






for i in tqdm(range(50000)): # 50k should be enough with Alpha
    episode = generate_episode(Q, track_map, epsilon)
    
    # Decaying Epsilon
    if i > 0 and i % 100 == 0:
        epsilon = max(0.01, epsilon * 0.99) # Floor at 0.01
        
    G = 0
    for state, action, reward in episode[::-1]:
        G = gamma * G + reward
        idx = (state[1], state[0], state[3], state[2], action)
        
        # Constant Alpha Update
        Q[idx] = Q[idx] + 0.1 * (G - Q[idx])

# Test Run
print("\nTest Runs (Deterministic):")
for i in range(5):
    episode = generate_episode(Q, track_map, 0)
    print(f"Run {i}: {len(episode)} steps")












# for i in range(100000):
#     episode = generate_episode(Q, track_map, epsilon)
    
#     if i % 30 == 0:
#         visualize_policy(Q, N, track_map, i, epsilon)

#     epsilon = 0.99995 * epsilon
#     G = 0
#     for state, action, reward in episode[::-1]:
#         G = gamma * G + reward
#         idx = (state[1], state[0], state[3], state[2], action)
#         N[idx] += 1
#         # Q[idx] = Q[idx] + 0.05*(G - Q[idx])
#         Q[idx] = Q[idx] + (G - Q[idx])/N[idx]

# for i in range(100):
#     episode = generate_episode(Q, track_map, 0)
#     print(len(episode))