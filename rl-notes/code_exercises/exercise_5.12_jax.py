# TODO: delete this file

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import time

# jax.config.update('jax_disable_jit', True)

@nnx.dataclass
class Track(nnx.Pytree):
    track_map: jax.Array = nnx.data()
    start_locations: jax.Array = nnx.data(init=False)
    finish_locations: jax.Array = nnx.data(init=False)

    def __post_init__(self):
        start_locations = []
        finish_locations = []
        for idx, val in np.ndenumerate(self.track_map):
            if val == 2:
                start_locations.append(idx[::-1])
            elif val == 3:
                finish_locations.append(idx[::-1])
        self.start_locations = jnp.array(start_locations)
        self.finish_locations = jnp.array(finish_locations)

@nnx.dataclass
class State(nnx.Pytree):
    xy: jax.Array = nnx.data()
    vxvy: jax.Array = nnx.data()

@nnx.dataclass
class Frame(nnx.Pytree):
    state: State = nnx.data()
    action: jax.Array = nnx.data()
    reward: jax.Array = nnx.data()

track = Track(jnp.array([
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
]))

epsilon = 0.9
Q = jnp.zeros((*track.track_map.shape, 11, 11, 9)) # y, x, vy, vx, action
action_map = jnp.array([[-1,1], [0, 1], [1, 1], [-1,0], [0, 0], [1, 0], [-1,-1], [0, -1], [1, -1]])
rngs = nnx.Rngs(2)

# temp = State(jnp.array([0,0]), jnp.array([1,1]))
# greedy = Q[temp.xy[1], temp.xy[0], temp.vxvy[1], temp.vxvy[0]].argmax()
# print(greedy)

# state is [y, x, vy, vx]

@nnx.jit
def generate_episode(Q: jnp.ndarray, track: Track, rngs: nnx.Rngs):

    def refresh_state(state: State, rngs):
        start_location = rngs.choice(track.start_locations)
        return State(start_location, jnp.array([5,5]))

    def check_is_complete(state: State):
        matches = state.xy == track.finish_locations
        row_matches = jnp.all(matches, axis=-1)
        return jnp.any(row_matches)
    
    def choose_action(state: State, rngs: nnx.Rngs):
        p = rngs.uniform(minval=0, maxval=1)
        
        greedy_choice = lambda: Q[state.xy[1], state.xy[0], state.vxvy[1], state.vxvy[0]].argmax()
        random_choice = rngs.randint(shape=(), minval=0, maxval=9)
        choice = jax.lax.cond(p<epsilon, lambda: random_choice, greedy_choice)

        return choice
    
    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def loop_body(carry: tuple[State, nnx.Rngs], index):
        state, rngs = carry

        def agent_step(state: State, action_idx, possible_new_state) -> tuple[State, jnp.ndarray, jnp.ndarray]:
            action = action_map[action_idx]
            new_state = State(state.xy + (state.vxvy - 5), jnp.clip(state.vxvy + action, 0, 10))

            is_valid = (0 <= new_state.xy[1]) & (new_state.xy[1] < track.track_map.shape[0]) \
                & (0 <= new_state.xy[0]) & (new_state.xy[0] < track.track_map.shape[1]) \
                & (track.track_map[new_state.xy[1], new_state.xy[0]] != 0)

            return (jax.lax.cond(is_valid, lambda: new_state, lambda: possible_new_state), action_idx, jnp.array(-1))

        def complete_step(state,_,__):
            return (state, jnp.array(-1), jnp.array(0))

        is_complete = check_is_complete(state)
        action_idx = choose_action(state, rngs)
        possible_new_state = refresh_state(state, rngs)
        new_state, action, reward = jax.lax.cond(is_complete, complete_step, agent_step, state, action_idx, possible_new_state)
        state.xy = new_state.xy
        state.vxvy = new_state.vxvy
        
        frame = Frame(State(state.xy, state.vxvy), action, reward)
        return (state, rngs), frame

    state = State(jnp.array([0,0]),jnp.array([0,0]))
    starting_state = refresh_state(state, rngs)
    _, episode = loop_body((starting_state, rngs), jnp.arange(5000))
    return episode

print("starting")
start = time.time()
ep = generate_episode(Q, track, rngs)
end = time.time()
print(f"Compiling took {end - start}")

start = time.time()
ep = generate_episode(Q, track, rngs)
end = time.time()
print(f"Execution took {end - start}")


for i in range(len(ep.action)-20, len(ep.action)):
    # Slice the data for the current step
    # Because Frame and State are NNX dataclasses/Pytrees, 
    # indexing into the arrays gives you the value at that specific step.
    current_pos = ep.state.xy[i]
    current_vel = ep.state.vxvy[i]
    current_act = ep.action[i]
    current_rew = ep.reward[i]
    
    print(f"Frame {i:02d} | Pos: {current_pos} | Vel: {current_vel} | Action Index: {current_act} | Reward: {current_rew}")
# def generate_episode(policy, track_map, key):
#     key, subkey = jax.random.split(key)
#     jax.random.choice(subkey)

#     start_choice = np.random.randint(0, len(start_locations))
#     state = np.append(start_locations[start_choice], [5,5])

#     log = []

#     while True:
#         initial_state = tuple(state)

#         policy_probabilities = policy[initial_state]
#         indices = np.arange(len(policy_probabilities))
#         selected_action_index = np.random.choice(indices, p=policy_probabilities)
#         action = action_map[selected_action_index]

#         state[0:2] += state[2:4] - 5
#         state[2:4] += action
#         is_valid = (0 <= state[0] < track_map.shape[0]) and (0 <= state[1] < track_map.shape[1])
#         state[2:4] = np.clip(state[2:4], 0, 10)
#         is_valid = is_valid and track_map[tuple(state[0:2])] != 0

#         if not is_valid:
#             start_choice = np.random.randint(0, len(start_locations))
#             state = np.append(start_locations[start_choice], [5,5])

#         log.append((initial_state, selected_action_index, -1))

#         if tuple(state[0:2]) in finish_locations:
#             break

#     return log

# episode = generate_episode(policy, track_map)

# print(episode[0:8])