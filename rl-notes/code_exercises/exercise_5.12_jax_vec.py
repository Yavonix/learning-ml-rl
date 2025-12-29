import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import time

print(jax.devices())

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

# We use a raw key here to make vmap easier with standard JAX tools, 
# though nnx.vmap is also possible.
def generate_episode_single(Q, track, key):
    # Create a local Rngs object for this single episode
    rngs = nnx.Rngs(key)

    def reset_state(state: State, rngs: nnx.Rngs):
        state.xy = rngs.choice(track.start_locations)
        state.vxvy = jnp.array([5,5])

    state = State(jnp.array([0,0]), jnp.array([0,0]))
    reset_state(state, rngs)
    
    def is_complete(state: State):
        matches = state.xy == track.finish_locations
        row_matches = jnp.all(matches, axis=-1)
        return jnp.any(row_matches)
    
    def choose_action(state: State, rngs: nnx.Rngs, Q):
        p = rngs.uniform(minval=0, maxval=1)

        greedy_choice = lambda Q: Q[state.xy[1], state.xy[0], state.vxvy[1], state.vxvy[0]].argmax()
        random_choice = rngs.randint(shape=(), minval=0, maxval=9)
        choice = jax.lax.cond(p<epsilon, lambda _: random_choice, greedy_choice, Q)

        return choice
    
    @nnx.scan(in_axes=(nnx.Carry, 0, None), out_axes=(nnx.Carry, 0))
    def loop_body(carry: tuple[State, nnx.Rngs], index, Q):
        state, rngs = carry

        def agent_step(state: State, rngs: nnx.Rngs, Q) -> tuple[jnp.ndarray, jnp.ndarray]:
            ## Choose action

            action_idx = choose_action(state, rngs, Q)
            action = action_map[action_idx]

            ## Update state
            state.xy = state.xy + (state.vxvy - 5)
            state.vxvy = jnp.clip(state.vxvy + action, 0, 10)

            # NEEDS FIXING
            is_valid = (0 <= state.xy[1]) & (state.xy[1] < track.track_map.shape[0]) \
                & (0 <= state.xy[0]) & (state.xy[0] < track.track_map.shape[1]) \
                & (track.track_map[state.xy[1], state.xy[0]] != 0)
            
            nnx.cond(is_valid, lambda s, r: None, reset_state, state, rngs)

            return action_idx, jnp.array(-1)

        def null_step(state, rngs, Q) -> tuple[jnp.ndarray, jnp.ndarray]:
            return (jnp.array(-1), jnp.array(0))

        flag = is_complete(state)
        action, reward = nnx.cond(flag, null_step, agent_step, state, rngs, Q)
    
        frame = Frame(nnx.clone(state), action, reward)
        return (state, rngs), frame    

    _, episode = loop_body((state, rngs), jnp.arange(1000), Q)
    return episode

# Vectorize over the random keys
generate_episodes_vectorized = nnx.jit(nnx.vmap(generate_episode_single, in_axes=(None, None, 0)))

BATCH_SIZE = 6000
keys = jax.random.split(jax.random.key(0), BATCH_SIZE)

print(f"Starting compilation and run for {BATCH_SIZE} episodes...")
start = time.time()
# First run includes compilation
episodes = generate_episodes_vectorized(Q, track, keys)
jax.block_until_ready(episodes)
end = time.time()
print(f"First run (compile + exec) took {end - start:.4f}s")

print(f"Starting second run (execution only)...")
start = time.time()
episodes = generate_episodes_vectorized(Q, track, keys)
jax.block_until_ready(episodes)
end = time.time()
print(f"Execution took {end - start:.4f}s")
print(f"Time per episode: {(end - start) / BATCH_SIZE * 1000:.4f} ms")

# Check output shape
print(f"Actions shape: {episodes.action.shape}")

print(episodes)

episode_id = 400

print()

current_act = episodes.action[:, 10000-1] == jnp.array(-1)
print(f"{100*current_act.sum()/current_act.size:.2f}% of episodes completed")



# for i in range(0, 100, 1):
#     # Slice the data for the current step
#     # Because Frame and State are NNX dataclasses/Pytrees, 
#     # indexing into the arrays gives you the value at that specific step.
#     current_pos = episodes.state.xy[episode_id][i]
#     current_vel = episodes.state.vxvy[episode_id][i]
#     current_act = episodes.action[episode_id][i]
#     current_rew = episodes.reward[episode_id][i]
    
#     print(f"Frame {i:02d} | Pos: {current_pos} | Vel: {current_vel} | Action Index: {current_act} | Reward: {current_rew}")