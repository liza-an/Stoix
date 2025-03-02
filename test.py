import random
import jax
from stoix.custom_environments.bell_state_env import BellStateEnv


def test_bell_state_env():
    # Initialize the environment
    max_steps = 20
    env = BellStateEnv(max_steps=max_steps)

    # Generate a random key for the environment
    key = jax.random.PRNGKey(43)

    # Reset the environment
    print("Testing Reset:")
    state, timestep = env.reset(key)
    print(f"Initial Observation Shape: {timestep.observation.shape}")
    print(f"Initial Reward: {timestep.reward}")
    print(f"Initial Step Type: {timestep.step_type}")
    print()

    # Step through the environment
    print("Testing Step Functionality:")



    for step in range(max_steps):
        action = random.randint(0, 3)
        print(f"Step {step + 1}:")
        print(f"  Action: {action}")


        # Perform the step
        state, timestep, info = env.step(state, action)

        # Print results
        print(state["qc"])
        print(f"  Step Type: {timestep.step_type}")
        print(f"  Reward: {timestep.reward}")
        print(f"  Fidelity: {info['fidelity']:.4f}")
        print(f"  Done: {timestep.step_type == 2}")  # StepType.LAST == 2
        print()

        # Stop stepping if the environment signals the episode is done
        if timestep.step_type == 2:  # StepType.LAST
            break


if __name__ == "__main__":
    test_bell_state_env()
