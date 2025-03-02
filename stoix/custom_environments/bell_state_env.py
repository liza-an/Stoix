import random

import chex
import numpy as np
from jumanji import Environment
from jumanji.env import specs
from jumanji.types import StepType, TimeStep
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from stoix.custom_environments.utils import observation_to_qc

import jax
import jax.numpy as jnp
from qiskit.quantum_info import partial_trace
from jax import jit
import jax.lax

import chex
from typing import Any, Tuple

jax.config.update("jax_disable_jit", True)


@chex.dataclass
class EnvState:
    statevector: jnp.ndarray  # JAX array for quantum state
    step_count: int  # Integer step count
    key: Any  # Random seed



class BellStateEnv(Environment):
    def __init__(self, max_steps: int = 5):
        super().__init__()
        self.num_qubits = 2
        self.max_steps = max_steps

        # Define observation and action spaces
        # Observation: flattened statevector (4 complex numbers -> 8 real numbers)
        self._observation_spec = specs.BoundedArray(
            shape=(8,), dtype=jnp.float32, minimum=-1, maximum=1, name="observation"
        )
        # Actions: H0, H1, CNOT01, CNOT10
        self._action_spec = specs.DiscreteArray(num_values=4, dtype=int, name="action")

        # Target Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        self.target_state = jnp.array([1/jnp.sqrt(2), 0, 0, 1/jnp.sqrt(2)], dtype=jnp.complex64)

    def _compute_fidelity(self, statevector: jnp.ndarray) -> jnp.ndarray:
        """Compute fidelity between the current and target states using pure JAX."""
        # F = |⟨ψ|ϕ⟩|²
        inner_product = jnp.sum(jnp.conj(statevector) * self.target_state)
        return jnp.abs(inner_product) ** 2

    def reset(self, key: chex.PRNGKey) -> Tuple[EnvState, TimeStep]:
        """Reset the environment to initial state |00⟩."""
        # Initialize to |00⟩ state
        statevector = jnp.array([1.0 + 0.j, 0.j, 0.j, 0.j], dtype=jnp.complex64)
        
        # Create environment state
        env_state = EnvState(
            statevector=statevector,
            step_count=0,
            key=key,
        )

        # Convert statevector to real observation
        observation = jnp.concatenate([statevector.real, statevector.imag])

        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=0.0,
            discount=1.0,
            observation=observation,
            extras={"fidelity": self._compute_fidelity(statevector)}
        )
        return env_state, timestep

    def step(self, env_state: EnvState, action: jnp.ndarray) -> Tuple[EnvState, TimeStep]:
        """Execute a quantum gate and update the state."""
        def terminate_episode(_):
            """Return terminal timestep when max steps reached."""
            observation = jnp.concatenate([env_state.statevector.real, env_state.statevector.imag])
            fidelity = self._compute_fidelity(env_state.statevector)
            timestep = TimeStep(
                step_type=StepType.LAST,
                reward=-1.0,  # Small penalty for timeout
                discount=0.0,
                observation=observation,
                extras={"fidelity": fidelity}
            )
            return env_state, timestep

        def continue_episode(_):
            """Continue the episode with the selected action."""
            # Get current state
            statevector = env_state.statevector

            # Define quantum gates
            H = (1 / jnp.sqrt(2)) * jnp.array([[1, 1], [1, -1]], dtype=jnp.complex64)
            I = jnp.eye(2, dtype=jnp.complex64)
            CNOT = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=jnp.complex64)

            # Apply selected gate
            # H0: Hadamard on qubit 0
            h0_result = jnp.dot(jnp.kron(H, I), statevector)
            # H1: Hadamard on qubit 1
            h1_result = jnp.dot(jnp.kron(I, H), statevector)
            # CNOT01: Control=0, Target=1
            cnot01_result = jnp.dot(CNOT, statevector)
            # CNOT10: Control=1, Target=0
            cnot10_result = jnp.dot(jnp.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=jnp.complex64), statevector)

            # Select result based on action
            statevector = jnp.where(action == 0, h0_result, statevector)
            statevector = jnp.where(action == 1, h1_result, statevector)
            statevector = jnp.where(action == 2, cnot01_result, statevector)
            statevector = jnp.where(action == 3, cnot10_result, statevector)

            # Compute fidelity with target state
            fidelity = self._compute_fidelity(statevector)

            # Create new state
            new_env_state = EnvState(
                statevector=statevector,
                step_count=env_state.step_count + 1,
                key=env_state.key,
            )

            # Convert statevector to real observation
            observation = jnp.concatenate([statevector.real, statevector.imag])

            # Determine if done and compute reward
            done = fidelity >= 0.99
            # Reward structure:
            # - Large reward (10.0) for reaching target state
            # - Small reward based on improvement in fidelity
            # - Small step penalty (-0.1) to encourage efficiency
            reward = jnp.where(
                done,
                10.0,  # Success reward
                -0.1   # Step penalty
            )

            timestep = TimeStep(
                step_type=jnp.where(done, StepType.LAST, StepType.MID),
                reward=reward,
                discount=jnp.where(done, 0.0, 1.0),
                observation=observation,
                extras={"fidelity": fidelity}
            )

            return new_env_state, timestep

        return jax.lax.cond(
            env_state.step_count >= self.max_steps,
            terminate_episode,
            continue_episode,
            operand=None
        )

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec


def apply_gate(statevector, gate: int, qubits, num_qubits=2):
    """Applies a gate using an integer instead of a string."""
    if gate == 0:  # Hadamard gate
        H = (1 / jnp.sqrt(2)) * jnp.array([[1, 1], [1, -1]], dtype=jnp.complex64)
        return apply_single_qubit_gate(statevector, H, qubits, num_qubits)

    elif gate == 1:  # CNOT gate
        CNOT = jnp.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=jnp.complex64)
        return apply_two_qubit_gate(statevector, CNOT, qubits)

    else:
        raise ValueError(f"Unknown gate: {gate}")


# Example of how to call apply_gate with num_qubits as a static argument
# apply_gate_jit = jax.jit(apply_gate, static_argnames=["gate", "num_qubits"])


def apply_single_qubit_gate(statevector, gate, target, num_qubits=2):
    """Applies a single-qubit gate to the target qubit."""
    # Create identity matrices for tensor product
    I2 = jnp.eye(2, dtype=jnp.complex64)
    
    # Build the full operator based on target qubit
    if target == 0:
        full_operator = jnp.kron(gate, I2)
    else:
        full_operator = jnp.kron(I2, gate)
    
    return jnp.dot(full_operator, statevector)


import jax.lax as lax


def apply_two_qubit_gate(statevector, gate, targets):
    """Applies a two-qubit gate directly."""
    # For CNOT, we can apply the matrix directly since it's already in the correct basis
    return jnp.dot(gate, statevector)


# # Test the environment
# def test_bell_state_env():
#     env = BellStateEnv()
#     env_state, timestep = env.reset(chex.PRNGKey(42))  # Initialize with a key
#
#     actions = [0, 2]  # Optimal sequence: H on Q0, then CNOT targeting Q0
#
#     for step, action in enumerate(actions):
#         env_state, timestep, info = env.step(env_state, action)  # Properly update env_state
#
#         print(f"Step {step + 1}: Action {action}")
#         print(f"Observation: \n{observation_to_qc(timestep.observation)}")
#         print("Reward:", timestep.reward)
#         print("Done:", timestep.step_type == StepType.LAST)
#         print("Fidelity:", info["fidelity"])
#
#         if timestep.step_type == StepType.LAST:
#             break
#
#     if not timestep.step_type == StepType.LAST:
#         print("Environment terminated without achieving the goal.")
#
#     return timestep.observation
#
#
#
# # Test the environment
# def test_bell_state_env_random():
#     env = BellStateEnv()
#     env_state, timestep = env.reset(chex.PRNGKey(42))  # Initialize with a key
#
#     for step in range(env.max_steps):
#         action = random.randint(0, 3)  # Randomly select action
#         env_state, timestep, info = env.step(env_state, action)  # Properly update env_state
#
#         print(f"Step {step + 1}: Action {action}")
#         print(f"Observation: \n{observation_to_qc(timestep.observation)}")
#         print("Reward:", timestep.reward)
#         print("Done:", timestep.step_type == StepType.LAST)
#         print("Fidelity:", info["fidelity"])
#
#         if timestep.step_type == StepType.LAST:
#             break
#
#     if not timestep.step_type == StepType.LAST:
#         print("Environment terminated without achieving the goal.")
#
#     return timestep.observation


import jax


def test_bell_state_env_deterministic():
    """Test the environment using the optimal sequence of gates."""
    env = BellStateEnv()
    key = jax.random.PRNGKey(42)
    env_state, timestep = env.reset(key)

    # Optimal action sequence: H(0) → CNOT(0,1)
    optimal_actions = [0, 2]

    print("\n🧪 Running Deterministic Test (Optimal Actions)...")
    print(f"Initial state: |00⟩ = {env_state.statevector}")
    print(f"Target Bell state: {env.target_state}")

    for step, action in enumerate(optimal_actions):
        print(f"\n🔹 Step {step + 1}: Action {action}")
        
        # Take step
        env_state, timestep = env.step(env_state, action)
        
        # After step
        fidelity = env._compute_fidelity(env_state.statevector)
        if action == 0:
            print(f"After H(0): {env_state.statevector}")
            print(f"Expected: 1/√2(|0⟩ + |1⟩)|0⟩")
        elif action == 2:
            print(f"After CNOT: {env_state.statevector}")
            print(f"Expected: 1/√2(|00⟩ + |11⟩)")
        
        print(f"Fidelity with target: {fidelity:.4f}")
        print(f"Reward: {timestep.reward}")
        print(f"Done: {timestep.step_type == StepType.LAST}")

        if timestep.step_type == StepType.LAST:
            print("\n✅ Goal state reached!")
            break

    if timestep.step_type != StepType.LAST:
        print("\n⚠️ Environment terminated without achieving the goal.")
        print(f"Final state: {env_state.statevector}")
        print(f"Final fidelity: {fidelity:.4f}")

    return timestep.observation


def test_bell_state_env_random():
    """Test the environment with random actions."""
    env = BellStateEnv()
    key = jax.random.PRNGKey(42)  # ✅ Use JAX's PRNG key
    env_state, timestep = env.reset(key)  # Pass key to reset

    print("\n🎲 Running Random Test...")

    for step in range(env.max_steps):
        action = random.randint(0, 3)  # Select random action
        env_state, timestep = env.step(env_state, action)

        print(f"🔹 Step {step + 1}: Action {action}")
        print("Reward:", timestep.reward)
        print("Done:", timestep.step_type == StepType.LAST)

        if timestep.step_type == StepType.LAST:
            print("\n✅ Episode ended.")
            break

    if timestep.step_type != StepType.LAST:
        print("\n⚠️ Environment terminated without achieving the goal.")

    return timestep.observation


# Run the tests
# if __name__ == "__main__":
    # test_bell_state_env_deterministic()
    # test_bell_state_env_random()
