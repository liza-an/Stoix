from qiskit import QuantumCircuit
import numpy as np

# Constants for gate encoding
GATE_ENCODING = {
    "Hadamard": [1, 0, 0],
    "CNOT_q0_to_q1": [0, 1, 0],
    "CNOT_q1_to_q0": [0, 0, 1],
}

GATE_DECODING = {
    (1, 0, 0): "Hadamard",
    (0, 1, 0): "CNOT_q0_to_q1",
    (0, 0, 1): "CNOT_q1_to_q0",
}

def observation_to_qc(observation):
    """
    Converts a 3D observation matrix into a QuantumCircuit.

    Args:
        observation (np.ndarray): 3D matrix (time, qubit index, gate encoding).

    Returns:
        QuantumCircuit: The reconstructed quantum circuit.
    """
    num_qubits = observation.shape[1]
    qc = QuantumCircuit(num_qubits)

    for t in range(observation.shape[0]):  # Loop over time steps
        for qubit in range(num_qubits):  # Loop over qubits
            gate_vector = tuple(observation[t, qubit].astype(int))
            gate = GATE_DECODING.get(gate_vector, None)

            if gate == "Hadamard":
                qc.h(qubit)
            elif gate == "CNOT_q0_to_q1" and qubit == 0:
                qc.cx(0, 1)
            elif gate == "CNOT_q1_to_q0" and qubit == 1:
                qc.cx(1, 0)
    return qc


def qc_to_observation(qc, max_steps):
    """
    Converts a QuantumCircuit into a 3D observation matrix.

    Args:
        qc (QuantumCircuit): The quantum circuit to convert.
        max_steps (int): Maximum number of layers for the observation matrix.

    Returns:
        np.ndarray: The 3D observation matrix (time, qubit index, gate encoding).
    """
    num_qubits = qc.num_qubits
    observation = np.zeros((max_steps, num_qubits, 3), dtype=np.float32)

    for t, instruction in enumerate(qc.data):
        if t >= max_steps:  # Stop if max_steps is exceeded
            break

        gate_name = instruction[0].name
        qubits = [q.index for q in instruction[1]]  # Qubit indices

        if gate_name == "h":
            observation[t, qubits[0]] = GATE_ENCODING["Hadamard"]
        elif gate_name == "cx":
            if qubits == [0, 1]:  # CNOT q0 → q1
                observation[t, 0] = GATE_ENCODING["CNOT_q0_to_q1"]
            elif qubits == [1, 0]:  # CNOT q1 → q0
                observation[t, 1] = GATE_ENCODING["CNOT_q1_to_q0"]

    return observation
