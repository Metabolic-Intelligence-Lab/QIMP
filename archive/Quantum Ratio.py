# -*- coding: utf-8 -*-
"""
Created on Wed May 22 09:08:45 2024

@author: Giuseppe
"""

from FQRI_lib import *
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from scipy.ndimage import median_filter
from qiskit.circuit.library import MCXGate

# Define threshold value for pixel intensity
#intensity_threshold = 250  # Example threshold value

# Define a function to compute the ratio of angles
def compute_angle_ratio(theta1, theta2):
    # Avoid division by zero or invalid values
    if np.isclose(theta2, 0):
        return 0
    return np.arctan2(np.sin(theta1) * np.cos(theta2), np.cos(theta1) * np.sin(theta2))

# Load and convert images to angles
image1_name = "ratioch1_64.tif"
image2_name = "ratioch2_64.tif"
theta_values1, n1 = image_to_angles(image1_name)
theta_values2, n2 = image_to_angles(image2_name)

if n1 != n2:
    raise ValueError("Le due immagini devono avere la stessa dimensione.")

# Calculate the ratio of angles
theta_values_ratio = [compute_angle_ratio(theta1, theta2) for theta1, theta2 in zip(theta_values1, theta_values2)]

# Filter out any NaN values from the ratio angles
theta_values_ratio = [theta if not np.isnan(theta) else 0 for theta in theta_values_ratio]

# Setup quantum circuit for the ratio image
qc_ratio = setup_quantum_circuit(n1, theta_values_ratio)

# Remove background using quantum operations
def remove_background(qc, threshold=0.5):
    # Apply an X gate on the last qubit for values above the threshold
    last_qubit = qc.num_qubits - 1
    for i in range(qc.num_qubits - 1):
        qc.rx(2 * threshold, i)
    mcx_gate = MCXGate(qc.num_qubits - 1)
    qc.append(mcx_gate, list(range(qc.num_qubits - 1)) + [last_qubit])
    qc.x(last_qubit)
#remove_background(qc_ratio, threshold=0.5)
# Measure the quantum circuit
measure_quantum_circuit(qc_ratio)
# Simulate the circuit
simulator = AerSimulator()
qc_ratio = transpile(qc_ratio, simulator)
result = simulator.run(qc_ratio, shots=10**(n1 + 2)).result()
counts = result.get_counts(qc_ratio)
plot_histogram(counts, title='Ratio Image with Background Removed')

# Decode the quantum image
total_shots = sum(counts.values())
prob = {k: v / total_shots for k, v in counts.items()}
amp = {k: np.sqrt(v) for k, v in prob.items()}
psi_prime = " |ψ'⟩ = " + " + ".join([f"{amp_val:.4f}|{state}⟩" for state, amp_val in amp.items()])

prob_cond_0 = {}
prob_cond_1 = {}
values_q = {}
for state, p in prob.items():
    j = state[1:]
    if state[0] == '0':
        prob_cond_0[j] = p
    else:
        prob_cond_1[j] = p
for j in set(prob_cond_0.keys()) | set(prob_cond_1.keys()):
    p_0 = prob_cond_0.get(j, 0)
    p_1 = prob_cond_1.get(j, 0)
    if p_0 + p_1 > 0:
        values_q[j] = np.arccos(np.sqrt(p_0 / (p_0 + p_1))) * 255 * (2 / np.pi)
    else:
        values_q[j] = 0

num_pixels = 2 ** (2 * n1)
num_digits = len(next(iter(values_q.keys())))
sorted_pixel_values = {k: v for k, v in sorted(values_q.items(), key=lambda item: int(item[0], 2))}
image_quantum_ratio = np.zeros((int(np.sqrt(num_pixels)), int(np.sqrt(num_pixels))))
for index, value in sorted_pixel_values.items():
    row = int(index[:num_digits // 2], 2)
    col = int(index[num_digits // 2:], 2)
    image_quantum_ratio[row][col] = value

# Apply threshold to the quantum image
#image_quantum_ratio[image_quantum_ratio < intensity_threshold] = 0

# Apply median filter to the image
image_quantum_ratio_filtered = median_filter(image_quantum_ratio, size=3)

# Plot images
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(imageio.imread(image1_name), cmap='gray', vmin=0, vmax=255)
plt.colorbar()
plt.title("Immagine 1")
plt.subplot(1, 4, 2)
plt.imshow(imageio.imread(image2_name), cmap='gray', vmin=0, vmax=255)
plt.colorbar()
plt.title("Immagine 2")
plt.subplot(1, 4, 3)
plt.imshow(image_quantum_ratio, cmap='gray', vmin=0, vmax=255)
plt.colorbar()
plt.title("Immagine del Rapporto Quantistico")
plt.subplot(1, 4, 4)
plt.imshow(image_quantum_ratio_filtered, cmap='gray', vmin=0, vmax=255)
plt.colorbar()
plt.title("Immagine del Rapporto con Filtro Mediano")
plt.show()

# Save quantum ratio image as TIFF
save_image_as_tiff(image_quantum_ratio, "quantum_ratio_image_filtered_output.tif")
