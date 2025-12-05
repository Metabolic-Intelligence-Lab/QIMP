# -*- coding: utf-8 -*-
"""
Created on Wed May 22 09:08:45 2024

@author: Giuseppe
"""

from FQRI_lib import *



# Define threshold value for pixel intensity
intensity_threshold = 250  # Example threshold value

QiskitRuntimeService.save_account(channel="ibm_quantum", token="a701629b4cb81190a4a3707b135362ca3dd24e69f34a6920f7973bd9376caace2c95491188c2699cfae1b0078ed6b30dcd0774238a2d40ad8dcc740d18930471", overwrite=True)

image_name = "Clipboard-1.tif"
theta_values, n = image_to_angles(image_name)
n_qubits = 2 * n
n_shots = 10**(n + 2)
num_pixels = 2**(n_qubits)

image_values = np.array(theta_values) * 255 / np.pi
image_classic = image_values.reshape(int(np.sqrt(num_pixels)), int(np.sqrt(num_pixels)))

qc = setup_quantum_circuit(n, theta_values)
marked_states = [n_qubits]

apply_amplitude_amplification(qc, marked_states)
#apply_color_inversion(qc)
measure_quantum_circuit(qc)

simulator = AerSimulator()
qc = transpile(qc, simulator)
result = simulator.run(qc, shots=n_shots).result()
counts = result.get_counts(qc)
plot_histogram(counts, title='FRQI')
total_shots = sum(counts.values())
prob = {k: v / total_shots for k, v in counts.items()}
amp = {k: np.sqrt(v) for k, v in prob.items()}
psi_prime = " |ψ'⟩ = " + " + ".join([f"{amp_val:.4f}|{state}⟩" for state, amp_val in amp.items()])

# Decoding quantum image
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

num_digits = len(next(iter(values_q.keys())))
sorted_pixel_values = {k: v for k, v in sorted(values_q.items(), key=lambda item: int(item[0], 2))}
image_quantum = np.zeros((int(np.sqrt(num_pixels)), int(np.sqrt(num_pixels))))
for index, value in sorted_pixel_values.items():
    row = int(index[:num_digits // 2], 2)
    col = int(index[num_digits // 2:], 2)
    image_quantum[row][col] = value

# Apply threshold to the quantum image
image_quantum[image_quantum < intensity_threshold] = 0

# Plot images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_classic, cmap='gray', vmin=0, vmax=255)
plt.colorbar()
plt.title("Immagine Classica")
plt.subplot(1, 2, 2)
plt.imshow(image_quantum, cmap='gray', vmin=0, vmax=255)
plt.colorbar()
plt.title("Immagine Quantistica Ricostruita con Soglia")
plt.show()

# Save quantum image as TIFF
save_image_as_tiff(image_quantum, "quantum_image_output.tif")

