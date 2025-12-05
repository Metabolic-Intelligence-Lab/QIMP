# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:57:32 2024

@author: Giuseppe
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from scipy.ndimage import median_filter, gaussian_filter
from qiskit.circuit.library import MCXGate

from qiskit.visualization import plot_histogram
from FQRI_lib import *
# Definisce la soglia di intensità del pixel per la rimozione del background
intensity_threshold =1  # Soglia esempio, da regolare in base alle immagini



def main():
    image1_name = "ratioch1_64.tif"
    image2_name = "ratioch2_64.tif"

    # Carica le immagini e convertili in angoli
    theta_values1, n1 = image_to_angles(image1_name)
    theta_values2, n2 = image_to_angles(image2_name)

    if n1 != n2:
        raise ValueError("Le due immagini devono avere la stessa dimensione.")

    # Preprocessa le immagini
    image1 = preprocess_image(theta_values1.reshape(int(np.sqrt(len(theta_values1))), -1),intensity_threshold )
    image2 = preprocess_image(theta_values2.reshape(int(np.sqrt(len(theta_values2))), -1),intensity_threshold )
    
    theta_values1 = image1.flatten()
    theta_values2 = image2.flatten()

    # Calcola il rapporto degli angoli
    theta_values_ratio = [compute_angle_ratio(theta1, theta2) for theta1, theta2 in zip(theta_values1, theta_values2)]
    theta_values_ratio = [theta if not np.isnan(theta) else 0 for theta in theta_values_ratio]

    qc_ratio = setup_quantum_circuit(n1, theta_values_ratio)

    # Rimuovi il background usando operazioni quantistiche (opzionale)
    # remove_background(qc_ratio, threshold=0.5)

    measure_quantum_circuit(qc_ratio)

    simulator = AerSimulator()
    qc_ratio = transpile(qc_ratio, simulator)
    result = simulator.run(qc_ratio, shots=10**(n1 + 2)).result()
    counts = result.get_counts(qc_ratio)
    
    plot_histogram(counts, title='Ratio Image with Background Removed')

    total_shots = sum(counts.values())
    prob = {k: v / total_shots for k, v in counts.items()}
    amp = {k: np.sqrt(v) for k, v in prob.items()}

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

    # Applica filtro mediano all'immagine
    image_quantum_ratio_filtered = median_filter(image_quantum_ratio, size=3)

    # Visualizza le immagini
    plot_images(image1_name, image2_name, image_quantum_ratio, image_quantum_ratio_filtered)

    # Salva l'immagine quantistica filtrata come TIFF
    save_image_as_tiff(image_quantum_ratio_filtered, "quantum_ratio_image_filtered_output.tif")

if __name__ == "__main__":
    main()
