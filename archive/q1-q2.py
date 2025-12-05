# -*- coding: utf-8 -*-
"""
Created on Fri May 31 13:45:17 2024

@author: Giuseppe
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from scipy.ndimage import median_filter, gaussian_filter
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RYGate, MCXGate
from qiskit.visualization import plot_histogram
from qiskit import ClassicalRegister, QuantumRegister
from FQRI_lib import save_image_as_tiff

# Definisce la soglia di intensità del pixel per la rimozione del background
intensity_threshold = 10  # Soglia esempio, da regolare in base alle immagini

def preprocess_image(image):
    """Applica una soglia e un filtro passa-basso all'immagine."""
    # Debug: Visualizza valori massimi e minimi dell'immagine prima della preprocessazione
    print("Prima della preprocessazione: min =", np.min(image), ", max =", np.max(image))
    
    # Rimuovi il background
    image[image < intensity_threshold] = 0
    
    # Debug: Visualizza valori massimi e minimi dopo la rimozione del background
    print("Dopo la rimozione del background: min =", np.min(image), ", max =", np.max(image))
    
    # Applica filtro passa-basso (Gaussian filter)
    image_filtered = gaussian_filter(image, sigma=1)
    
    # Debug: Visualizza valori massimi e minimi dopo il filtro passa-basso
    print("Dopo il filtro passa-basso: min =", np.min(image_filtered), ", max =", np.max(image_filtered))
    
    return image_filtered

def image_to_quantum_register(image, qc, qubits):
    """Codifica un'immagine in un registro quantistico usando le porte RY."""
    n = int(np.log2(len(image)))
    for i, pixel in enumerate(image):
        binary_index = format(i, f'0{n}b')
        for j, bit in enumerate(binary_index):
            if bit == '0':
                qc.x(qubits[j])
        qc.ry(2 * np.arcsin(np.sqrt(pixel)), qubits[-1])
        for j, bit in enumerate(binary_index):
            if bit == '0':
                qc.x(qubits[j])
        qc.barrier()

def compute_ratio_circuit(image1, image2):
    """Crea un circuito quantistico che calcola il rapporto tra due immagini."""
    n = int(np.log2(len(image1)))
    qr = QuantumRegister(n + 1, 'q')
    cr = ClassicalRegister(n + 1, 'c')
    qc = QuantumCircuit(qr, cr)
    
    image_to_quantum_register(image1, qc, qr[:n])
    image_to_quantum_register(image2, qc, qr[:n])

    # Applicare operazioni per calcolare il rapporto
    for i in range(n):
        qc.ry(-2 * np.arcsin(np.sqrt(image2[i])), qr[i])
    
    qc.measure(qr, cr)
    
    return qc

def plot_images(image1, image2, image_quantum_ratio, image_quantum_ratio_filtered):
    """Visualizza le immagini originali, il rapporto quantistico e l'immagine filtrata."""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(image1, cmap='gray', vmin=0, vmax=255)
    plt.colorbar()
    plt.title("Immagine 1")
    
    plt.subplot(1, 4, 2)
    plt.imshow(image2, cmap='gray', vmin=0, vmax=255)
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

def main():
    image1_name = "ratioch1_64.tif"
    image2_name = "ratioch2_64.tif"

    # Carica le immagini
    image1 = imageio.imread(image1_name).astype(float) 
    image2 = imageio.imread(image2_name).astype(float) 

    # Visualizza le immagini originali prima della preprocessazione
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image1, cmap='gray')
    plt.colorbar()
    plt.title("Immagine 1 Originale")
    
    plt.subplot(1, 2, 2)
    plt.imshow(image2, cmap='gray')
    plt.colorbar()
    plt.title("Immagine 2 Originale")
    plt.show()
    
    # Preprocessa le immagini
    image1 = preprocess_image(image1)
    image2 = preprocess_image(image2)

    # Assicurati che le immagini siano piatte e abbiano la stessa dimensione
    if image1.shape != image2.shape:
        raise ValueError("Le due immagini devono avere la stessa dimensione.")
    
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()

    # Crea il circuito quantistico per calcolare il rapporto
    qc_ratio = compute_ratio_circuit(image1_flat, image2_flat)

    # Esegui il circuito su un simulatore quantistico
    simulator = AerSimulator()
    qc_ratio = transpile(qc_ratio, simulator)
    result = simulator.run(qc_ratio, shots=1024).result()
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

    num_pixels = 2 ** (2 * int(np.log2(len(image1_flat))))
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
    plot_images(image1, image2, image_quantum_ratio, image_quantum_ratio_filtered)

    # Salva l'immagine quantistica filtrata come TIFF
    save_image_as_tiff(image_quantum_ratio_filtered, "quantum_ratio_image_filtered_output.tif")

if __name__ == "__main__":
    main()
