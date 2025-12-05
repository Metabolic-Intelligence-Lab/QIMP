# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:48:39 2024

@author: Giuseppe
"""
import numpy as np
import matplotlib.pyplot as plt
import imageio
import math
import os
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
from qiskit import transpile
import argparse
import logging

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def image_to_angles(image_name):
    try:
        working_dir = os.getcwd()
        image_path = os.path.join(working_dir, image_name)
        image = imageio.imread(image_path)

        if len(image.shape) == 3:
            image = image[:, :, 0]

        image = image / 255.0
        image = np.clip(1 - 2 * image, -1, 1)
        angles = np.arccos(image)
        angles_flat = angles.flatten()

        num_pixels = angles_flat.size
        n = int(math.log2(math.sqrt(num_pixels)))

        return angles_flat, n
    except FileNotFoundError:
        logging.error(f"File {image_name} non trovato.")
        raise
    except Exception as e:
        logging.error(f"Errore durante la lettura dell'immagine: {e}")
        raise

def setup_quantum_circuit(n, theta_values):
    n_qubits = 2 * n + 1
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits - 1))

    indices = [format(i, '0' + str(n_qubits - 1) + 'b')[::-1] for i in range(2 ** (n_qubits - 1))]
    for index, theta in zip(indices, theta_values):
        for bit_position, bit in enumerate(index):
            if bit == '0':
                qc.x(bit_position)
        ry_gate = RYGate(theta).control(n_qubits - 1)
        qc.append(ry_gate, list(range(n_qubits - 1)) + [n_qubits - 1])
        for bit_position, bit in enumerate(index):
            if bit == '0':
                qc.x(bit_position)
        qc.barrier()

    return qc

def decode_quantum_image(counts, n):
    num_pixels = 2 ** (2 * n)
    total_shots = sum(counts.values())
    prob = {k: v / total_shots for k, v in counts.items()}

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
    sorted_pixel_values = {k: v for k, v in sorted(values_q.items(), key=lambda item: extract_index(item[0]))}
    image_quantum = np.zeros((int(np.sqrt(num_pixels)), int(np.sqrt(num_pixels))))

    for index, value in sorted_pixel_values.items():
        row = int(index[:num_digits // 2], 2)
        col = int(index[num_digits // 2:], 2)
        image_quantum[row][col] = value

    return image_quantum

def extract_index(bitstring):
    return int(bitstring, 2)

def measure_quantum_circuit(qc):
    n_qubits = qc.num_qubits
    qc.measure(range(n_qubits), range(n_qubits))

def rotate_90(qc, n):
    for i in range(n):
        x_qubit = i
        y_qubit = n + i
        qc.swap(x_qubit, y_qubit)
    for qubit in range(n):
        qc.x(qubit)
    return qc

def plot_images(classic_images, quantum_images, titles=None):
    num_images = len(classic_images)
    for i in range(num_images):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(classic_images[i], cmap='gray', vmin=0, vmax=255)
        plt.colorbar()
        plt.title(titles[i][0] if titles else "Immagine Classica")

        plt.subplot(1, 2, 2)
        plt.imshow(quantum_images[i], cmap='gray', vmin=0, vmax=255)
        plt.colorbar()
        plt.title(titles[i][1] if titles else "Immagine Quantistica Ricostruita")
        plt.show()

def save_image_as_tiff(image_array, file_name):
    imageio.imwrite(file_name, image_array.astype(np.uint8), format='TIFF')

def main():
    parser = argparse.ArgumentParser(description='Process quantum images.')
    parser.add_argument('image_name', type=str, help='Name of the input image file')
    args = parser.parse_args()

    logging.info("Caricamento immagine...")
    theta_values, n = image_to_angles(args.image_name)

    logging.info("Impostazione del circuito quantistico...")
    qc = setup_quantum_circuit(n, theta_values)
    qc = rotate_90(qc, n)
    measure_quantum_circuit(qc)

    logging.info("Simulazione del circuito quantistico...")
    simulator = AerSimulator()
    qc = transpile(qc, simulator)
    job = simulator.run(qc, shots=10**(n + 1))
    result = job.result()
    qc.draw()
    logging.info("Recupero dei risultati...")
    experiment = qc.name if qc.name in result.results else result.results[0].header.name
    counts = result.get_counts(experiment)

    logging.info("Decodifica dei risultati...")
    image_quantum = decode_quantum_image(counts, n)

    logging.info("Visualizzazione delle immagini...")
    classic_image = np.array(theta_values) * 255 / np.pi
    classic_image = classic_image.reshape(int(np.sqrt(2 ** (2 * n))), int(np.sqrt(2 ** (2 * n))))
    plot_images([classic_image], [image_quantum], titles=[("Immagine Classica", "Immagine Quantistica Ricostruita")])

    logging.info("Salvataggio dell'immagine quantistica...")
    save_image_as_tiff(image_quantum, "quantum_image_output.tif")

if __name__ == "__main__":
    image_name = "Clipboard-1 32.tif"
    main()
