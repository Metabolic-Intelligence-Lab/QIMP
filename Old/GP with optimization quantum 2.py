# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:19:15 2024

@author: Giuseppe
"""

from qiskit_aer import AerSimulator
from qiskit import transpile
from FQRI_lib2 import *
from qiskit.circuit import Parameter
from scipy.optimize import minimize
import numpy as np
import time
from qiskit.circuit.library import RYGate
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Funzione per calcolare il MAD
def calculate_mad(image1, image2):
    return np.mean(np.abs(image1 - image2))

# Funzione per eseguire il circuito quantistico
def run_quantum_circuit(qc, simulator, shots):
    transpiled_qc = transpile(qc, simulator)
    job = simulator.run(transpiled_qc, shots=shots)
    result = job.result()
    counts = result.get_counts()
    return counts

# Funzione per ottimizzare il circuito
def optimize_circuit(qc, n, m, angles, classic_images, normalization_factor, simulator, shots):
    mad_history = []  # Lista per memorizzare i valori di MAD

    def objective_function(theta):
        qc_copy = qc.copy()
        idx = 0
        for i in range(2 ** (2 * n)):
            for j in range(2 ** m):
                control_qubits = list(range(2 * n)) + [2 * n + j]
                control_qubits = list(set(control_qubits))
                ancilla_qubit = qc_copy.num_qubits - 1
                if ancilla_qubit in control_qubits:
                    control_qubits.remove(ancilla_qubit)
                if len(control_qubits) > 0:
                    qc_copy.mcx(control_qubits, ancilla_qubit)
                    qc_copy.rz(theta[idx], ancilla_qubit)
                    qc_copy.mcx(control_qubits, ancilla_qubit)
                    idx += 1

        measure_quantum_circuit(qc_copy)
        counts = run_quantum_circuit(qc_copy, simulator, shots)
        quantum_images = decode_quantum_images(counts, n, m, normalization_factor)

        quantum_gp = quantum_images[0] if len(quantum_images) > 0 else np.zeros_like(classic_images[0])
        classical_gp = calculate_classical_gp(classic_images[0], classic_images[1])
        classical_gp_normalized = renormalize_image(classical_gp)
        mad = calculate_mad(classical_gp_normalized, quantum_gp)

        mad_history.append(mad)  # Aggiungi il valore di MAD alla lista
        print(f"Iterazione {len(mad_history)}: MAD = {mad}")
        return mad

    initial_theta = np.zeros(2 ** (2 * n) * 2 ** m)
    result = minimize(objective_function, initial_theta, method='COBYLA')

    return result.x, mad_history

# Funzione per calcolare il GP
def calculate_gp(qc, n, m, classic_images, normalization_factor, simulator, shots):
    n_qubits = 2 * n + m + 1
    ancilla_qubit = n_qubits - 1
    optimized_angles, mad_history = optimize_circuit(qc, n, m, np.zeros((2 ** (2 * n) * 2 ** m)), classic_images, normalization_factor, simulator, shots)
    idx = 0
    for i in range(2 ** (2 * n)):
        binary = format(i, f'0{2 * n}b')
        for j in range(2 ** m):
            control_qubits = list(range(2 * n)) + [2 * n + j]
            control_qubits = list(set(control_qubits))
            if ancilla_qubit in control_qubits:
                control_qubits.remove(ancilla_qubit)
            if len(control_qubits) > 0:
                qc.mcx(control_qubits, ancilla_qubit)
                qc.rz(optimized_angles[idx], ancilla_qubit)
                qc.mcx(control_qubits, ancilla_qubit)
                idx += 1
    qc.x(ancilla_qubit)
    return qc, mad_history

# Funzione per rinormalizzare l'immagine
def renormalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image

# Funzione per calcolare il GP classico
def calculate_classical_gp(image1, image2):
    image1 = np.array(image1, dtype=np.float32)
    image2 = np.array(image2, dtype=np.float32)
    gp = (image1 - image2) / (image1 + image2 + 1e-8)
    gp = np.clip(gp, -1, 1)
    return gp

# Funzione per calcolare la somiglianza
def calculate_similarity(classical_gp, quantum_gp):
    if classical_gp.shape != quantum_gp.shape:
        raise ValueError("Le dimensioni delle immagini GP classica e quantistica devono essere uguali.")
    mad = np.mean(np.abs(classical_gp - quantum_gp))
    return mad

# Funzione principale
def main(image_names):
    print("Caricamento delle immagini e codifica degli angoli...")
    angles_list, n, m, normalization_factor = load_and_encode_images(image_names)
    
    print("Impostazione del circuito quantistico...")
    qc = setup_quantum_circuit(n, m, angles_list)
    
    print("Calcolo del GP...")
    simulator = AerSimulator()
    shots = 10**(n + 2)
    classic_images = [np.array(angles) * normalization_factor / np.pi for angles in angles_list]
    classic_images = [image.reshape(int(np.sqrt(2 ** (2 * n))), int(np.sqrt(2 ** (2 * n)))) for image in classic_images]
    qc, mad_history = calculate_gp(qc, n, m, classic_images, normalization_factor, simulator, shots)
    measure_quantum_circuit(qc)
    
    print("Simulazione del circuito quantistico...")
    qc = transpile(qc, simulator)
    job = simulator.run(qc, shots=shots)
    result = job.result()
    
    print("Recupero dei risultati...")
    experiment = qc.name if qc.name in result.results else result.results[0].header.name
    counts = result.get_counts(experiment)
    
    print("Decodifica dei risultati...")
    quantum_images = decode_quantum_images(counts, n, m, normalization_factor)
    
    print("Calcolo classico del GP...")
    classical_gp = calculate_classical_gp(classic_images[0], classic_images[1])
    
    print("Rinormalizzazione del GP classico...")
    classical_gp_normalized = renormalize_image(classical_gp)
    
    print("Visualizzazione delle immagini...")
    titles = [(f"Immagine Classica {i+1}", f"Immagine Quantistica Ricostruita {i+1}") for i in range(len(image_names))]
    plot_images_strip(classic_images, quantum_images, titles=titles)
    
    print("Visualizzazione del confronto tra calcolo classico e quantistico...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(classical_gp_normalized, cmap='gray')
    axes[0].set_title("Immagine GP Classica Rinormalizzata")
    axes[0].axis('off')
    
    if len(quantum_images) > 0:
        axes[1].imshow(quantum_images[0], cmap='gray')
        axes[1].set_title("Immagine GP Quantistica")
        axes[1].axis('off')
    
    plt.show()
    
    print("Salvataggio delle immagini quantistiche...")
    save_images_as_tiff(quantum_images, "quantum_image_output")
    
    print("Salvataggio delle immagini classiche...")
    save_images_as_tiff(classic_images, "classic_image_output")
    save_images_as_tiff([classical_gp_normalized], "classical_gp_output")
    
    print("Calcolo della somiglianza tra GP classica e quantistica...")
    if len(quantum_images) > 0:
        quantum_gp = quantum_images[0]
        similarity_score = calculate_similarity(classical_gp_normalized, quantum_gp)
        print(f"Differenza Assoluta Media (MAD) tra GP classica e quantistica: {similarity_score}")
    
    # Visualizzazione del circuito ottimizzato
    print("Visualizzazione del circuito quantistico ottimizzato...")
    optimized_circuit = qc.draw(output='mpl')
    plt.show(optimized_circuit)
    
    # Visualizzazione del calo del MAD
    print("Visualizzazione del calo del MAD durante l'ottimizzazione...")
    plt.figure()
    plt.plot(mad_history, marker='o')
    plt.xlabel('Iterazione')
    plt.ylabel('MAD')
    plt.title('Calo del MAD durante l\'ottimizzazione')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    image_names = ["ratioch1_4-1.tif", "ratioch2_4-1.tif"]
    main(image_names)
