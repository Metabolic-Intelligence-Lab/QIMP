# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:18:25 2024

@author: Giuseppe
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import math
from qiskit.circuit.library import RYGate
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from matplotlib.gridspec import GridSpec
from PIL import Image

# --- Section 1: Image Loading and Encoding ---



def load_and_encode_images(image_names):

    angles_list = []
    n = None
    m = int(math.log2(len(image_names)))
    normalization_factor = 4096

    for image_name in image_names:
        working_dir = os.getcwd()
        image_path = os.path.join(working_dir, image_name)

        # Usa Pillow per caricare l'immagine TIFF
        image = Image.open(image_path)
        image = np.array(image)

        normalization_factor = 4096.0
        image = image / normalization_factor
        image = np.clip(1 - 2 * image, -1, 1)
        angles = np.arccos(image)
        angles_flat = angles.flatten()

        angles_list.append(angles_flat)

        if n is None:
            num_pixels = angles_flat.size
            n = int(math.log2(math.sqrt(num_pixels)))

    return angles_list, n, m, normalization_factor


# --- Section 2: Quantum Circuit Setup ---

def setup_quantum_circuit(n, m, angles_list):
    """
    Imposta il circuito quantistico con le immagini codificate.

    Args:
        n (int): Numero di qubit per dimensione.
        m (int): Numero di immagini.
        angles_list (list of np.ndarray): Lista di angoli codificati.

    Returns:
        QuantumCircuit: Circuito quantistico configurato.
    """
    n_qubits = 2 * n + m + 1
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    qc.h(range(n_qubits - 1))
    
    indices = [format(i, '0' + str(2 * n) + 'b')[::-1] for i in range(2 ** (2 * n))]
    selection_index = [format(j, '0' + str(m) + 'b')[::-1] for j in range(2**m)]
        
    for s_index, theta_values in enumerate(angles_list):
        
        for index, theta in zip(indices, theta_values):
            
            for bit_position, bit in enumerate(index):
                if bit == '0':
                    qc.x(bit_position)
                
            ry_gate = RYGate(theta).control(m + 2 * n)
            control_bits = list(range(n_qubits - 1)) + [n_qubits - 1]
            qc.append(ry_gate, control_bits)
                
            for bit_position, bit in enumerate(index):
                if bit == '0':
                    qc.x(bit_position)
            
            qc.barrier()

        for index2 in selection_index:
              
            for bit_position, bit in enumerate(index2):
                if bit == '0' and 2 * n + s_index < n_qubits:
                    qc.x(2 * n +s_index)

           
        qc.barrier()

        
    return qc

# --- Section 3: Quantum Circuit Measurement and Processing ---

def measure_quantum_circuit(qc):
    """
    Aggiunge le operazioni di misura al circuito quantistico.

    Args:
        qc (QuantumCircuit): Circuito quantistico da misurare.
    """
    n_qubits = qc.num_qubits
    qc.measure(range(n_qubits), range(n_qubits))

def neutralize_color(qc):
    """
    Applica una porta Hadamard all'ultimo qubit per neutralizzare il colore.

    Args:
        qc (QuantumCircuit): Circuito quantistico.

    Returns:
        QuantumCircuit: Circuito quantistico con il colore neutralizzato.
    """
    last_qubit = qc.num_qubits - 1
    qc.h(last_qubit)
    return qc

def apply_zz_gate(qc, qubit1, qubit2, theta):
    """
    Applica una porta ZZ tra due qubit.

    Args:
        qc (QuantumCircuit): Circuito quantistico.
        qubit1 (int): Indice del primo qubit.
        qubit2 (int): Indice del secondo qubit.
        theta (float): Angolo di rotazione.

    Returns:
        QuantumCircuit: Circuito quantistico con la porta ZZ applicata.
    """
    qc.cx(qubit1, qubit2)
    qc.rz(theta, qubit2)
    qc.cx(qubit1, qubit2)
    return qc

def apply_x_on_coordinate(qc, n, coord):
    """
    Applica una porta X sui qubit di coordinata.

    Args:
        qc (QuantumCircuit): Circuito quantistico.
        n (int): Numero di qubit per dimensione.
        coord (str): Coordinata ('x' o 'y').
    """
    if coord == 'x':
        for qubit in range(n):
            qc.x(qubit)
    elif coord == 'y':
        for qubit in range(n, 2 * n):
            qc.x(qubit)
    else:
        raise ValueError("Coordinate non riconosciute. Usare 'x' o 'y'.")

def apply_coordinate_swapping(qc, n):
    """
    Applica lo scambio di coordinate tra x e y.

    Args:
        qc (QuantumCircuit): Circuito quantistico.
        n (int): Numero di qubit per dimensione.
    """
    for i in range(n):
        x_qubit = i
        y_qubit = n + i
        qc.swap(x_qubit, y_qubit)

def rotate_90(qc, n):
    """
    Ruota l'immagine di 90 gradi.

    Args:
        qc (QuantumCircuit): Circuito quantistico.
        n (int): Numero di qubit per dimensione.

    Returns:
        QuantumCircuit: Circuito quantistico con l'immagine ruotata di 90 gradi.
    """
    apply_coordinate_swapping(qc, n)
    apply_x_on_coordinate(qc, n, 'x')
    return qc

def rotate_180(qc, n):
    """
    Ruota l'immagine di 180 gradi.

    Args:
        qc (QuantumCircuit): Circuito quantistico.
        n (int): Numero di qubit per dimensione.

    Returns:
        QuantumCircuit: Circuito quantistico con l'immagine ruotata di 180 gradi.
    """
    apply_x_on_coordinate(qc, n, 'y')
    apply_x_on_coordinate(qc, n, 'x')
    return qc

def rotate_270(qc, n):
    """
    Ruota l'immagine di 270 gradi.

    Args:
        qc (QuantumCircuit): Circuito quantistico.
        n (int): Numero di qubit per dimensione.

    Returns:
        QuantumCircuit: Circuito quantistico con l'immagine ruotata di 270 gradi.
    """
    apply_coordinate_swapping(qc, n)
    apply_x_on_coordinate(qc, n, 'y')
    return qc

def apply_color_inversion(qc):
    """
    Inverte i colori dell'immagine.

    Args:
        qc (QuantumCircuit): Circuito quantistico.

    Returns:
        QuantumCircuit: Circuito quantistico con i colori invertiti.
    """
    last_qubit = qc.num_qubits - 1
    qc.x(last_qubit)
    return qc

def apply_color_transformation(qc):
    """
    Applica una porta Z all'ultimo qubit per la trasformazione del colore.

    Args:
        qc (QuantumCircuit): Circuito quantistico.

    Returns:
        QuantumCircuit: Circuito quantistico con la trasformazione del colore.
    """
    last_qubit = qc.num_qubits - 1
    qc.z(last_qubit)
    return qc

def apply_custom_color_transformation(qc, theta):
    """
    Applica una trasformazione di colore personalizzata utilizzando una matrice unitaria.

    Args:
        qc (QuantumCircuit): Circuito quantistico.
        theta (float): Angolo di rotazione per la trasformazione del colore.

    Returns:
        QuantumCircuit: Circuito quantistico con la trasformazione del colore applicata.
    """
    cos_theta = np.cos(theta / 2)
    sin_theta = np.sin(theta / 2)
    unitary_matrix = np.array([[cos_theta, sin_theta], [sin_theta, -cos_theta]])
    last_qubit = qc.num_qubits - 1
    custom_gate = Operator(unitary_matrix)
    qc.unitary(custom_gate, [last_qubit], label=f"C(2*{theta})")
    return qc

# --- Section 4: Quantum Image Decoding ---

def decode_quantum_images(counts, n, m, normalization_factor):

    
    num_pixels_per_image = 2 ** (2 * n)
    num_images = 2 ** m
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
            values_q[j] = np.arccos(np.sqrt(p_1 / (p_0 + p_1))) * normalization_factor * (2 / np.pi)
        else:
            values_q[j] = 0
    
    num_digits = len(next(iter(values_q.keys())))
    sorted_pixel_values = {k: v for k, v in sorted(values_q.items(), key=lambda item: extract_index(item[0]))}
    
    image_size = int(np.sqrt(num_pixels_per_image))
    quantum_images = [[] for _ in range(num_images)]
    
    for index, value in sorted_pixel_values.items():
        image_index = int(index[:m], 2) if m > 0 else 0
        pixel_index = index[m:]
        row = int(pixel_index[:num_digits // 2], 2)
        col = int(pixel_index[num_digits // 2:], 2)
        if row >= image_size or col >= image_size:
            print(f"Index out of bounds: row {row}, col {col}, image_size {image_size}")
        else:
            quantum_images[image_index].append((row, col, value))
    
    decoded_images = []
    for pixel_values in quantum_images:
        image_quantum = np.zeros((image_size, image_size))
        for row, col, value in pixel_values:
            image_quantum[row, col] = value
        decoded_images.append(image_quantum)
    
    return decoded_images


def extract_index(bitstring):
    """
    Estrae l'indice da una stringa di bit.

    Args:
        bitstring (str): Stringa di bit.

    Returns:
        int: Indice estratto.
    """
    return int(bitstring, 2)

# --- Section 5: Image Visualization and Saving ---

def plot_images_strip(classic_images, quantum_strip, titles=None):

    

    num_classic = len(classic_images)
    num_quantum = len(quantum_strip)
    total_images = num_classic + num_quantum
    
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, max(num_classic, num_quantum), figure=fig)
    
    for i in range(num_classic):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(classic_images[i], cmap='gray')
        ax.axis('off')
        title = titles[i][0] if titles and len(titles) > i else "Immagine Classica"
        ax.set_title(title)
    
    for j in range(num_quantum):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(quantum_strip[j], cmap='gray')
        ax.axis('off')
        title = titles[j][1] if titles and len(titles) > j else "Immagine Quantistica"
        ax.set_title(title)
    
    plt.tight_layout()
    plt.show()

    

def save_images_as_tiff(images, base_filename):
    """
    Salva una lista di immagini come file TIFF con nomi incrementali.

    Args:
        images (list of np.ndarray): Lista delle immagini da salvare.
        base_filename (str): Nome base del file da utilizzare per il salvataggio.
    """
    for i, image in enumerate(images):
        filename = f"{base_filename}_{i+1}.tif"
        imageio.imwrite(filename, image.astype(np.uint8), format='TIFF')
        print(f"Immagine salvata come {filename}")




def apply_gp_function(qc, n, m, params):
    """
    Applica le operazioni quantistiche per calcolare (I1 - I2) / (I1 + I2).
    
    Args:
        qc (QuantumCircuit): Il circuito quantistico.
        n (int): Numero di qubit per dimensione spaziale.
        m (int): Numero di qubit per selezionare l'immagine.
        params (list of Parameter): Parametri del circuito da ottimizzare.
    """
    control_qubit = 2 * n
    color_qubit = 2 * n + m

    param_idx = 0
    for i in range(2 ** (2 * n)):
        binary_index = format(i, '0' + str(2 * n) + 'b')[::-1]
        for bit_position, bit in enumerate(binary_index):
            if bit == '0':
                qc.x(bit_position)
        
        # Utilizzare parametri variabili per le rotazioni controllate
        qc.cry(params[param_idx], control_qubit, color_qubit)  # Operazione di differenza
        param_idx += 1
        # Simulare CH gate parametrizzato
        qc.append(CRYGate(params[param_idx]), [control_qubit, color_qubit])
        qc.append(HGate(), [color_qubit])
        param_idx += 1
        
        for bit_position, bit in enumerate(binary_index):
            if bit == '0':
                qc.x(bit_position)
        
        qc.barrier()

def calculate_gp_image(I1, I2):
    """
    Calcola la funzione GP (I1 - I2) / (I1 + I2) dalle immagini classiche.
    
    Args:
        I1 (np.ndarray): Prima immagine.
        I2 (np.ndarray): Seconda immagine.
    
    Returns:
        np.ndarray: Immagine risultante dalla funzione GP.
    """
    return (I1 - I2) / (I1 + I2 + 1e-10)  # Aggiungi epsilon per evitare divisione per zero


def mse(imageA, imageB):
    """
    Calcola l'Errore Quadratico Medio (MSE) tra due immagini.
    
    Args:
        imageA (np.ndarray): Prima immagine.
        imageB (np.ndarray): Seconda immagine.
    
    Returns:
        float: Valore dell'MSE.
    """
    return np.mean((imageA - imageB) ** 2)

def psnr(imageA, imageB):
    """
    Calcola il Picco del Rapporto Segnale-Rumore (PSNR) tra due immagini.
    
    Args:
        imageA (np.ndarray): Prima immagine.
        imageB (np.ndarray): Seconda immagine.
    
    Returns:
        float: Valore del PSNR in decibel.
    """
    mse_value = mse(imageA, imageB)
    if mse_value == 0:
        return float('inf')
    max_pixel = 1.0  # Supponendo che l'immagine sia normalizzata tra 0 e 1
    return 20 * np.log10(max_pixel / np.sqrt(mse_value))

def total_variation(image):
    """
    Calcola la Total Variation (TV) di un'immagine.
    
    Args:
        image (np.ndarray): Immagine di input.
    
    Returns:
        float: Valore della TV.
    """
    tv = np.sum(np.sqrt(np.diff(image, axis=0, append=image[-1:])**2 + np.diff(image, axis=1, append=image[:,-1:])**2))
    return tv



