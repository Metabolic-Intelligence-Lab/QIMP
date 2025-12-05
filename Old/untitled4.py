# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:44:13 2025

@author: Giuseppe
"""

from qiskit.circuit.library import CRYGate, HGate


from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import CRYGate, HGate
from scipy.optimize import minimize
from FQRI_lib2 import *
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter

# Parametri del filtro Gaussian e mediano
sigma = 1
size = 3

# Funzione per applicare filtri (senza soglia)
def apply_filters(channel_array):
    filtered_array = gaussian_filter(channel_array, sigma=sigma)
    filtered_array = median_filter(filtered_array, size=size)
    return filtered_array

# Funzione per calcolare l'immagine GP
def calculate_gp_image(green_channel, red_channel, G=0.5):
    gp_array = (green_channel - G * red_channel) / (green_channel + G * red_channel + 1e-10)
    gp_array = np.clip(gp_array, -1, 1)
    gp_array[green_channel + G * red_channel == 0] = 0
    return gp_array


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

def combined_objective(mse_value, psnr_value, tv_value, alpha=1, beta=1, gamma=1):
    """
    Combina MSE, PSNR e TV in un'unica funzione obiettivo.
    
    Args:
        mse_value (float): Valore dell'MSE.
        psnr_value (float): Valore del PSNR.
        tv_value (float): Valore della TV.
        alpha (float): Peso per l'MSE.
        beta (float): Peso per il PSNR.
        gamma (float): Peso per la TV.
    
    Returns:
        float: Valore della funzione obiettivo combinata.
    """
    return alpha * mse_value - beta * psnr_value + gamma * tv_value

def optimize_gp_function(n, m, gp_image_classic, angles_list, alpha=1, beta=1, gamma=1):
    """
    Ottimizza la funzione GP nel circuito quantistico per minimizzare l'MSE con l'immagine GP classica.
    
    Args:
        n (int): Numero di qubit per dimensione spaziale.
        m (int): Numero di qubit per selezionare l'immagine.
        gp_image_classic (np.ndarray): Immagine GP classica.
        angles_list (list): Lista di angoli per il setup iniziale del circuito.
        alpha (float): Peso per l'MSE.
        beta (float): Peso per il PSNR.
        gamma (float): Peso per la TV.
    
    Returns:
        QuantumCircuit: Circuito quantistico ottimizzato.
        list: Lista dei valori dell'MSE durante l'ottimizzazione.
        list: Lista dei valori del PSNR durante l'ottimizzazione.
        list: Lista dei valori della TV durante l'ottimizzazione.
        list: Lista dei valori della funzione obiettivo combinata durante l'ottimizzazione.
        list: Lista dei tempi ad ogni iterazione.
    """
    qc = setup_quantum_circuit(n, m, angles_list)
    params = [Parameter(f'θ{i}') for i in range(2 * (2 ** (2 * n)))]
    apply_gp_function(qc, n, m, params)
    measure_quantum_circuit(qc)
    
    mse_values = []
    psnr_values = []
    tv_values = []
    combined_values = []
    times = []
    start_time = time.time()
    
    def objective_function(values):
        bound_qc = qc.assign_parameters({param: value for param, value in zip(params, values)})
        simulator = AerSimulator()
        result = simulator.run(transpile(bound_qc, simulator), shots=10**(n+3)).result()
        counts = result.get_counts()
        quantum_images = decode_quantum_images(counts, n, m, 1)
        mse_value = mse(gp_image_classic, quantum_images[1])  # Calcola l'MSE rispetto alla seconda immagine quantistica
        psnr_value = psnr(gp_image_classic, quantum_images[1])  # Calcola il PSNR rispetto alla seconda immagine quantistica
        tv_value = total_variation(quantum_images[1])  # Calcola la TV rispetto alla seconda immagine quantistica
        combined_value = combined_objective(mse_value, psnr_value, tv_value, alpha, beta, gamma)
        mse_values.append(mse_value)
        psnr_values.append(psnr_value)
        tv_values.append(tv_value)
        combined_values.append(combined_value)
        times.append(time.time() - start_time)
        print(f"Iteration: {len(mse_values)}, MSE: {mse_value}, PSNR: {psnr_value} dB, TV: {tv_value}, Combined: {combined_value}")
        return combined_value
    
  #  initial_params = np.random.uniform(-np.pi, np.pi, size=len(params))
    # Inizializzazione dei parametri su un intervallo più ampio
    initial_params = np.random.uniform(-2 * np.pi, 2 * np.pi, size=len(params))

    result = minimize(objective_function, initial_params,method='COBYLA', options={'maxiter': 100,'disp': True})
    optimized_params = result.x
    
    optimized_qc = qc.assign_parameters({param: value for param, value in zip(params, optimized_params)})
    return optimized_qc, mse_values, psnr_values, tv_values, combined_values, times

    
    
# Funzione per caricare e preprocessare le immagini
def load_and_preprocess_images(image_paths):
    """
    Carica le immagini, applica filtri e calcola la GP.
    
    Args:
        image_paths (list): Percorsi delle immagini in formato RGB.
    
    Returns:
        tuple: Canale rosso, canale verde, immagine GP.
    """
    if len(image_paths) < 2:
        raise ValueError("Servono almeno due immagini RGB per calcolare la GP.")

    # Carica le immagini
    red_channel = np.array(Image.open(image_paths[0]).convert('L'), dtype=np.uint16)
    green_channel = np.array(Image.open(image_paths[1]).convert('L'), dtype=np.uint16)

    # Applica i filtri
    filtered_red = apply_filters(red_channel)
    filtered_green = apply_filters(green_channel)

    # Ridimensiona le immagini a 16x16
    resized_red = np.array(Image.fromarray(filtered_red).resize((16, 16), Image.LANCZOS))
    resized_green = np.array(Image.fromarray(filtered_green).resize((16, 16), Image.LANCZOS))

    # Calcola la GP
    gp_image = calculate_gp_image(resized_green, resized_red)

    return resized_red, resized_green, gp_image






def main(image_paths, alpha=1, beta=1, gamma=1):
    # Preprocessa le immagini e calcola la GP
    print("Caricamento e preprocessing delle immagini...")
    red_channel, green_channel, gp_image_classic = load_and_preprocess_images(image_paths)
    
    # Imposta i parametri per il circuito
    n = int(np.log2(red_channel.shape[0]))  # Calcola 'n' dalla dimensione
    m = 1  # Supponiamo un'immagine GP

    # Genera gli angoli per il circuito
    normalization_factor = 4096 / np.pi
    angles_list = [
        np.arccos(1 - 2 * red_channel / 4096).flatten(),
        np.arccos(1 - 2 * green_channel / 4096).flatten(),
    ]

    print("Ottimizzazione del circuito quantistico...")
    qc, mse_values, psnr_values, tv_values, combined_values, times = optimize_gp_function(
        n, m, gp_image_classic, angles_list, alpha, beta, gamma
    )

    # Visualizza il circuito ottimizzato
    print("Visualizzazione del circuito ottimizzato...")
    optimized_circuit_diagram = qc.draw(output='mpl')
    plt.show()

    print("Simulazione del circuito quantistico ottimizzato...")
    simulator = AerSimulator()
    qc = transpile(qc, simulator)
    job = simulator.run(qc, shots=10**(n+3))
    result = job.result()

    print("Recupero dei risultati...")
    counts = result.get_counts()
    quantum_images = decode_quantum_images(counts, n, m, normalization_factor)

    # Visualizza i risultati
    titles = [("Canale Verde", "Canale Rosso"), ("Immagine GP Classica", "Immagine GP Quantistica")]
    plot_images_strip([green_channel, red_channel, gp_image_classic], quantum_images, titles=titles)

    print("Salvataggio delle immagini quantistiche...")
    save_images_as_tiff(quantum_images, "quantum_image_output")

    print("Salvataggio delle immagini classiche...")
    save_images_as_tiff([green_channel, red_channel, gp_image_classic], "classic_image_output")

    print("Fine del programma.")

if __name__ == "__main__":
    # Percorsi delle immagini RGB
    image_paths = ["IMMAGINI PER QUANTUM/trainQML/Train_QML_16/membraneStack_Sample011_L_UV_DC_001rbc0DM2_ch01_16x16.tif", "IMMAGINI PER QUANTUM/trainQML/Train_QML_16/membraneStack_Sample011_L_UV_DC_001rbc0DM2_ch02_16x16.tif"]
    main(image_paths, alpha=0.001, beta=0.1, gamma=0.1)
