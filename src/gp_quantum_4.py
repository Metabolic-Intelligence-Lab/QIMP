

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:44:13 2025

@author: Giuseppe
"""

# Optimized imports
from qiskit.circuit.library import CRYGate, HGate
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.circuit import Parameter
from scipy.optimize import minimize
from FQRI_lib2 import *
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image

# Import optimized utilities
from quantum_utils import (
    ProcessingConfig, MemoryPool, SimulatorManager,
    calculate_gp_image, apply_filters_optimized, 
    visualize_channels_and_gp, get_base_quantum_circuit,
    adaptive_shots, performance_monitor, get_memory_pool,
    load_and_preprocess_image
)

# Optimized configuration
config = ProcessingConfig(
    n_spatial_qubits=4,
    m_image_qubits=1, 
    base_shots=10000,
    optimization_method='COBYLA',
    max_iterations=50
)

# Global instances
sim_manager = SimulatorManager()
memory_pool = get_memory_pool(image_size=16, max_images=50)


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

@performance_monitor
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
    # Use cached base circuit for efficiency
    qc = get_base_quantum_circuit(n, m).copy()
    encode_images_in_circuit(qc, n, m, angles_list)  # From FQRI_lib2
    params = [Parameter(f'θ{i}') for i in range(2 * (2 ** (2 * n)))]
    apply_gp_function(qc, n, m, params)
    measure_quantum_circuit(qc)
    
    # Get optimized simulator
    simulator = sim_manager.get_simulator()
    
    mse_values = []
    psnr_values = []
    tv_values = []
    combined_values = []
    times = []
    start_time = time.time()
    
    def objective_function(values):
        # Optimized parameter binding
        bound_qc = qc.assign_parameters({param: value for param, value in zip(params, values)})
        
        # Use adaptive shot count
        shots = adaptive_shots(bound_qc.depth(), combined_values[-5:] if len(combined_values) >= 5 else [], config.base_shots)
        
        # Use optimized simulator
        result = simulator.run(transpile(bound_qc, simulator), shots=shots).result()
        counts = result.get_counts()
        
        # Use memory pool for quantum images
        quantum_images = decode_quantum_images(counts, n, m, 1)
        
        # Calculate metrics
        mse_value = mse(gp_image_classic, quantum_images[1])
        psnr_value = psnr(gp_image_classic, quantum_images[1]) 
        tv_value = total_variation(quantum_images[1])
        combined_value = combined_objective(mse_value, psnr_value, tv_value, alpha, beta, gamma)
        
        # Store metrics
        mse_values.append(mse_value)
        psnr_values.append(psnr_value)
        tv_values.append(tv_value)
        combined_values.append(combined_value)
        times.append(time.time() - start_time)
        
        print(f"Iter: {len(mse_values)}, MSE: {mse_value:.4f}, PSNR: {psnr_value:.2f}dB, TV: {tv_value:.4f}, Shots: {shots}")
        return combined_value
    
  #  initial_params = np.random.uniform(-np.pi, np.pi, size=len(params))
    # Inizializzazione dei parametri su un intervallo più ampio
    initial_params = np.random.uniform(-2 * np.pi, 2 * np.pi, size=len(params))

    result = minimize(objective_function, initial_params,method='COBYLA', options={'maxiter': 10,'disp': True})
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






@performance_monitor
def main(image_paths, alpha=1, beta=1, gamma=1):
    # Optimized image preprocessing using quantum_utils
    print("🚀 Caricamento e preprocessing ottimizzato delle immagini...")
    
    # Use optimized preprocessing from quantum_utils
    processed_images = []
    for img_path in image_paths:
        red_channel, green_channel = load_and_preprocess_image(
            img_path, 
            target_size=(16, 16),
            apply_filters=True,
            sigma=1.0,
            median_size=3
        )
        processed_images.append((red_channel, green_channel))
    
    # Calculate GP using optimized function
    if len(processed_images) >= 2:
        red_channel, green_channel = processed_images[0][0], processed_images[1][1]
    else:
        red_channel, green_channel = processed_images[0]
    
    gp_image_classic = calculate_gp_image(green_channel, red_channel, G=0.5, output_format='normalized')
    
    # Optimize circuit parameters
    n = int(np.log2(red_channel.shape[0]))
    m = config.m_image_qubits

    # Generate angles using memory pool
    normalization_factor = 4096 / np.pi
    angles_list = [
        np.arccos(1 - 2 * red_channel).flatten(),
        np.arccos(1 - 2 * green_channel).flatten(),
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
