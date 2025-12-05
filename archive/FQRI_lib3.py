# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:24:39 2025

@author: Giuseppe
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 16:36:53 2025

@author: Giuseppe
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 16:20:21 2025

@author: Giuseppe
"""

from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import CRYGate, HGate
from scipy.optimize import minimize
from FQRI_lib2 import *
import matplotlib.pyplot as plt
import numpy as np
import time
import os
#from qiskit.providers.aer import AerError


def visualize_circuit(qc):
    """
    Visualizza il circuito quantistico ottimizzato.
    
    Args:
        qc (QuantumCircuit): Il circuito quantistico da visualizzare.
    """
    print("Visualizzazione del circuito ottimizzato...")
    optimized_circuit_diagram = qc.draw(output='mpl')
    plt.show()

def simulate_circuit(qc, n, simulator_device='GPU'):
    """
    Gestione automatica GPU/CPU
    """
    try:
        simulator = AerSimulator(device=simulator_device)
        # Test transpilazione per verificare compatibilità GPU
        test_circuit = transpile(qc, simulator)
        simulator.run(test_circuit, shots=1).result()
    except:
        print("GPU non disponibile, si utilizza la CPU")
        simulator = AerSimulator(device='CPU')
    
    qc_transpiled = transpile(qc, simulator)
    job = simulator.run(qc_transpiled, shots=10**(n + 3))
    return job.result().get_counts()

def visualize_images(classic_images, quantum_images, gp_image_classic, quantum_gp_image):
    """
    Visualizza immagini classiche, quantistiche e confronta GP classica e quantistica.
    
    Args:
        classic_images (list): Lista di immagini classiche.
        quantum_images (list): Lista di immagini quantistiche.
        gp_image_classic (ndarray): Immagine GP classica.
        quantum_gp_image (ndarray): Immagine GP quantistica ricostruita.
    """
    titles = [(f"Immagine Classica {i+1}", f"Immagine Quantistica Ricostruita {i+1}") for i in range(len(classic_images))]
    plot_images_strip(classic_images, quantum_images, titles=titles)
    
    # Visualizza GP classica e quantistica
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(gp_image_classic, cmap='gray')
    axs[0].set_title('Immagine GP Classica')
    axs[0].axis('off')
    axs[1].imshow(quantum_gp_image, cmap='gray')
    axs[1].set_title('Immagine GP Quantistica')
    axs[1].axis('off')
    plt.tight_layout()
    plt.show()
    
def save_images(classic_images, quantum_images, output_directory):
    """
    Salva immagini classiche e quantistiche in directory separate.
    
    Args:
        classic_images (list): Lista di immagini classiche.
        quantum_images (list): Lista di immagini quantistiche.
        output_directory (str): Directory di output.
    """
    print("Salvataggio delle immagini quantistiche...")
    save_images_as_tiff(quantum_images, os.path.join(output_directory, "quantum_image_output"))
    print("Salvataggio delle immagini classiche...")
    save_images_as_tiff(classic_images, os.path.join(output_directory, "classic_image_output"))
    
def plot_optimization_metrics(times, mse_values, psnr_values, combined_values):
    """
    Visualizza grafici delle metriche di ottimizzazione (MSE, PSNR, funzione obiettivo combinata).
    
    Args:
        times (list): Lista di tempi.
        mse_values (list): Lista di valori MSE.
        psnr_values (list): Lista di valori PSNR.
        combined_values (list): Lista di valori della funzione obiettivo combinata.
    """
    # Grafico dell'MSE e PSNR
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Tempo (s)')
    ax1.set_ylabel('MSE', color='tab:blue')
    ax1.plot(times, mse_values, marker='o', linestyle='-', color='tab:blue', label='MSE')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('PSNR (dB)', color='tab:orange')
    ax2.plot(times, psnr_values, marker='x', linestyle='-', color='tab:orange', label='PSNR')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.tight_layout()
    plt.title('Andamento dell\'ottimizzazione nel tempo')
    plt.grid(True)
    plt.show()
    
    # Grafico della funzione obiettivo combinata
    plt.figure(figsize=(10, 5))
    plt.plot(times, combined_values, marker='s', linestyle='-', color='tab:green', label='Combined Objective')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Combined Objective')
    plt.title('Andamento della funzione obiettivo combinata nel tempo')
    plt.grid(True)
    plt.show()


def load_and_encode_image_pairs(pairs):
    """
    Carica le coppie di immagini dai file, combina i due canali e li codifica in una lista di angoli.

    Args:
        pairs (list of tuples): Lista di tuple (ch01_file, ch02_file, gp_file), 
                                con i percorsi dei file immagine per ch01 e ch02.

    Returns:
        tuple: Lista di angoli codificati, numero di qubit per dimensione, numero di immagini, fattore di normalizzazione.
    """
    angles_list = []
    n = None
    normalization_factor = 4096.0  # Valore di normalizzazione (può essere modificato se necessario)

    for ch01_file, ch02_file, _ in pairs:
        # Carica le immagini dei due canali
        ch01_image = np.array(Image.open(ch01_file))
        ch02_image = np.array(Image.open(ch02_file))
        
        # Combina le immagini (ad esempio, media pixel a pixel)
        combined_image = (ch01_image + ch02_image) / 2.0
        
        # Normalizza e codifica
        combined_image = combined_image / normalization_factor
        combined_image = np.clip(1 - 2 * combined_image, -1, 1)
        angles = np.arccos(combined_image)
        angles_flat = angles.flatten()

        angles_list.append(angles_flat)

        if n is None:
            num_pixels = angles_flat.size
            n = int(math.log2(math.sqrt(num_pixels)))

    m = int(math.log2(len(pairs)))  # Numero di immagini log2(len(pairs))
    return angles_list, n, m, normalization_factor


def pair_input_output_files(input_files, output_files):
    """
    Pair input files (ch01 and ch02) with their corresponding output file (GP).
    
    Args:
        input_files (list): List of input file paths.
        output_files (list): List of output file paths.
        
    Returns:
        list of tuples: Each tuple contains (ch01_file, ch02_file, gp_file).
    """
    pairs = []
    for output_file in output_files:
        base_name = os.path.basename(output_file).replace("_GP.tif", "")
        ch01_file = [f for f in input_files if f"{base_name}_ch01_16x16.tif" in f]
        ch02_file = [f for f in input_files if f"{base_name}_ch02_16x16.tif" in f]
        if ch01_file and ch02_file:
            pairs.append((ch01_file[0], ch02_file[0], output_file))
    return pairs


def gpu_check():
    try:
        # Prova a inizializzare il simulatore GPU
        simulator = AerSimulator(device='GPU')
        print("GPU disponibile per Qiskit AerSimulator!")
        return True
    except AerError as e:
        print("GPU non disponibile per Qiskit AerSimulator.")
        print(f"Errore: {e}")
        return False

# Esegui il controllo
if gpu_check():
    print("Pronto per simulare con la GPU!")
else:
    print("Continueremo a usare la CPU.")
    
    


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
    
    initial_params = np.random.uniform(-2 * np.pi, 2 * np.pi, size=len(params))

    result = minimize(objective_function, initial_params, method='COBYLA', options={'maxiter': 5, 'disp': True})
    optimized_params = result.x
    
    optimized_qc = qc.assign_parameters({param: value for param, value in zip(params, optimized_params)})
    return optimized_qc, mse_values, psnr_values, tv_values, combined_values, times

def process_new_images(image_names, optimized_qc, n, m, normalization_factor):
    # Carica e codifica le nuove immagini
    new_angles_list, _, _, _ = load_and_encode_images(image_names)
    
    # Carica le immagini classiche per il confronto
    classic_images = [np.array(angles).reshape(int(np.sqrt(2 ** (2 * n))), int(np.sqrt(2 ** (2 * n)))) * normalization_factor / np.pi for angles in new_angles_list]
    
    # Simula il circuito quantistico con le nuove immagini
    simulator = AerSimulator(device='GPU')
    optimized_qc_transpiled = transpile(optimized_qc, simulator)
    job = simulator.run(optimized_qc_transpiled, shots=10**(n + 3))
    result = job.result()
    counts = result.get_counts()
    
    # Decodifica i risultati quantistici
    quantum_images = decode_quantum_images(counts, n, m, normalization_factor)
    
    # Calcola le metriche di somiglianza
    gp_image_classic = None
    for image_name in image_names:
        gp_name = image_name.replace('_ch01_16x16.tif', '_GP.tif').replace('_ch02_16x16.tif', '_GP.tif')
        gp_path = os.path.join(output_directory, gp_name)
        if os.path.exists(gp_path):
            gp_image_classic = imageio.imread(gp_path)
            break
    if gp_image_classic is None:
        raise FileNotFoundError(f'GP image not found for the given input file: {image_name}')  # Carica l'immagine GP classica da file
    mse_value = mse(gp_image_classic, quantum_images[1])
    psnr_value = psnr(gp_image_classic, quantum_images[1])
    tv_value = total_variation(quantum_images[1])
    
    # Visualizza i risultati
    titles = [(f"Nuova Immagine Classica {i+1}", f"Nuova Immagine Quantistica {i+1}") for i in range(len(image_names))]
    plot_images_strip(classic_images, quantum_images, titles=titles)
    
    # Visualizza l'immagine GP classica e quantistica
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(gp_image_classic, cmap='gray')
    ax1.set_title('GP Classica (Nuove Immagini)')
    ax1.axis('off')
    ax2.imshow(quantum_images[1], cmap='gray')
    ax2.set_title('GP Quantistica (Nuove Immagini)')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"MSE tra GP Classica e GP Quantistica: {mse_value}")
    print(f"PSNR tra GP Classica e GP Quantistica: {psnr_value} dB")
    print(f"Total Variation della GP Quantistica: {tv_value}")
    
    return quantum_images, mse_value, psnr_value, tv_value



# Imposta la GPU da utilizzare (ad esempio, GPU 0 o GPU 1)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Sostituisci "0" con l'indice della GPU desiderata
    
# Directory di input e output
input_directory = 'IMMAGINI PER QUANTUM\\trainQML\\Train_QML_16'
output_directory = 'IMMAGINI PER QUANTUM\\trainQML\\GP_output'




def main(image_directory, alpha=1, beta=1, gamma=1):
    print("Caricamento delle immagini e corrispondenza input-output...")
    input_files = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if "ch" in file]
    output_files = [os.path.join(output_directory, file) for file in os.listdir(output_directory) if "GP" in file]
    pairs = pair_input_output_files(input_files, output_files)
    
    if not pairs:
        raise ValueError("Non sono state trovate corrispondenze tra input e output!")
    
    angles_list, n, m, normalization_factor = load_and_encode_image_pairs(pairs)
    gp_image_classic = imageio.imread(pairs[0][2])
    
    print("Ottimizzazione del circuito quantistico...")
    qc, mse_values, psnr_values, tv_values, combined_values, times = optimize_gp_function(
        n, m, gp_image_classic, angles_list, alpha, beta, gamma
    )
    
    visualize_circuit(qc)
    
    counts = simulate_circuit(qc, n)
    quantum_images = decode_quantum_images(counts, n, m, normalization_factor)
    classic_images = [np.array(Image.open(pair[0])) for pair in pairs]
    
    save_images(classic_images, quantum_images, output_directory)
    visualize_images(classic_images, quantum_images, gp_image_classic, quantum_images[1])
    plot_optimization_metrics(times, mse_values, psnr_values, combined_values)
    
    print("Processamento completato!")

if __name__ == "__main__":
    main(input_directory, alpha=0.001, beta=0.1, gamma=0.1) 