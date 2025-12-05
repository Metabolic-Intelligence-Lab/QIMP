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

def gpu_check():
    try:
        # Prova a inizializzare il simulatore GPU
        simulator = AerSimulator(device='GPU')
        print("GPU disponibile per Qiskit AerSimulator!")
        return 'GPU'
    except Exception as e:
        print("GPU non disponibile per Qiskit AerSimulator.")
        print(f"Errore: {e}")
        return 'CPU'


def apply_gp_function(qc, n, m, params):
    """
    Applica le operazioni quantistiche per calcolare (I1 - I2) / (I1 + I2) e genera la Quantum Strip.
    
    Args:
        qc (QuantumCircuit): Il circuito quantistico.
        n (int): Numero di qubit per dimensione spaziale.
        m (int): Numero di qubit per selezionare l'immagine (in questo caso 1 immagine di output).
        params (list of Parameter): Parametri del circuito da ottimizzare.
    """
    control_qubit = 2 * n
    color_qubit = 2 * n + m  # Questi qubit rappresentano il colore

    param_idx = 0
    for i in range(2 ** (2 * n)):
        binary_index = format(i, '0' + str(2 * n) + 'b')[::-1]
        for bit_position, bit in enumerate(binary_index):
            if bit == '0':
                qc.x(bit_position)
        
        # Calcolo della differenza e somma controllata
        qc.cry(params[param_idx], control_qubit, color_qubit)  # Differenza (I1 - I2)
        param_idx += 1
        qc.cry(params[param_idx], control_qubit, color_qubit)  # Somma (I1 + I2)
        param_idx += 1
        
        # Neutralizzare il colore per l'immagine zero
        qc.x(color_qubit)  # Imposta l'immagine zero
        qc.h(color_qubit)
        
        for bit_position, bit in enumerate(binary_index):
            if bit == '0':
                qc.x(bit_position)
        
        qc.barrier()

def decode_quantum_images(counts, n, m, normalization_factor):
    """
    Decodifica le immagini quantistiche dai conteggi per generare la Quantum Strip.
    Args:
        counts (dict): Conteggi ottenuti dal simulatore.
        n (int): Numero di qubit per dimensione spaziale.
        m (int): Numero di qubit per selezionare l'immagine.
        normalization_factor (float): Fattore di normalizzazione.
    
    Returns:
        list of np.ndarray: Quantum Strip (una lista di due immagini).
    """
    image_size = 2 ** n
    decoded_images = [np.zeros((image_size, image_size)), np.zeros((image_size, image_size))]

    total_shots = sum(counts.values())
    for state, count in counts.items():
        reversed_state = state[::-1]
        pos_bits = reversed_state[m:2*n + m]
        
        y_bin = pos_bits[:n][::-1]
        x_bin = pos_bits[n:][::-1]
        
        x = int(x_bin, 2)
        y = int(y_bin, 2)
        
        color_bit = reversed_state[-1]
        value = (1 if color_bit == '1' else 0) * normalization_factor
        
        if color_bit == '1':
            decoded_images[0][y, x] += count / total_shots  # Immagine GP
        else:
            decoded_images[1][y, x] = 0  # Immagine zero (nessun incremento)

    # Normalizzazione
    for img in decoded_images:
        img /= np.max(img) if np.max(img) > 0 else 1
        img = np.clip(img, 0, 1)

    return decoded_images




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

    result = minimize(objective_function, initial_params, method='COBYLA', options={'maxiter': 2, 'disp': True})
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
def load_images_and_calculate_gp(image_directory, normalization_factor):
    """
    Carica due immagini dalla directory e calcola la GP classica.

    Args:
        image_directory (str): Directory contenente le immagini.
        normalization_factor (float): Fattore di normalizzazione per le immagini.

    Returns:
        tuple: Lista dei nomi delle immagini, immagini classiche, e immagine GP classica.
    """
    image_names = [os.path.join(image_directory, image) for image in os.listdir(image_directory) if image.endswith('.tif')]
    if len(image_names) != 2:
        raise ValueError("La directory deve contenere esattamente due immagini per calcolare la GP.")

    # Carica le immagini
    image1 = imageio.imread(image_names[0]) / normalization_factor
    image2 = imageio.imread(image_names[1]) / normalization_factor

    # Calcola la GP
    gp_image_classic = calculate_gp_image(image1, image2)

    return image_names, [image1, image2], gp_image_classic


def visualize_images(image1, image2, gp_image_classic):
    """
    Visualizza le immagini classiche e la GP classica.

    Args:
        image1 (np.ndarray): Prima immagine classica.
        image2 (np.ndarray): Seconda immagine classica.
        gp_image_classic (np.ndarray): Immagine GP classica.
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image1, cmap='gray')
    axs[0].set_title('Immagine 1')
    axs[0].axis('off')

    axs[1].imshow(image2, cmap='gray')
    axs[1].set_title('Immagine 2')
    axs[1].axis('off')

    axs[2].imshow(gp_image_classic, cmap='gray')
    axs[2].set_title('GP Classica')
    axs[2].axis('off')
    plt.tight_layout()
    plt.show()


def run_quantum_simulation(image_names, gp_image_classic, normalization_factor, alpha, beta, gamma):
    """
    Ottimizza il circuito quantistico e simula la Quantum Strip.

    Args:
        image_names (list): Lista dei nomi delle immagini.
        gp_image_classic (np.ndarray): Immagine GP classica.
        normalization_factor (float): Fattore di normalizzazione per le immagini.
        alpha (float): Peso per l'MSE.
        beta (float): Peso per il PSNR.
        gamma (float): Peso per la TV.

    Returns:
        tuple: Quantum circuit, Quantum Strip, e dati del training.
    """
    angles_list, n, m, _ = load_and_encode_images(image_names)

    # Ottimizza il circuito quantistico
    qc, mse_values, psnr_values, tv_values, combined_values, times = optimize_gp_function(n, m, gp_image_classic, angles_list, alpha, beta, gamma)

    # Usa sempre la CPU per il simulatore
    simulator = AerSimulator(device='CPU')

    # Simulazione del circuito ottimizzato
    qc = transpile(qc, simulator)
    job = simulator.run(qc, shots=10**(n+3))
    result = job.result()

    # Decodifica della Quantum Strip
    counts = result.get_counts()
    quantum_images = decode_quantum_images(counts, n, m, normalization_factor)

    return qc, quantum_images, (mse_values, psnr_values, tv_values, combined_values, times)


def visualize_training(mse_values, psnr_values, times):
    """
    Visualizza l'andamento del training.

    Args:
        mse_values (list): Valori di MSE durante il training.
        psnr_values (list): Valori di PSNR durante il training.
        times (list): Tempi associati a ciascuna iterazione.
    """
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

    
    
# Directory di input e output
input_directory = 'IMMAGINI PER QUANTUM/trainQML/Train_QML_16'
output_directory = '\IMMAGINI PER QUANTUM\trainQML\GP_output'


def main(image_directory, alpha=1, beta=1, gamma=1):
    normalization_factor = 4096  # Definito una volta per evitare ripetizioni

    print("Caricamento delle immagini e calcolo della GP...")
    image_names, classic_images, gp_image_classic = load_images_and_calculate_gp(image_directory, normalization_factor)

    print("Visualizzazione delle immagini classiche e della GP classica...")
    visualize_images(classic_images[0], classic_images[1], gp_image_classic)

    print("Ottimizzazione e simulazione del circuito quantistico...")
    qc, quantum_images, training_data = run_quantum_simulation(image_names, gp_image_classic, normalization_factor, alpha, beta, gamma)

    print("Visualizzazione della Quantum Strip...")
    titles = [('GP Quantistica', 'Immagine Zero')]
    plot_images_strip([gp_image_classic], quantum_images, titles=titles)

    print("Salvataggio delle immagini...")
    save_images_as_tiff(quantum_images, os.path.join(output_directory, "quantum_strip"))

    print("Visualizzazione dell'andamento del training...")
    visualize_training(*training_data[:3])

    print("Simulazione completata!")



if __name__ == "__main__":
    main(input_directory, alpha=0.001, beta=0.1, gamma=0.1) 