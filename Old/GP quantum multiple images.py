from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import CRYGate, HGate
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import time
from FQRI_lib2 import *



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
    return alpha * mse_value - beta * psnr_value + gamma * tv_value

def optimize_gp_function(n, m, training_images, angles_list, alpha=1, beta=1, gamma=1):
    qc = setup_quantum_circuit(n, m, angles_list)
    params = [Parameter(f'θ{i}') for i in range(2 * (2 ** (2 * n)))]
    apply_gp_function(qc, n, m, params)
    measure_quantum_circuit(qc)

    mse_values, psnr_values, tv_values, combined_values, times = [], [], [], [], []
    start_time = time.time()
    
    def objective_function(values):
        bound_qc = qc.assign_parameters({param: value for param, value in zip(params, values)})
        simulator = AerSimulator()
        result = simulator.run(transpile(bound_qc, simulator), shots=10**(n + 3)).result()
        counts = result.get_counts()
        
        total_mse, total_psnr, total_tv = 0, 0, 0
        for image1, image2 in training_images:
            quantum_images = decode_quantum_images(counts, n, m, 1)
            mse_value = mse(image1, quantum_images[1])
            psnr_value = psnr(image1, quantum_images[1])
            tv_value = total_variation(quantum_images[1])
            total_mse += mse_value
            total_psnr += psnr_value
            total_tv += tv_value
        
        avg_mse = total_mse / len(training_images)
        avg_psnr = total_psnr / len(training_images)
        avg_tv = total_tv / len(training_images)
        combined_value = combined_objective(avg_mse, avg_psnr, avg_tv, alpha, beta, gamma)

        mse_values.append(avg_mse)
        psnr_values.append(avg_psnr)
        tv_values.append(avg_tv)
        combined_values.append(combined_value)
        times.append(time.time() - start_time)
        
        return combined_value

    initial_params = np.random.uniform(-2 * np.pi, 2 * np.pi, size=len(params))
    result = minimize(objective_function, initial_params, method='COBYLA', options={'maxiter': 500, 'disp': True})
    optimized_params = result.x
    
    optimized_qc = qc.assign_parameters({param: value for param, value in zip(params, optimized_params)})
    return optimized_qc, mse_values, psnr_values, tv_values, combined_values, times

# --- Main Function ---

def main(image_pairs, alpha=1, beta=1, gamma=1):
    print("Loading and encoding training images...")
    angles_lists = []
    training_images = []
    n, m, normalization_factor = None, None, None

    for image_names in image_pairs:
        angles_list, n, m, normalization_factor = load_and_encode_images(image_names)
        angles_lists.append(angles_list)

        classic_images = [np.array(angles).reshape(int(np.sqrt(2 ** (2 * n))), int(np.sqrt(2 ** (2 * n)))) * normalization_factor / np.pi for angles in angles_list]
        gp_image_classic = calculate_gp_image(classic_images[0], classic_images[1])
        training_images.append((classic_images[0], classic_images[1]))

    print("Optimizing quantum circuit for multiple training images...")
    qc, mse_values, psnr_values, tv_values, combined_values, times = optimize_gp_function(n, m, training_images, angles_lists[0], alpha, beta, gamma)

    # Visualizzazione dell'andamento dell'ottimizzazione
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('MSE', color='tab:blue')
    ax1.plot(times, mse_values, marker='o', linestyle='-', color='tab:blue', label='MSE')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('PSNR (dB)', color='tab:orange')
    ax2.plot(times, psnr_values, marker='x', linestyle='-', color='tab:orange', label='PSNR')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title('Optimization Progress Over Time')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(times, combined_values, marker='s', linestyle='-', color='tab:green', label='Combined Objective')
    plt.xlabel('Time (s)')
    plt.ylabel('Combined Objective')
    plt.title('Combined Objective Progress')
    plt.grid(True)
    plt.show()

    # Visualizzazione delle immagini GP classiche e quantistiche
    for idx, (image1, image2) in enumerate(training_images):
        gp_image_classic = calculate_gp_image(image1, image2)
        simulator = AerSimulator()
        qc_transpiled = transpile(qc, simulator)
        job = simulator.run(qc_transpiled, shots=10**(n + 3))
        result = job.result()
        counts = result.get_counts()
        quantum_images = decode_quantum_images(counts, n, m, normalization_factor)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(gp_image_classic, cmap='gray')
        ax1.set_title(f'GP Classica (Coppia Immagini {idx + 1})')
        ax1.axis('off')

        ax2.imshow(quantum_images[1], cmap='gray')
        ax2.set_title(f'GP Quantistica (Coppia Immagini {idx + 1})')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()
        
        print(f"MSE tra GP Classica e GP Quantistica (Coppia {idx + 1}): {mse(gp_image_classic, quantum_images[1])}")
        print(f"PSNR tra GP Classica e GP Quantistica (Coppia {idx + 1}): {psnr(gp_image_classic, quantum_images[1])} dB")
        print(f"Total Variation della GP Quantistica (Coppia {idx + 1}): {total_variation(quantum_images[1])}")

if __name__ == "__main__":
    image_pairs = [
        ["ratioch1_8-1.tif", "ratioch2_8-1.tif"],
        ["ratiotest2_ch1.tif", "ratiotest2_ch2.tif"],
        # Aggiungi altre coppie di immagini se necessario
    ]
    main(image_pairs, alpha=0.001, beta=0.1, gamma=0.1)

