# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:51:58 2024

@author: miste
"""

from qiskit_aer import AerSimulator
from qiskit import transpile, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import CRYGate, HGate
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import time
from FQRI_lib2 import *
# Funzione per calcolare la funzione GP
def calculate_gp_image(I1, I2):
    return (I1 - I2) / (I1 + I2 + 1e-10)

# Funzione per applicare la funzione GP al circuito quantistico
def apply_gp_function(qc, n, m, params):
    control_qubit = 2 * n
    color_qubit = 2 * n + m
    param_idx = 0
    for i in range(2 ** (2 * n)):
        binary_index = format(i, '0' + str(2 * n) + 'b')[::-1]
        for bit_position, bit in enumerate(binary_index):
            if bit == '0':
                qc.x(bit_position)
        qc.cry(params[param_idx], control_qubit, color_qubit)
        param_idx += 1
        qc.append(CRYGate(params[param_idx]), [control_qubit, color_qubit])
        qc.append(HGate(), [color_qubit])
        param_idx += 1
        for bit_position, bit in enumerate(binary_index):
            if bit == '0':
                qc.x(bit_position)
        qc.barrier()

# Funzione obiettivo combinata
def combined_objective(mse_value, psnr_value, tv_value, alpha=1, beta=1, gamma=1):
    return alpha * mse_value - beta * psnr_value + gamma * tv_value

# Ottimizzazione della funzione GP
def optimize_gp_function(n, m, gp_image_classic, angles_list, alpha=1, beta=1, gamma=1):
    qc = setup_quantum_circuit(n, m, angles_list)
    params = [Parameter(f'θ{i}') for i in range(2 * (2 ** (2 * n)))]
    apply_gp_function(qc, n, m, params)
    measure_quantum_circuit(qc)
    mse_values, psnr_values, tv_values, combined_values, times = [], [], [], [], []
    start_time = time.time()

    def objective_function(values):
        bound_qc = qc.assign_parameters({param: value for param, value in zip(params, values)})
        simulator = AerSimulator()
        result = simulator.run(transpile(bound_qc, simulator), shots=10**(n+3)).result()
        counts = result.get_counts()
        quantum_images = decode_quantum_images(counts, n, m, 1)
        mse_value = mse(gp_image_classic, quantum_images[1])
        psnr_value = psnr(gp_image_classic, quantum_images[1])
        tv_value = total_variation(quantum_images[1])
        combined_value = combined_objective(mse_value, psnr_value, tv_value, alpha, beta, gamma)
        mse_values.append(mse_value)
        psnr_values.append(psnr_value)
        tv_values.append(tv_value)
        combined_values.append(combined_value)
        times.append(time.time() - start_time)
        print(f"Iteration: {len(mse_values)}, MSE: {mse_value}, PSNR: {psnr_value} dB, TV: {tv_value}, Combined: {combined_value}")
        return combined_value
    
    initial_params = np.random.uniform(-2 * np.pi, 2 * np.pi, size=len(params))
    result = minimize(objective_function, initial_params, method='COBYLA', options={'maxiter': 500, 'disp': True})
    optimized_params = result.x
    optimized_qc = qc.assign_parameters({param: value for param, value in zip(params, optimized_params)})
    return optimized_qc, mse_values, psnr_values, tv_values, combined_values, times

# Funzione principale
def main(image_names, alpha=1, beta=1, gamma=1):
    angles_list, n, m, normalization_factor = load_and_encode_images(image_names)
    classic_images = [np.array(angles) * normalization_factor / np.pi for angles in angles_list]
    classic_images = [image.reshape(int(np.sqrt(2 ** (2 * n))), int(np.sqrt(2 ** (2 * n)))) for image in classic_images]
    gp_image_classic = calculate_gp_image(classic_images[0], classic_images[1])
    
    qc, mse_values, psnr_values, tv_values, combined_values, times = optimize_gp_function(n, m, gp_image_classic, angles_list, alpha, beta, gamma)
    print("Ottimizzazione completata.")

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
    
    new_image_names = ["ratiotest2_ch1.tif", "ratiotest2_ch2.tif"]
    quantum_results, mse_value, psnr_value, tv_value = process_new_images(new_image_names, qc, n, m, normalization_factor)
    return qc.draw(output='mpl')

if __name__ == "__main__":
    image_names = ["ratioch1_8-1.tif", "ratioch2_8-1.tif"]
    main(image_names, alpha=0.001, beta=0.1, gamma=0.1)
