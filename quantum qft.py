from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import CRYGate, HGate, QFT
from scipy.optimize import minimize
from FQRI_lib2 import *
import matplotlib.pyplot as plt
import numpy as np
import time

# Funzione per applicare la QFT e la QFT inversa
def apply_qft(qc, n):
    """
    Applica la QFT sui primi n qubit.
    
    Args:
        qc (QuantumCircuit): Circuito quantistico.
        n (int): Numero di qubit su cui applicare la QFT.
    """
    qc.append(QFT(n).to_gate(), range(n))  # Applica la QFT

def apply_inverse_qft(qc, n):
    """
    Applica la QFT inversa sui primi n qubit.
    
    Args:
        qc (QuantumCircuit): Circuito quantistico.
        n (int): Numero di qubit su cui applicare la QFT inversa.
    """
    qc.append(QFT(n, inverse=True).to_gate(), range(n))  # Applica la QFT inversa

# Modifica della funzione per applicare la QFT durante il calcolo di (I1 - I2) / (I1 + I2)
def apply_gp_function(qc, n, m, params):
    """
    Applica le operazioni quantistiche per calcolare (I1 - I2) / (I1 + I2) usando la QFT.
    
    Args:
        qc (QuantumCircuit): Circuito quantistico.
        n (int): Numero di qubit per dimensione spaziale.
        m (int): Numero di qubit per selezionare l'immagine.
        params (list of Parameter): Parametri del circuito da ottimizzare.
    """
    control_qubit = 2 * n
    color_qubit = 2 * n + m

    # Applica la QFT sui qubit che rappresentano I1 e I2
    apply_qft(qc, n)

    param_idx = 0
    for i in range(2 ** (2 * n)):
        binary_index = format(i, '0' + str(2 * n) + 'b')[::-1]
        for bit_position, bit in enumerate(binary_index):
            if bit == '0':
                qc.x(bit_position)
        
        # Operazioni con CRY gate parametrizzati per (I1 - I2) / (I1 + I2)
        qc.cry(params[param_idx], control_qubit, color_qubit)  # Operazione di differenza
        param_idx += 1

        qc.append(CRYGate(params[param_idx]), [control_qubit, color_qubit])
        qc.append(HGate(), [color_qubit])
        param_idx += 1

        for bit_position, bit in enumerate(binary_index):
            if bit == '0':
                qc.x(bit_position)

        qc.barrier()

    # Applica la QFT inversa per tornare nel dominio classico
    apply_inverse_qft(qc, n)



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
# Funzione per ottimizzare il circuito che calcola la funzione GP con la QFT
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

    initial_params = np.random.uniform(-np.pi, np.pi, size=len(params))
    result = minimize(objective_function, initial_params, method='COBYLA', options={'maxiter': 500,'disp': True})
    optimized_params = result.x

    optimized_qc = qc.assign_parameters({param: value for param, value in zip(params, optimized_params)})
    return optimized_qc, mse_values, psnr_values, tv_values, combined_values, times

def process_new_images(image_names, optimized_qc, n, m, normalization_factor):
    # Carica e codifica le nuove immagini
    new_angles_list, _, _, _ = load_and_encode_images(image_names)
    
    # Carica le immagini classiche per il confronto
    classic_images = [np.array(angles).reshape(int(np.sqrt(2 ** (2 * n))), int(np.sqrt(2 ** (2 * n)))) * normalization_factor / np.pi for angles in new_angles_list]
    
    # Calcola l'immagine GP classica
    gp_image_classic = calculate_gp_image(classic_images[0], classic_images[1])
    
    # Simula il circuito quantistico con le nuove immagini
    simulator = AerSimulator()
    optimized_qc_transpiled = transpile(optimized_qc, simulator)
    job = simulator.run(optimized_qc_transpiled, shots=10**(n + 3))
    result = job.result()
    counts = result.get_counts()
    
    # Decodifica i risultati quantistici
    quantum_images = decode_quantum_images(counts, n, m, normalization_factor)
    
    # Calcola le metriche di somiglianza
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







def main(image_names, alpha=1, beta=1, gamma=1):
    print("Caricamento delle immagini e codifica degli angoli...")
    angles_list, n, m, normalization_factor = load_and_encode_images(image_names)
    
    # Calcola la GP dalle immagini classiche
    classic_images = [np.array(angles) * normalization_factor / np.pi for angles in angles_list]
    classic_images = [image.reshape(int(np.sqrt(2 ** (2 * n))), int(np.sqrt(2 ** (2 * n)))) for image in classic_images]
    image1 = classic_images[0]
    image2 = classic_images[1]
    gp_image_classic = calculate_gp_image(image1, image2)
    
    print("Ottimizzazione del circuito quantistico...")
    qc, mse_values, psnr_values, tv_values, combined_values, times = optimize_gp_function(n, m, gp_image_classic, angles_list, alpha, beta, gamma)
   
    # Visualizza il circuito ottimizzato
    print("Visualizzazione del circuito ottimizzato...")
    optimized_circuit_diagram = qc.draw(output='mpl')  # Usare Matplotlib per visualizzare il circuito
    plt.show()  # Mostra la visualizzazione del circuito
    
    print("Simulazione del circuito quantistico ottimizzato...")
    simulator = AerSimulator()
    qc = transpile(qc, simulator)
    job = simulator.run(qc, shots=10**(n+3))
    result = job.result()
    
    print("Recupero dei risultati...")
    experiment = qc.name if qc.name in result.results else result.results[0].header.name
    counts = result.get_counts(experiment)
    
    print("Decodifica dei risultati...")
    quantum_images = decode_quantum_images(counts, n, m, normalization_factor)
    #quantum_images = [1 - image for image in quantum_images]
    titles = [(f"Immagine Classica {i+1}", f"Immagine Quantistica Ricostruita {i+1}") for i in range(len(image_names))]
    
    plot_images_strip(classic_images, quantum_images, titles=titles)
    
    print("Salvataggio delle immagini quantistiche...")
    save_images_as_tiff(quantum_images, "quantum_image_output")
    
    print("Salvataggio delle immagini classiche...")
    save_images_as_tiff(classic_images, "classic_image_output")
 
    # Visualizza la GP classica e le immagini quantistiche ricostruite
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(gp_image_classic, cmap='gray')
    axs[0].set_title('Immagine GP Classica')
    axs[0].axis('off')
    
    axs[1].imshow(quantum_images[1], cmap='gray')
    axs[1].set_title('Immagine GP Quantistica 2')
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.show()

    
    mse_value_2 = mse(gp_image_classic, quantum_images[1])
    psnr_value_2 = psnr(gp_image_classic, quantum_images[1])
   
    print(f"MSE tra GP Classica e GP Quantistica 2: {mse_value_2}")
    print(f"PSNR tra GP Classica e GP Quantistica 2: {psnr_value_2} dB")
    
    # Grafico dell'MSE, del PSNR e della funzione obiettivo combinata in funzione del tempo
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
    
    # Grafico della funzione obiettivo combinata in funzione del tempo
    plt.figure(figsize=(10, 5))
    plt.plot(times, combined_values, marker='s', linestyle='-', color='tab:green', label='Combined Objective')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Combined Objective')
    plt.title('Andamento della funzione obiettivo combinata nel tempo')
    plt.grid(True)
    plt.show()
    
    new_image_names = ["ratiotest2_ch1.tif", "ratiotest2_ch2.tif"]
    quantum_results, mse_value, psnr_value, tv_value = process_new_images(new_image_names, qc, n, m, normalization_factor)
    return optimized_circuit_diagram 


if __name__ == "__main__":
    image_names = ["ratioch1_8-1.tif", "ratioch2_8-1.tif"]
    main(image_names, alpha=0.001, beta=0.1, gamma=0.1)


