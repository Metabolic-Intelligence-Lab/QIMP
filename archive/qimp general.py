from qiskit import QuantumCircuit, transpile
from math import pi
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import RYGate, RZGate, MCMT, HGate
from qiskit.quantum_info import Operator
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
import os

import imageio



QiskitRuntimeService.save_account(channel="ibm_quantum", token="a701629b4cb81190a4a3707b135362ca3dd24e69f34a6920f7973bd9376caace2c95491188c2699cfae1b0078ed6b30dcd0774238a2d40ad8dcc740d18930471", overwrite=True)

#ENCODING 
def extract_index(binary_key):
    return int(binary_key, 2)

def setup_quantum_circuit(n, theta_values):
    """
    Configura e restituisce un circuito quantistico per la codifica di un'immagine 
    2^n x 2^n utilizzando rotazioni RY controllate.
    
    La funzione inizia applicando la porta Hadamard a tutti i qubit di posizione per 
    creare una sovrapposizione uniforme di stati. Successivamente, genera e applica 
    rotazioni RY controllate basate su una sequenza lineare di angoli theta, che sono
    calcolati per dividere l'intervallo [0, pi] uniformemente in base al numero di stati
    possibili (2^(2n)). Ogni rotazione RY è controllata da tutti i qubit di posizione
    e mirata a un qubit ausiliario. Le porte X sono usate per preparare gli stati di controllo 
    necessari prima e dopo ogni rotazione.
    
    :param n: int, il numero di qubit per dimensione (con 2*n qubit totali per le coordinate 
             x e y più un qubit ausiliario).
    :return: QuantumCircuit, il circuito configurato.
    """
    n_qubits = 2 * n
    qc = QuantumCircuit(n_qubits + 1, n_qubits + 1)

    # Applicazione della porta Hadamard a tutti i qubit di posizione
    qc.h(range(n_qubits))

    # Valori di theta generati linearmente
   # increment = np.pi / (2 ** n_qubits)
  #  theta_values = [i * increment for i in range(2 ** n_qubits)]

    # Genera tutti gli indici binari possibili per n_qubits in little-endian
    indices = [format(i, '0' + str(n_qubits) + 'b')[::-1] for i in range(2 ** n_qubits)]

    # Applicazione delle rotazioni condizionate
    for index, theta in zip(indices, theta_values):
        # Applica la porta X dove il bit è 0 (preparazione dello stato di controllo)
        for bit_position, bit in enumerate(index):
            if bit == '0':
                qc.x(bit_position)

        # Applica la rotazione RY controllata
        ry_gate = RYGate(theta).control(n_qubits)
        qc.append(ry_gate, list(range(n_qubits)) + [n_qubits])

        # Reset dello stato di controllo
        for bit_position, bit in enumerate(index):
            if bit == '0':
                qc.x(bit_position)
        qc.barrier()
    return qc

def measure_quantum_circuit(qc):
    """
    Aggiunge le operazioni di misura a un circuito quantistico esistente.
 
    :param qc: QuantumCircuit, il circuito quantistico a cui aggiungere le misure.
    """
    n_qubits = qc.num_qubits
    qc.measure(range(n_qubits), range(n_qubits))
    
def apply_zz_gate(qc, qubit1, qubit2, theta):
    """
    Applica un gate ZZ simulato usando porte CNOT e RZ.
    :param qc: QuantumCircuit, il circuito su cui operare.
    :param qubit1: int, indice del primo qubit.
    :param qubit2: int, indice del secondo qubit.
    :param theta: float, parametro theta per la rotazione RZ.
    """
    qc.cx(qubit1, qubit2)
    qc.rz(theta, qubit2)
    qc.cx(qubit1, qubit2)
    
 #GEOMETRIC TRANSFORMATIONS   
    
def apply_x_on_coordinate(qc, n, coord):
    """
    Applica la porta X ai qubit che codificano per la coordinata specificata.
    
    :param qc: QuantumCircuit, il circuito quantistico a cui applicare la porta X.
    :param n: int, il numero di qubit per dimensione (n per x e n per y, in un'immagine 2^n x 2^n).
    :param coord: str, 'x' per applicare su coordinate x, 'y' per coordinate y.
    """
    if coord == 'x':
        # Applica porta X ai primi n qubit che codificano la coordinata x
        for qubit in range(n):
            qc.x(qubit)
    elif coord == 'y':
        # Applica porta X ai successivi n qubit che codificano la coordinata y
        for qubit in range(n, 2*n):
            qc.x(qubit)
    else:
        raise ValueError("Coordinate non riconosciute. Usare 'x' o 'y'.")

def apply_coordinate_swapping(qc, n):
    """
    Applica le porte SWAP per scambiare i qubit che codificano le coordinate x e y.
    
    :param qc: QuantumCircuit, il circuito quantistico su cui operare.
    :param n: int, il numero di qubit per dimensione in un'immagine 2^n x 2^n.
    """
    for i in range(n):
        x_qubit = i        # i-esimo qubit per la coordinata x
        y_qubit = n + i    # i-esimo qubit per la coordinata y
        qc.swap(x_qubit, y_qubit)

def rotate_90(qc, n):
    """
    Applica una rotazione di 90 gradi al circuito (R90 = CFx).
    :param qc: QuantumCircuit, il circuito quantistico.
    :param n: int, numero di qubit per dimensione.
    """
    apply_coordinate_swapping(qc, n)  # Coordinate swapping
    apply_x_on_coordinate(qc, n, 'x')  # Flipping su x

def rotate_180(qc, n):
    """
    Applica una rotazione di 180 gradi al circuito (R180 = FyFx).
    :param qc: QuantumCircuit, il circuito quantistico.
    :param n: int, numero di qubit per dimensione.
    """
    apply_x_on_coordinate(qc, n, 'y')  # Flipping su y
    apply_x_on_coordinate(qc, n, 'x')  # Flipping su x

def rotate_270(qc, n):
    """
    Applica una rotazione di 270 gradi al circuito (R270 = CFy).
    :param qc: QuantumCircuit, il circuito quantistico.
    :param n: int, numero di qubit per dimensione.
    """
    apply_coordinate_swapping(qc, n)  # Coordinate swapping
    apply_x_on_coordinate(qc, n, 'y')  # Flipping su y

#COLOR TRANSFORMATIONS

def apply_color_inversion(qc):
    """
    Applica un gate X all'ultimo qubit di un circuito quantistico per invertire il colore.
    
    :param qc: QuantumCircuit, il circuito quantistico su cui operare.
    """
    last_qubit = qc.num_qubits - 1  # Indice dell'ultimo qubit
    qc.x(last_qubit)  # Applica la porta X all'ultimo qubit

def apply_color_transformation(qc):
    """
    Applica una porta Pauli-Z all'ultimo qubit di un circuito quantistico per trasformare il colore.
    La porta Z modifica la fase dello stato |1⟩, aggiungendo un fattore di fase -1.
    
    :param qc: QuantumCircuit, il circuito quantistico su cui operare.
    """
    last_qubit = qc.num_qubits - 1  # Indice dell'ultimo qubit
    qc.z(last_qubit)  # Applica la porta Pauli-Z all'ultimo qubit

def neutralize_color(qc):
    """
    Applica una porta Hadamard all'ultimo qubit di un circuito quantistico per portare il qubit del colore 
    in una sovrapposizione di |0⟩ e |1⟩, neutralizzando l'effetto del colore.
    
    :param qc: QuantumCircuit, il circuito quantistico su cui operare.
    """
    last_qubit = qc.num_qubits - 1  # Indice dell'ultimo qubit
    qc.h(last_qubit)  # Applica la porta Hadamard all'ultimo qubit
  
def apply_custom_color_transformation(qc, theta):
    """
    Applica una trasformazione unitaria personalizzata C(2*theta) al qubit specificato,
    cambiando l'encoding del colore.

    :param qc: QuantumCircuit, il circuito quantistico su cui operare.
    :param theta: float, l'angolo theta per la trasformazione.
    :param qubit_index: int, indice del qubit al quale applicare la trasformazione.
    """
    # Definizione della matrice unitaria C(2*theta)
    cos_theta = np.cos(theta/2)
    sin_theta = np.sin(theta/2)
    unitary_matrix = np.array([[cos_theta, sin_theta], 
                               [sin_theta, -cos_theta]])
    last_qubit = qc.num_qubits - 1  # Indice dell'ultimo qubit
    
    # Creazione dell'operatore unitario e applicazione al circuito
    custom_gate = Operator(unitary_matrix)
    #qc.z(last_qubit) 
    qc.unitary(custom_gate, [last_qubit], label=f"C(2*{theta})")


"""
def initialize_state(n):

    Inizializza un immagine random.
    
    Args:
    n (int): Dimensione di ogni lato della griglia N x N.
    
    Returns:
    numpy.ndarray: Una griglia N x N di valori angolari continui tra 0 e π.
 
    N = 2**n  # Calcola la dimensione della griglia basata su n
   # state = np.random.uniform(0, pi, size=(N**2))  # Inizializza con valori continui
    p=0.6
    state = np.random.choice([0, np.pi], size=(N**2), p=[p, 1-p]) 
   
    return state
"""
def image_to_angles(image_name):
    """
    Carica un'immagine TIFF dalla directory di lavoro di Spyder e trasforma i valori dei pixel in angoli.

    Args:
    image_name (str): Nome dell'immagine TIFF.

    Returns:
    numpy.ndarray: Un array unidimensionale di valori angolari.
    """
    # Trova la directory di lavoro di Spyder
    working_dir = os.getcwd()
    # Combina il percorso della directory di lavoro con il nome dell'immagine
    image_path = os.path.join(working_dir, image_name)
    
    # Carica l'immagine
    image = imageio.imread(image_path)
    
    
    # Normalizza i valori dei pixel nell'intervallo [0, 1]
    image = image / 255.0
    
    # Calcola gli angoli corrispondenti ai valori dei pixel
    angles = np.arccos(1 - 2 * image)
    
    # Imposta gli angoli diversi da zero a pi
   # angles[angles != 0] = np.pi
    
    # Appiattisce la matrice bidimensionale in un array unidimensionale
    angles_flat = angles.flatten()
    
    return angles_flat


def initialize_state(n, pattern='random', density=0.5):
    """
    Inizializza un immagine con una struttura a rete.

    Args:
    n (int): Dimensione di ogni lato della griglia N x N.
    pattern (str): Il tipo di pattern della rete ('random', 'grid', 'circle', 'fractal').
    density (float): La densità della rete (valido solo per pattern='random').

    Returns:
    numpy.ndarray: Una griglia N x N di valori angolari basata sulla struttura della rete.
    """
    N = 2**n  # Calcola la dimensione della griglia basata su n
    
    if pattern == 'random':
        # Crea un grafo casuale
        G = nx.gnm_random_graph(N**2, int(density * N**2))
        pos = nx.spring_layout(G)  # Posizioni casuali dei nodi
        
    elif pattern == 'grid':
        # Crea un grafo con disposizione a griglia
        G = nx.grid_2d_graph(N, N)
        pos = dict((n, (n[0] / (N-1), n[1] / (N-1))) for n in G.nodes())
    
    elif pattern == 'circle':
        # Crea un grafo con disposizione circolare
        G = nx.cycle_graph(N**2)
        pos = nx.circular_layout(G)
    
    elif pattern == 'fractal':
        # Crea un grafo con una struttura frattale-like
        G = nx.random_geometric_graph(N**2, 0.3)
        pos = nx.spring_layout(G)
    
    # Calcola gli angoli utilizzando le posizioni dei nodi
    angles = np.zeros(N**2)
    for node, (x, y) in pos.items():
        i = int(x * (N-1))
        j = int(y * (N-1))
        angles[i*N + j] = np.arctan2(y - 0.5, x - 0.5)
    
    # Imposta gli angoli diversi da zero a pi
    angles[angles != 0] = np.pi
    
    return angles






def apply_amplitude_amplification(qc, marked_states):
    """ Applica l'amplificazione di ampiezza per gli stati specificati. """
    # Inversione di fase sugli stati marcati
    for state in marked_states:
        qc.x(state)  # Supponendo che 'state' sia l'indice del qubit da marcare
        qc.z(state)
        qc.x(state)

    # Operatore di diffusione
    for q in range(qc.num_qubits):
        qc.h(q)
    for q in range(qc.num_qubits):
        qc.x(q)
    L=1
    # Append MCMT per multi-controlled-X gate
    qc.append(MCMT('x', num_ctrl_qubits=L, num_target_qubits=qc.num_qubits - L), list(range(qc.num_qubits)))

    for q in range(qc.num_qubits):
        qc.x(q)
    for q in range(qc.num_qubits):
        qc.h(q)

# Creazione del circuito
    
#Generazione dei valori di theta
n = 6
n_qubits = 2*n
n_shots = 10**(n+2)
num_pixels = 2**(n_qubits)
"""
center_size_ratio = 0.25  # Configura la dimensione del quadrato centrale
noise_intensity = 0.2  # Configura l'intensità dello spot noise
# Esempio di utilizzo
patterns = ['random', 'grid', 'circle', 'fractal']
pattern = 'grid'
"""

# Generazione dei valori di theta
#theta_values = initialize_state(n, pattern, 0.5)






# Esempio di utilizzo
image_name = "Clipboard-1.tif"  # Sostituisci "nome_immagine.tif" con il nome effettivo della tua immagine TIFF
theta_values = image_to_angles(image_name)
image_values = np.array(theta_values) * 255 /pi



#♣theta_values = theta_out
#Generazione del Circuito Quantistico per FRQI

qc = setup_quantum_circuit(n , theta_values)
#rotate_270(qc,n)
#apply_coordinate_swapping(qc, n)
#apply_x_on_coordinate(qc, n, 'x')
#apply_x_on_coordinate(qc, n, 'y')
#apply_color_inversion(qc)
#apply_color_transformation(qc)
#neutralize_color(qc)
#theta0 = np.pi/2  # Ad esempio, theta = pi/4
#apply_custom_color_transformation(qc, theta0)  # Applica la trasformazione al qubit 0
marked_states=[n_qubits]
apply_coordinate_swapping(qc, n)
apply_amplitude_amplification(qc, marked_states)
apply_coordinate_swapping(qc, n)

measure_quantum_circuit(qc)

print(qc)


#Esecuzione del Circuito Quantistico sul Simulatore
simulator = AerSimulator()
qc = transpile(qc, simulator)
result = simulator.run(qc, shots = n_shots).result()
counts = result.get_counts(qc)
plot_histogram(counts, title='FRQI')
total_shots = sum(counts.values())
prob = {k: v / total_shots for k, v in counts.items()}
amp = {k: np.sqrt(v) for k, v in prob.items()}

# Costruzione dello stato psi' come stringa per la visualizzazione
psi_prime = " |ψ'⟩ = "
psi_prime += " + ".join([f"{amp_val:.4f}|{state}⟩" for state, amp_val in amp.items()])
print(psi_prime)



# Calcolo delle probabilità condizionali P(j|0) e P(j|1)
prob_cond_0 = {}
prob_cond_1 = {}
values_q = {}

for state, p in prob.items():
    # Il qubit del colore è il più a sinistra, quindi guardiamo il primo bit
    j = state[1:]  # Prendiamo i due bit più a destra come 'j' per la posizione
    if state[0] == '0':
        prob_cond_0[j] = p
    else:
        prob_cond_1[j] = p

for j in set(prob_cond_0.keys()) | set(prob_cond_1.keys()):
    p_0 = prob_cond_0.get(j,0)
    p_1 = prob_cond_1.get(j, 0)
    if p_0 + p_1 > 0:
        values_q[j] = np.arccos(np.sqrt(p_0 / (p_0 + p_1))) * 255 * (2 / np.pi)
    else:
        values_q[j] = 0  # Imposta theta a 0 se non ci sono probabilità

# Stampa i valori calcolati
"""
print("Probabilità condizionali P(j|0):", prob_cond_0)
print("Probabilità condizionali P(j|1):", prob_cond_1)
print("Valori  retrieved:", values_q)
print("valori originali:", image_values)
"""
num_digits = len(next(iter(values_q.keys())))
sorted_pixel_values = {k: v for k, v in sorted(values_q.items(), key=lambda item: extract_index(item[0]))}
image_quantum = np.zeros((int(math.sqrt(num_pixels)),int( math.sqrt(num_pixels))))

for index, value in sorted_pixel_values.items():
    row = int(index[:num_digits // 2], 2)
    col = int(index[num_digits // 2:], 2)
    image_quantum[row][col] = value


# Reshape dei valori dei pixel in una matrice 2x2 per l'immagine
image_classic = image_values.reshape(int(math.sqrt(num_pixels)),int( math.sqrt(num_pixels)))

# Crea una figura e definisci due subplot affiancati
plt.figure(figsize=(10, 5))  # Dimensione della figura che contiene i subplot

# Subplot per l'immagine classica
plt.subplot(1, 2, 1)  # (righe, colonne, indice del subplot)
plt.imshow(image_classic, cmap='gray', vmin=0, vmax=255)
plt.colorbar()
plt.title("Immagine Classica")

# Subplot per l'immagine quantistica ricostruita
plt.subplot(1, 2, 2)  # (righe, colonne, indice del subplot)
plt.imshow(image_quantum, cmap='gray', vmin=0, vmax=255)
plt.colorbar()
plt.title("Immagine Quantistica Ricostruita")

plt.show()



# Supponendo che 'image_quantum' sia la matrice NxN che hai ottenuto dalla misurazione quantistica
def image_to_theta(image_quantum):
    # Normalizzazione dei valori a una scala da 0 a pi
    theta_out = (image_quantum / 255) * np.pi
    
    # Trasformazione della matrice in un vettore
    theta_out_vector = theta_out.flatten()
    
    return theta_out_vector

# Uso della funzione
theta_out = image_to_theta(image_quantum)

# theta_out ora può essere usato come nuovo set di angoli theta per riapplicare al circuito

