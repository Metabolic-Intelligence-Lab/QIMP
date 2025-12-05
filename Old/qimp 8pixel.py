# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:30:38 2024

@author: Giuseppe
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:16:53 2024

@author: Giuseppe
"""
from qiskit import QuantumCircuit, transpile
from math import pi
from qiskit_ibm_runtime import QiskitRuntimeService, Options, SamplerV2 as Sampler
from qiskit_aer import AerSimulator
from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Session, SamplerV2 as Sampler
from qiskit.visualization import plot_histogram, plot_state_city

import numpy as np
import matplotlib.pyplot as plt


QiskitRuntimeService.save_account(channel="ibm_quantum", token="a701629b4cb81190a4a3707b135362ca3dd24e69f34a6920f7973bd9376caace2c95491188c2699cfae1b0078ed6b30dcd0774238a2d40ad8dcc740d18930471", overwrite=True)

from qiskit import QuantumCircuit
import numpy as np

from qiskit import QuantumCircuit
import numpy as np

# Definizione dei valori di theta per ogni pixel

# Definizione del numero di valori di theta desiderati

num_pixels = 64
# Calcolo degli incrementi per generare i valori di theta
increment = np.pi / (num_pixels *2 )

# Generazione dei valori di theta
theta_values = [i * increment for i in range(num_pixels)]

# Stampa dei valori di theta generati
print("Theta values:")
print(theta_values)


image_values = np.array(theta_values) * 255 *2 /pi


# Funzione per creare il circuito quantistico FRQI per l'immagine
def create_FRQI_circuit(theta_values):
    num_qubits = 7  # 6 per l'indirizzamento, 1 per la codifica dell'intensità
    qc = QuantumCircuit(num_qubits)  # Non sono necessari bit classici per ora
    # Applicazione delle porte Hadamard ai primi 6 qubit qubit
    qc.h([0, 1,2,3,4,5])
    for i, theta in enumerate(theta_values):
        # Calcolo degli indici dei qubit per l'indirizzamento basato su i
        address = format(i, '06b')  # Converti i in una stringa binaria di 6 bit
        
        # Preparazione dello stato di indirizzamento
        for j, bit in enumerate(address):
            if bit == '1':
                qc.x(j)
        
        # Codifica dell'intensità del pixel utilizzando una rotazione controllata
        qc.mcry(theta, [0, 1, 2, 3, 4, 5], 6, None)  # Usiamo mcry per un controllo multi-qubit
        
        # Reset dei qubit di indirizzamento
        for j, bit in enumerate(address):
            if bit == '1':
                qc.x(j)
    
    return qc

# Creazione del circuito FRQI e visualizzazione
qc = create_FRQI_circuit(theta_values)
print(qc.draw())























"""



# Creazione del circuito quantistico
qc = QuantumCircuit(7, 7)  # 7 qubit e 7 bit classici

# Applicazione delle porte Hadamard ai primi 6 qubit qubit
qc.h([0, 1,2,3,4,5])




# Ciclo per l'applicazione della logica per ogni pixel
for i, theta in enumerate(theta_values):
    
    # Applicazione della NOT a tutti i primi 6 qubit
    qc.x(range(6))
    
    # Applicazione della NOT solo al primo qubit per i pixel pari
    if i % 2 == 0:
        qc.x(0)
    
    # Applicazione della rotazione CRy basata su theta al settimo qubit
    qc.cry(theta, 6, range(6))
    for j in range(5):
        qc.cx(j, j+1)
        qc.cry(-theta, j+1, 6)
        qc.cx(j, j+1)
        qc.cry(theta, j+1, 6)

# Misurazione finale
qc.measure(range(7), range(7))

# Visualizzazione del circuito
print(qc.draw())


"""





# Transpile for simulator
simulator = AerSimulator()
qc = transpile(qc, simulator)

# Run and get counts
result = simulator.run(qc, shots = 16192).result()
counts = result.get_counts(qc)
plot_histogram(counts, title='FRQI')



# Calcola le probabilità (opzionale, a seconda del tuo approccio)
total_shots = sum(counts.values())
prob = {k: v / total_shots for k, v in counts.items()}



# Calcolo delle ampiezze a partire dalle probabilità
amp = {k: np.sqrt(v) for k, v in prob.items()}

# Costruzione dello stato psi' come stringa per la visualizzazione
psi_prime = " |ψ'⟩ = "
psi_prime += " + ".join([f"{amp_val:.4f}|{state}⟩" for state, amp_val in amp.items()])

print(psi_prime)



# Calcolo delle probabilità condizionali P(j|0) e P(j|1)
prob_cond_0 = {}
prob_cond_1 = {}

for state, p in prob.items():
    # Il qubit del colore è il più a sinistra, quindi guardiamo il primo bit
    j = state[1:]  # Prendiamo i due bit più a destra come 'j' per la posizione
    if state[0] == '0':
        prob_cond_0[j] = p
    else:
        prob_cond_1[j] = p

# Calcola theta utilizzando le probabilità condizionali
values_q = {}

for j in set(prob_cond_0.keys()) | set(prob_cond_1.keys()):
    p_0 = prob_cond_0.get(j,0)
    p_1 = prob_cond_1.get(j, 0)
    if p_0 + p_1 > 0:
        values_q[j] = np.arccos(np.sqrt(p_0 / (p_0 + p_1))) * 255 * (2 / np.pi)
    else:
        values_q[j] = 0  # Imposta theta a 0 se non ci sono probabilità












# Stampa i valori calcolati
print("Probabilità condizionali P(j|0):", prob_cond_0)
print("Probabilità condizionali P(j|1):", prob_cond_1)
print("Valori  retrieved:", values_q)
print("valori originali:", image_values)

# Inizializzazione dell'immagine quantistica
size = int(np.sqrt(len(values_q)))
image_quantum = np.zeros((size, size))

# Ciclo su tutti gli elementi di values_q
for key, value in values_q.items():
    # Converti la chiave in coordinate decimali
    decimal_index = int(key, 2)
    # Calcola le coordinate i e j a partire dall'indice decimale
    i, j = divmod(decimal_index, size)
    # Assegna il valore di fase quantistica corrispondente alla posizione corretta nella matrice dell'immagine
    image_quantum[i, j] = value
        
        
        

# Creazione dell'immagine classica
image_values = np.array(theta_values) * 255 * 2 / np.pi
size_classic = int(np.sqrt(len(image_values)))
image_classic = image_values.reshape((size_classic, size_classic))

# Plot dell'immagine classica e dell'immagine quantistica ricostruita
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