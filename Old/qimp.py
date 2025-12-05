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
# Inizializzazione dei valori di theta per ogni pixel


theta_values = [0.5*pi , 0 * pi, 0 * pi, 0.25* pi]
#theta_values = [0, 0.5 * np.pi, 0.5 * np.pi, 0]

#+theta_values = [pi/2,pi/2,pi/2,pi/2]

image_values = np.array(theta_values) * 255 *2 /pi

# Reshape dei valori dei pixel in una matrice 2x2 per l'immagine
image_classic = image_values.reshape((2, 2))



# Creazione del circuito
qc = QuantumCircuit(3, 3)  # 3 qubit e 3 bit classici

# Applicazione delle porte Hadamard ai primi due qubit
qc.h([0, 1])

# Inserimento della logica per ogni pixel
for i, theta in enumerate(theta_values):
    if i == 0:  # Primo pixel
        qc.x([0, 1])  # Applica NOT ai primi due qubit
        qc.barrier()
    elif i == 1:  # Secondo pixel
        qc.x(0)  # Applica NOT solo al primo qubit
        qc.barrier()
    elif i == 2:  # Terzo pixel
        qc.x([0, 1])  # Applica NOT ai primi due qubit
        qc.barrier()
    elif i == 3:  # Quarto pixel
        qc.x(0)  # Applica NOT solo al primo qubit
        qc.barrier()

    # Applicazione della rotazione CRy basata su theta al terzo qubit
    
    qc.cry(theta, 0, 2)
    qc.cx(0, 1)
    qc.cry(-theta, 1, 2)
    
    qc.cx(0, 1)
    qc.cry(theta, 1, 2)







# Misurazione finale (opzionale se vuoi misurare l'intensità effettiva)
qc.measure([0, 1, 2], [0, 1, 2])

# Visualizzazione del circuito
print(qc.draw())

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

# Creazione dell'immagine classica ricostruita utilizzando i valori di theta
image_quantum = np.array([
    [values_q['00'], values_q['01']],
    [values_q['10'], values_q['11']]
])




# Crea una figura e definisci due subplot affiancati
plt.figure(figsize=(10, 5))  # Dimensione della figura che contiene i subplot

# Subplot per l'immagine classica
plt.subplot(1, 2, 1)  # (righe, colonne, indice del subplot)
plt.imshow(image_classic, cmap='gray', vmin=0, vmax=255)
plt.colorbar()
plt.title("Immagine Classica 2x2")

# Subplot per l'immagine quantistica ricostruita
plt.subplot(1, 2, 2)  # (righe, colonne, indice del subplot)
plt.imshow(image_quantum, cmap='gray', vmin=0, vmax=255)
plt.colorbar()
plt.title("Immagine Quantistica Ricostruita")

plt.show()





