# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:57:32 2024

@author: Giuseppe
"""
from FQRI_lib import *
#QiskitRuntimeService.save_account(channel="ibm_quantum", token="a701629b4cb81190a4a3707b135362ca3dd24e69f34a6920f7973bd9376caace2c95491188c2699cfae1b0078ed6b30dcd0774238a2d40ad8dcc740d18930471", overwrite=True)
from qiskit_aer import AerSimulator
import numpy as np
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram


def main():
    image1_name = "Clipboard-1 32.tif"

    print("Caricamento immagine...")
    theta_values1, n1 = image_to_angles(image1_name)

    print("Impostazione del circuito quantistico...")
    qc = setup_quantum_circuit_single_image(n1, theta_values1)
    #qc = neutralize_color(qc)
    measure_quantum_circuit(qc)

    print("Simulazione del circuito quantistico...")
    simulator = AerSimulator()
    qc = transpile(qc, simulator)
    job = simulator.run(qc, shots=10**(n1 + 2))
    result = job.result()
    
    print("Recupero dei risultati...")
    experiment = qc.name if qc.name in result.results else result.results[0].header.name
    counts = result.get_counts(experiment)
    
    plot_histogram(counts, title='Quantum Image')

    print("Decodifica dei risultati...")
    image_quantum = decode_quantum_image(counts, n1)

    print("Visualizzazione delle immagini...")
    plot_images_single(image1_name, image_quantum)

    print("Salvataggio dell'immagine quantistica...")
    save_image_as_tiff(image_quantum, "quantum_image_output.tif")

if __name__ == "__main__":
    main()