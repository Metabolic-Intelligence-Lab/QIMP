# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:34:07 2025

@author: miste
"""

import torch

# Verifica se CUDA è disponibile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizzo del dispositivo: {device}")

# Crea due tensori casuali di grandi dimensioni
size = (10000, 10000)
tensor_a = torch.rand(size, device=device)
tensor_b = torch.rand(size, device=device)

# Esegue un'operazione sulla GPU
print("Esecuzione di un'operazione tensoriale sulla GPU...")
result = torch.matmul(tensor_a, tensor_b)

# Mostra informazioni sul risultato
print(f"Dimensioni del risultato: {result.size()}")
print(f"Alcuni elementi del risultato: {result[0, :5]}")

# Informazioni sulla memoria della GPU
if torch.cuda.is_available():
    print(f"Memoria GPU utilizzata: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
    print(f"Memoria GPU riservata: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
