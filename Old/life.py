# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:26:09 2024

@author: Giuseppe
"""



def initialize_game_state(n, p):
    """
    Inizializza uno stato di gioco per il Gioco della Vita.
    
    Args:
    n (int): Dimensione di ogni lato della griglia N x N.
    
    Returns:
    numpy.ndarray: Una griglia N x N di valori binari (0 o 1).
    """
    N = 2**n  # Calcola la dimensione della griglia basata su n
    
    state = np.random.choice([0, np.pi], size=(N**2), p=[p, 1-p])  # Inizializza con probabilità
    return state


def evolve_game_state(state, n):
    """
    Evolve the state of the game grid for one generation using Conway's Game of Life rules.
    
    Args:
        state (numpy.ndarray): Current state of the game, an array of theta values (0 or pi).
        n (int): The power used to determine the grid size (N x N, with N = 2**n).
    
    Returns:
        numpy.ndarray: The updated grid state after applying the Game of Life rules.
    """
    N = 2**n  # Dimensione della griglia
    new_state = np.zeros((N, N), dtype=state.dtype)  # Crea un nuovo stato con la stessa forma e tipo di dati
    state = state.reshape((N, N))  # Ridimensiona in 2D per facilitare il processo

    # Itera su ogni cella per determinare il suo nuovo stato
    for i in range(N):
        for j in range(N):
            # Calcola il numero di vicini vivi usando np.isclose per evitare problemi di precisione
            num_alive_neighbors = sum(
                np.isclose(state[(i + di) % N, (j + dj) % N], np.pi)
                for di in [-1, 0, 1] for dj in [-1, 0, 1]
                if (di, dj) != (0, 0)
            )

            # Applica le regole del Game of Life
            if np.isclose(state[i, j], np.pi):  # Cella attualmente viva
                if num_alive_neighbors in [2, 3]:
                    new_state[i, j] = np.pi  # Sopravvive
                else:
                    new_state[i, j] = 0  # Muore
            else:  # Cella attualmente morta
                if num_alive_neighbors == 3:
                    new_state[i, j] = np.pi  # Nasce

    return new_state.flatten()  # Appiattisce il nuovo stato prima di restituirlo

def calculate_grid(nsteps):
    nrows = int(math.sqrt(nsteps))
    ncols = nrows
    while nrows * ncols < nsteps:
        ncols += 1
    return nrows, ncols




nsteps = 30
nrows, ncols = calculate_grid(nsteps)

theta_values = initialize_game_state(n, p)

# Preparazione del plot
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)

for i in range(nsteps):
    row = i // ncols
    col = i % ncols
    theta_values = evolve_game_state(theta_values, n)
    image_values = np.array(theta_values).reshape((2**n, 2**n)) * 255 / np.pi
    
    
    
    
    
    
    
    ax = axes[row, col]
    ax.imshow(image_values, cmap='gray', interpolation='nearest')
    ax.set_title(f'Step {i+1}')
    ax.axis('off')









# Nascondi gli assi vuoti se nsteps non è un quadrato perfetto
for idx in range(i + 1, nrows * ncols):
    row = idx // ncols
    col = idx % ncols
    axes[row, col].axis('off')

plt.tight_layout()




"""
def initialize_game_state(n, center_size_ratio=0.25, noise_intensity=0.8):
    N = 2**n  # Calcola la dimensione della griglia basata su n
    state = np.zeros((N, N))  # Inizializza con valori zero

    # Determina le dimensioni del quadrato centrale
    center_size = int(N * center_size_ratio)
    start_idx = (N - center_size) // 2
    end_idx = start_idx + center_size

    # Imposta il quadrato centrale a π
    state[start_idx:end_idx, start_idx:end_idx] = np.pi

    # Genera spot noise per le regioni esterne
    for i in range(N):
        for j in range(N):
            if not (start_idx <= i < end_idx and start_idx <= j < end_idx):
                #state[i, j] = np.random.uniform(0, np.pi)

                state[i, j] = np.random.choice([0, np.pi], p=[1 - noise_intensity, noise_intensity])
            
    return state.flatten()
"""