#!/usr/bin/env python3
"""
Quantum Image Processing Utilities - Optimized Shared Module

This module contains optimized, shared functions for quantum image processing
to eliminate code duplication and improve performance.

Author: Giuseppe (Optimized by Claude Code)
Created: 2025-09-10
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter
from functools import lru_cache
from typing import Tuple, List, Optional, Union
import logging
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Centralized configuration for quantum processing"""
    n_spatial_qubits: int = 4
    m_image_qubits: int = 1
    base_shots: int = 10000
    optimization_method: str = 'COBYLA'
    max_iterations: int = 50
    convergence_threshold: float = 1e-6
    gpu_enabled: bool = True
    memory_pool_size: int = 100
    
    def __post_init__(self):
        """Validate configuration"""
        if self.n_spatial_qubits < 1:
            raise ValueError("n_spatial_qubits must be >= 1")
        if self.base_shots < 1000:
            logger.warning(f"Low shot count ({self.base_shots}) may affect accuracy")

class MemoryPool:
    """Optimized memory pool for image arrays to reduce allocations"""
    
    def __init__(self, max_images: int, image_size: int):
        self.max_images = max_images
        self.image_size = image_size
        self.image_pool = np.zeros((max_images, image_size, image_size), dtype=np.float32)
        self.angle_pool = np.zeros((max_images, image_size * image_size), dtype=np.float32)
        self.current_idx = 0
        logger.info(f"Memory pool initialized: {max_images} images of {image_size}x{image_size}")
    
    def get_image_buffer(self) -> np.ndarray:
        """Get a reusable image buffer"""
        buffer = self.image_pool[self.current_idx % self.max_images]
        buffer.fill(0)  # Clear previous data
        return buffer
    
    def get_angle_buffer(self) -> np.ndarray:
        """Get a reusable angle buffer"""
        buffer = self.angle_pool[self.current_idx % self.max_images]
        buffer.fill(0)  # Clear previous data
        self.current_idx += 1
        return buffer

class SimulatorManager:
    """Singleton simulator manager with GPU/CPU optimization"""
    
    _instance = None
    _gpu_available = None
    _simulator = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._gpu_available = cls._detect_gpu()
            cls._simulator = cls._create_simulator()
        return cls._instance
    
    @classmethod
    def _detect_gpu(cls) -> bool:
        """Detect if GPU is available for quantum simulation"""
        try:
            test_simulator = AerSimulator(device='GPU')
            # Test with a simple circuit
            test_qc = QuantumCircuit(2)
            test_qc.h(0)
            test_qc.measure_all()
            test_simulator.run(test_qc, shots=1).result()
            logger.info("✅ GPU simulator available")
            return True
        except Exception as e:
            logger.info(f"⚠️ GPU not available, using CPU: {e}")
            return False
    
    @classmethod
    def _create_simulator(cls):
        """Create optimized simulator instance"""
        device = 'GPU' if cls._gpu_available else 'CPU'
        return AerSimulator(device=device)
    
    def get_simulator(self):
        """Get the singleton simulator instance"""
        return self._simulator
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available"""
        return self._gpu_available

# Global memory pool instance
_memory_pool = None

def get_memory_pool(image_size: int = 16, max_images: int = 100) -> MemoryPool:
    """Get or create global memory pool"""
    global _memory_pool
    if _memory_pool is None or _memory_pool.image_size != image_size:
        _memory_pool = MemoryPool(max_images, image_size)
    return _memory_pool

def calculate_gp_image(green_channel: np.ndarray, 
                      red_channel: np.ndarray, 
                      G: float = 0.5,
                      output_format: str = 'normalized') -> np.ndarray:
    """
    Optimized GP (Green-Purple) image calculation with multiple output formats.
    
    Args:
        green_channel: Green channel intensities
        red_channel: Red channel intensities  
        G: GP calculation parameter (default 0.5)
        output_format: 'normalized' (-1 to 1), '16bit' (0 to 4096), 'uint8' (0 to 255)
    
    Returns:
        GP calculated image in specified format
    """
    # Vectorized calculation with numerical stability
    denominator = green_channel + G * red_channel + 1e-10
    gp_array = (green_channel - G * red_channel) / denominator
    
    # Handle edge cases efficiently
    gp_array = np.clip(gp_array, -1, 1)
    zero_mask = (green_channel + G * red_channel) == 0
    gp_array[zero_mask] = 0
    
    # Convert to requested format
    if output_format == 'normalized':
        return gp_array
    elif output_format == '16bit':
        return np.interp(gp_array, [-1, 1], [0, 4096]).astype(np.uint16)
    elif output_format == 'uint8':
        return np.interp(gp_array, [-1, 1], [0, 255]).astype(np.uint8)
    else:
        raise ValueError(f"Unknown output_format: {output_format}")

def apply_filters_optimized(channel_array: np.ndarray, 
                          sigma: float = 1.0, 
                          median_size: int = 3,
                          output_buffer: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Optimized filter application with optional output buffer for memory efficiency.
    
    Args:
        channel_array: Input image channel
        sigma: Gaussian filter sigma
        median_size: Median filter kernel size
        output_buffer: Optional pre-allocated output buffer
        
    Returns:
        Filtered image array
    """
    if output_buffer is not None:
        # Use provided buffer to avoid allocation
        gaussian_filter(channel_array, sigma=sigma, output=output_buffer)
        median_filter(output_buffer, size=median_size, output=output_buffer)
        return output_buffer
    else:
        # Standard processing
        filtered_array = gaussian_filter(channel_array, sigma=sigma)
        filtered_array = median_filter(filtered_array, size=median_size)
        return filtered_array

def visualize_channels_and_gp(green_channel: np.ndarray, 
                             red_channel: np.ndarray, 
                             gp_image: np.ndarray,
                             title: str = "Channel Analysis") -> None:
    """
    Optimized visualization of channels and GP image.
    
    Args:
        green_channel: Green channel data
        red_channel: Red channel data  
        gp_image: GP calculated image
        title: Plot title
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Use efficient colormap rendering
    axs[0].imshow(green_channel, cmap='Greens', aspect='auto')
    axs[0].set_title('Green Channel')
    axs[0].axis('off')
    
    axs[1].imshow(red_channel, cmap='Reds', aspect='auto')
    axs[1].set_title('Red Channel')
    axs[1].axis('off')
    
    axs[2].imshow(gp_image, cmap='gray', aspect='auto')
    axs[2].set_title('GP Image')
    axs[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

@lru_cache(maxsize=32)
def get_base_quantum_circuit(n_spatial: int, m_image: int) -> QuantumCircuit:
    """
    Cached quantum circuit template creation to avoid repeated construction.
    
    Args:
        n_spatial: Number of spatial qubits per dimension
        m_image: Number of image selection qubits
        
    Returns:
        Base quantum circuit template
    """
    n_qubits = 2 * n_spatial + m_image + 1
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Initialize spatial qubits in superposition
    qc.h(range(2 * n_spatial))
    qc.barrier()
    
    logger.debug(f"Created base circuit: {n_qubits} qubits")
    return qc

def adaptive_shots(circuit_depth: int, 
                  convergence_history: List[float], 
                  base_shots: int = 10000) -> int:
    """
    Calculate adaptive shot count based on circuit complexity and convergence.
    
    Args:
        circuit_depth: Depth of quantum circuit
        convergence_history: Recent convergence values
        base_shots: Base number of shots
        
    Returns:
        Optimized shot count
    """
    # Check convergence stability
    if len(convergence_history) >= 3:
        recent_std = np.std(convergence_history[-3:])
        if recent_std < 0.01:  # Converged
            return max(base_shots // 2, 1000)
    
    # Scale with circuit depth but cap maximum
    scaling_factor = min(2.0, 1.0 + circuit_depth / 50.0)
    optimized_shots = int(base_shots * scaling_factor)
    
    return min(optimized_shots, 100000)  # Cap at 100k shots

def performance_monitor(func):
    """Decorator to monitor function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.3f}s")
        return result
    return wrapper

@performance_monitor
def batch_process_images(image_list: List[np.ndarray], 
                        processor_func,
                        batch_size: int = 4) -> List:
    """
    Batch process multiple images for improved efficiency.
    
    Args:
        image_list: List of images to process
        processor_func: Processing function to apply
        batch_size: Number of images to process simultaneously
        
    Returns:
        List of processed results
    """
    results = []
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i + batch_size]
        batch_results = [processor_func(img) for img in batch]
        results.extend(batch_results)
        logger.info(f"Processed batch {i//batch_size + 1}/{(len(image_list)-1)//batch_size + 1}")
    
    return results

def load_and_preprocess_image(image_path: str, 
                             target_size: Tuple[int, int] = (16, 16),
                             apply_filters: bool = True,
                             sigma: float = 1.0,
                             median_size: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized image loading and preprocessing pipeline.
    
    Args:
        image_path: Path to RGB image file
        target_size: Target image size (height, width)
        apply_filters: Whether to apply Gaussian and median filters
        sigma: Gaussian filter parameter
        median_size: Median filter kernel size
        
    Returns:
        Tuple of (red_channel, green_channel) as normalized float arrays
    """
    try:
        # Load image efficiently
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize using high-quality resampling
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        image_array = np.array(image, dtype=np.float32)
        
        # Extract channels
        red_channel = image_array[:, :, 0] / 255.0  # Normalize to [0,1]
        green_channel = image_array[:, :, 1] / 255.0
        
        # Apply filters if requested
        if apply_filters:
            red_channel = apply_filters_optimized(red_channel, sigma, median_size)
            green_channel = apply_filters_optimized(green_channel, sigma, median_size)
        
        logger.debug(f"Loaded and preprocessed: {image_path}")
        return red_channel, green_channel
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        raise

def clear_memory_cache():
    """Clear function caches and memory pools"""
    get_base_quantum_circuit.cache_clear()
    global _memory_pool
    _memory_pool = None
    logger.info("Memory cache cleared")

# Configuration validation
def validate_config(config: ProcessingConfig) -> ProcessingConfig:
    """Validate and possibly adjust configuration parameters"""
    if config.base_shots < 1000:
        logger.warning(f"Increasing base_shots from {config.base_shots} to 1000")
        config.base_shots = 1000
    
    if config.n_spatial_qubits > 6:
        logger.warning(f"Large n_spatial_qubits ({config.n_spatial_qubits}) may cause memory issues")
    
    return config

# Export main functions and classes
__all__ = [
    'ProcessingConfig',
    'MemoryPool', 
    'SimulatorManager',
    'calculate_gp_image',
    'apply_filters_optimized',
    'visualize_channels_and_gp',
    'get_base_quantum_circuit',
    'adaptive_shots',
    'batch_process_images',
    'load_and_preprocess_image',
    'performance_monitor',
    'get_memory_pool',
    'clear_memory_cache',
    'validate_config'
]

if __name__ == "__main__":
    # Test the optimization utilities
    print("🧪 Testing Quantum Utils Optimization Module")
    
    # Test configuration
    config = ProcessingConfig(n_spatial_qubits=4, base_shots=5000)
    print(f"✅ Config created: {config}")
    
    # Test memory pool
    pool = get_memory_pool(image_size=16, max_images=10)
    test_buffer = pool.get_image_buffer()
    print(f"✅ Memory pool working: {test_buffer.shape}")
    
    # Test simulator manager  
    sim_manager = SimulatorManager()
    print(f"✅ Simulator manager: GPU={sim_manager.is_gpu_available()}")
    
    # Test GP calculation
    test_green = np.random.rand(8, 8)
    test_red = np.random.rand(8, 8)
    gp_result = calculate_gp_image(test_green, test_red)
    print(f"✅ GP calculation: {gp_result.shape}, range=[{gp_result.min():.3f}, {gp_result.max():.3f}]")
    
    # Test circuit caching
    base_circuit = get_base_quantum_circuit(4, 1)
    cached_circuit = get_base_quantum_circuit(4, 1)  # Should be from cache
    print(f"✅ Circuit caching: {base_circuit.num_qubits} qubits")
    
    print("🚀 All optimization tests passed!")