#!/usr/bin/env python3
"""
Test delle ottimizzazioni implementate nel modulo quantum_utils

Questo script dimostra i miglioramenti di performance e memory management.
"""

import numpy as np
import time
from quantum_utils import (
    ProcessingConfig, SimulatorManager, get_memory_pool,
    calculate_gp_image, apply_filters_optimized, 
    get_base_quantum_circuit, adaptive_shots,
    performance_monitor, batch_process_images
)

def test_memory_pool():
    """Test memory pool efficiency"""
    print("🧪 Testing Memory Pool...")
    
    # Test without memory pool (standard approach)
    start_time = time.time()
    images_standard = []
    for i in range(50):
        img = np.random.rand(16, 16)  # New allocation each time
        images_standard.append(img)
    time_standard = time.time() - start_time
    
    # Test with memory pool
    pool = get_memory_pool(image_size=16, max_images=50)
    start_time = time.time()
    images_pooled = []
    for i in range(50):
        img_buffer = pool.get_image_buffer()
        img_buffer[:] = np.random.rand(16, 16)  # Reuse buffer
        images_pooled.append(img_buffer.copy())
    time_pooled = time.time() - start_time
    
    print(f"  Standard allocation: {time_standard:.4f}s")
    print(f"  Memory pool: {time_pooled:.4f}s") 
    print(f"  Speedup: {time_standard/time_pooled:.2f}x")

def test_gp_optimization():
    """Test GP calculation optimization"""
    print("\n🧪 Testing GP Calculation Optimization...")
    
    # Generate test data
    green_channel = np.random.rand(64, 64)
    red_channel = np.random.rand(64, 64)
    
    # Test standard calculation (simulated old version)
    start_time = time.time()
    for _ in range(100):
        # Old approach (simulated)
        gp_old = (green_channel - 0.5 * red_channel) / (green_channel + 0.5 * red_channel + 1e-10)
        gp_old = np.clip(gp_old, -1, 1)
        gp_old[green_channel + 0.5 * red_channel == 0] = 0
    time_old = time.time() - start_time
    
    # Test optimized calculation
    start_time = time.time()
    for _ in range(100):
        gp_new = calculate_gp_image(green_channel, red_channel, G=0.5)
    time_new = time.time() - start_time
    
    print(f"  Standard GP calculation: {time_old:.4f}s")
    print(f"  Optimized GP calculation: {time_new:.4f}s")
    print(f"  Speedup: {time_old/time_new:.2f}x")

def test_circuit_caching():
    """Test quantum circuit caching"""
    print("\n🧪 Testing Circuit Caching...")
    
    # Test without caching (simulated)
    from qiskit import QuantumCircuit
    start_time = time.time()
    circuits_no_cache = []
    for _ in range(20):
        qc = QuantumCircuit(10, 10)  # Create new circuit each time
        qc.h(range(8))
        qc.barrier()
        circuits_no_cache.append(qc)
    time_no_cache = time.time() - start_time
    
    # Test with caching
    start_time = time.time()
    circuits_cached = []
    for _ in range(20):
        qc = get_base_quantum_circuit(4, 1)  # Uses cache after first call
        circuits_cached.append(qc.copy())
    time_cached = time.time() - start_time
    
    print(f"  Without caching: {time_no_cache:.4f}s")
    print(f"  With caching: {time_cached:.4f}s")
    print(f"  Speedup: {time_no_cache/time_cached:.2f}x")

def test_adaptive_shots():
    """Test adaptive shot calculation"""
    print("\n🧪 Testing Adaptive Shot Calculation...")
    
    # Test different convergence scenarios
    test_cases = [
        ("Converged", [0.001, 0.0015, 0.001], 10),
        ("Still optimizing", [0.1, 0.08, 0.12], 20),
        ("Deep circuit", [0.05, 0.04, 0.06], 100)
    ]
    
    for case_name, history, depth in test_cases:
        shots = adaptive_shots(depth, history, base_shots=10000)
        print(f"  {case_name} (depth={depth}): {shots} shots")

@performance_monitor
def test_batch_processing():
    """Test batch processing efficiency"""
    print("\n🧪 Testing Batch Processing...")
    
    # Generate test images
    test_images = [np.random.rand(16, 16) for _ in range(20)]
    
    def simple_processor(img):
        return np.mean(img)
    
    # Sequential processing
    start_time = time.time()
    results_sequential = [simple_processor(img) for img in test_images]
    time_sequential = time.time() - start_time
    
    # Batch processing  
    start_time = time.time()
    results_batch = batch_process_images(test_images, simple_processor, batch_size=4)
    time_batch = time.time() - start_time
    
    print(f"  Sequential: {time_sequential:.4f}s")
    print(f"  Batch: {time_batch:.4f}s")

def test_simulator_manager():
    """Test singleton simulator manager"""
    print("\n🧪 Testing Simulator Manager...")
    
    # Test singleton behavior
    sim1 = SimulatorManager()
    sim2 = SimulatorManager()
    
    print(f"  Same instance: {sim1 is sim2}")
    print(f"  GPU available: {sim1.is_gpu_available()}")

if __name__ == "__main__":
    print("🚀 Testing Quantum Utils Optimizations")
    print("=" * 50)
    
    # Run all tests
    test_memory_pool()
    test_gp_optimization()
    test_circuit_caching()
    test_adaptive_shots()
    test_batch_processing()
    test_simulator_manager()
    
    print("\n" + "=" * 50)
    print("✅ All optimization tests completed!")
    print("\n📊 Summary of improvements:")
    print("  - Memory pool: Reduced allocation overhead")
    print("  - GP calculation: Vectorized operations")
    print("  - Circuit caching: Avoid repeated construction")
    print("  - Adaptive shots: Dynamic shot optimization")
    print("  - Batch processing: Improved throughput")
    print("  - Singleton simulator: Resource efficiency")