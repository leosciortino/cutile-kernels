import argparse
import cupy as cp
import numpy as np
import cuda.tile as ct

ConstInt = ct.Constant[int]

@ct.kernel
def stencil_1d_kernel(left_arr, center_arr, right_arr, output_arr, tile_size: ConstInt):
    # 1. Get Tile Index
    # pid corresponds to the index in "Tile Space"
    # If pid=0, we want the 0th tile from all input arrays
    pid = ct.bid(0)
    
    # 2. Load Tiles
    # We load from 3 different array views. 
    # Because the views are shifted in memory (by python slicing), 
    # loading index=(pid,) from 'left_arr' automatically gets us the [i-1] data.
    
    # Shape of all loads is (TILE_SIZE,)
    left   = ct.load(left_arr,   index=(pid,), shape=(tile_size,))
    center = ct.load(center_arr, index=(pid,), shape=(tile_size,))
    right  = ct.load(right_arr,  index=(pid,), shape=(tile_size,))

    # 3. Compute Stencil
    # Formula: 0.25 * (left + 2*center + right)
    # This happens entirely in registers
    result = (left + (center * 2) + right) * 0.25

    # 4. Store Result
    ct.store(output_arr, index=(pid,), tile=result)


def cutile_stencil1d(input_arr: cp.ndarray, tile_size: int = 128) -> cp.ndarray:
    """
    Performs a 1D stencil operation (0.25 * (left + 2*center + right)) using cuTile.
    
    Args:
        input_arr (cp.ndarray): The input array (includes halo). 
                                Shape should be (N + 2,), where N is the valid domain.
        tile_size (int): The tile size for the kernel.
        
    Returns:
        cp.ndarray: The result array of shape (N,).
    """
    # Create Shifted Views
    # Left View:   Starts at index 0. (Contains elements -1 relative to center)
    # Center View: Starts at index 1. (The valid domain)
    # Right View:  Starts at index 2. (Contains elements +1 relative to center)
    left_view   = input_arr[0 : -2]
    center_view = input_arr[1 : -1]
    right_view  = input_arr[2 : ]
    
    vector_size = center_view.shape[0]
    output_arr = cp.zeros(vector_size, dtype=input_arr.dtype)
    
    grid_dim = ct.cdiv(vector_size, tile_size)
    grid = (grid_dim, 1, 1)

    ct.launch(cp.cuda.get_current_stream(), grid, stencil_1d_kernel, 
              (left_view, center_view, right_view, output_arr, tile_size))
    
    return output_arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the results",
        default=True,
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark",
        default=False,
    )
    args = parser.parse_args()

    print("--- Running cuTile 1D Stencil Example ---")

    # Configuration
    TILE_SIZE = 128
    VECTOR_SIZE = 1024 * 1024 * 64  # 64M elements 

    # 1. Allocate Data
    # Total size includes padding for the halo (1 element on each side)
    # [ P | ... Valid Data ... | P ]
    total_size = VECTOR_SIZE + 2
    
    host_data = np.random.rand(total_size).astype(np.float32)
    dev_data = cp.asarray(host_data)

    print(f"Vector Size: {VECTOR_SIZE}, Tile Size: {TILE_SIZE}")
    print(f"Input shape (with halo): {dev_data.shape}, dtype: {dev_data.dtype}")

    # Run cuTile Stencil
    dev_out = cutile_stencil1d(dev_data, tile_size=TILE_SIZE)
    print(f"cuTile Output shape: {dev_out.shape}, dtype: {dev_out.dtype}")

    if args.correctness_check:
        print("\n--- Checking Correctness ---")
        check_size = 1024
        
        # CPU Reference
        h_left   = host_data[0:check_size]
        h_center = host_data[1:check_size+1]
        h_right  = host_data[2:check_size+2]
        
        expected = (h_left + 2*h_center + h_right) * 0.25
        
        # GPU Result
        result_gpu = cp.asnumpy(dev_out[:check_size])

        if np.allclose(result_gpu, expected, atol=1e-5):
            print("Correctness check passed")
        else:
            print("Correctness check failed")
    else:
        print("Correctness check disabled")

    if args.benchmark:
        print("\n--- Running Benchmark ---")
        ITERATIONS = 20
        
        # Views are recreated here just to simulate the full call or we can call the wrapper
        # Calling wrapper includes allocation overhead, which might not be desired for pure kernel benchmark,
        # but for end-to-end it is fair. For pure kernel, we'd loop inside.
        # Let's verify bandwidth of the *kernel* as per original script logic.
        
        left_view   = dev_data[0 : -2]
        center_view = dev_data[1 : -1]
        right_view  = dev_data[2 : ]
        dev_out_bench = cp.zeros(VECTOR_SIZE, dtype=np.float32)
        grid_dim = ct.cdiv(VECTOR_SIZE, TILE_SIZE)
        grid = (grid_dim, 1, 1)

        print("Warming up...")
        ct.launch(cp.cuda.get_current_stream(), grid, stencil_1d_kernel, 
                (left_view, center_view, right_view, dev_out_bench, TILE_SIZE))
        cp.cuda.Device().synchronize()

        print(f"Executing {ITERATIONS} iterations...")
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        start_event.record()
        for _ in range(ITERATIONS):
            ct.launch(cp.cuda.get_current_stream(), grid, stencil_1d_kernel, 
                    (left_view, center_view, right_view, dev_out_bench, TILE_SIZE))
        end_event.record()
        end_event.synchronize()

        total_time_ms = cp.cuda.get_elapsed_time(start_event, end_event)
        avg_time_ms = total_time_ms / ITERATIONS
        
        total_bytes = VECTOR_SIZE * 4 * 4  # 3 Reads + 1 Write * 4 bytes
        throughput = (total_bytes / 1e9) / (avg_time_ms / 1000)

        print(f"Average Time: {avg_time_ms:.4f} ms")
        print(f"Throughput:   {throughput:.2f} GB/s")

    print("\n--- cuTile 1D Stencil example complete ---")