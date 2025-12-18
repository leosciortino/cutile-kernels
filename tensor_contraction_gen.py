import sys
import subprocess
import os

class CuTileSolutionGenerator:
    def __init__(self, equation, dim_sizes):
        self.equation = equation.replace(" ", "")
        self.lhs, self.rhs = self.equation.split("->")
        self.input_a_str, self.input_b_str = self.lhs.split(",")
        self.output_str = self.rhs
        self.dim_sizes = dim_sizes
        self.filename = f"{os.path.splitext("_generated_kernel_tc.py")[0]}_{self.equation.replace(',', '_').replace('->', '_to_')}.py"

    def _get_shape_tuple(self, indices_str):
        return tuple(self.dim_sizes[c] for c in indices_str)

    def generate_and_run(self):
        # 1. Parse Indices
        idxs_a = list(self.input_a_str)
        idxs_b = list(self.input_b_str)
        idxs_out = list(self.output_str)

        # 2. Identify Roles
        contracted = [x for x in idxs_a if x in idxs_b and x not in idxs_out]
        free_a = [x for x in idxs_a if x not in contracted] 
        free_b = [x for x in idxs_b if x not in contracted]

        # 3. Calculate Permutations
        target_a = free_a + contracted
        perm_a = [idxs_a.index(x) for x in target_a]
        
        target_b = contracted + free_b
        perm_b = [idxs_b.index(x) for x in target_b]

        # 4. Calculate Flattened Dimensions
        def prod_dims(indices):
            return " * ".join([str(self.dim_sizes[x]) for x in indices]) if indices else "1"

        M_val = prod_dims(free_a)
        K_val = prod_dims(contracted)
        N_val = prod_dims(free_b)

        # 5. Build the Output Code
        code = []
        code.append("import torch")
        code.append("try:")
        code.append("    import cuda.tile as ct")
        code.append("except ImportError:")
        code.append("    print('Error: cuda.tile module not found. Is cuTile installed?')")
        code.append("    import sys; sys.exit(1)")
            
        code.append("")
        code.append("# --- Configuration ---")
        code.append(f"EQUATION = '{self.equation}'")
        for k, v in self.dim_sizes.items():
            code.append(f"DIM_{k.upper()} = {v}")
        code.append("")
        
        # Kernel Definition
        code.append("@ct.kernel")
        code.append(f"def generated_contraction(A_ptr, B_ptr, C_ptr):")
        code.append(f"    # 1. Load Data")
        
        # Helper to generate zero-index tuples (e.g., (0, 0, 0))
        def zero_idx(rank):
            return "(" + ", ".join(["0"] * rank) + ("," if rank == 1 else "") + ")"

        shape_a = ", ".join([f"DIM_{x.upper()}" for x in idxs_a])
        shape_b = ", ".join([f"DIM_{x.upper()}" for x in idxs_b])
        
        # FIXED: Added 'index' argument
        code.append(f"    tile_a = ct.load(A_ptr, index={zero_idx(len(idxs_a))}, shape=({shape_a}))")
        code.append(f"    tile_b = ct.load(B_ptr, index={zero_idx(len(idxs_b))}, shape=({shape_b}))")
        code.append("")
        
        # A Processing
        code.append(f"    # 2. Prepare A: Permute {idxs_a} -> {target_a}")
        if perm_a != list(range(len(idxs_a))):
            code.append(f"    tile_a = ct.permute(tile_a, axes={tuple(perm_a)})")
        code.append(f"    # Flatten to (M, K) = ({M_val}, {K_val})")
        code.append(f"    tile_a_2d = ct.reshape(tile_a, shape=({M_val}, {K_val}))")
        code.append("")
        
        # B Processing
        code.append(f"    # 3. Prepare B: Permute {idxs_b} -> {target_b}")
        if perm_b != list(range(len(idxs_b))):
            code.append(f"    tile_b = ct.permute(tile_b, axes={tuple(perm_b)})")
        code.append(f"    # Flatten to (K, N) = ({K_val}, {N_val})")
        code.append(f"    tile_b_2d = ct.reshape(tile_b, shape=({K_val}, {N_val}))")
        code.append("")
        
        # Matmul
        code.append(f"    # 4. Contract")
        code.append(f"    tile_c_2d = ct.matmul(tile_a_2d, tile_b_2d)")
        code.append("")
        
        # Output Processing
        current_out = free_a + free_b
        reshaped_dims = ", ".join([f"DIM_{x.upper()}" for x in current_out])
        
        code.append(f"    # 5. Reshape Output: (M, N) -> {current_out}")
        code.append(f"    tile_c = ct.reshape(tile_c_2d, shape=({reshaped_dims}))")
        
        if current_out != idxs_out:
            perm_out = [current_out.index(x) for x in idxs_out]
            code.append(f"    # Permute Output to requested: {idxs_out}")
            code.append(f"    tile_c = ct.permute(tile_c, axes={tuple(perm_out)})")
            
        # FIXED: Added 'index' argument to store
        code.append(f"    ct.store(C_ptr, tile=tile_c, index={zero_idx(len(idxs_out))})")
        code.append("")
        
        # Main Verification Block
        code.append("def main():")
        code.append("    print(f'Verifying Equation: {EQUATION}')")
        code.append("    if not torch.cuda.is_available():")
        code.append("        print('Error: CUDA not available for PyTorch.')")
        code.append("        return")
        code.append("")
        code.append("    # Initialize Tensors on GPU")
        code.append(f"    a = torch.randn({self._get_shape_tuple(idxs_a)}, device='cuda', dtype=torch.float16)")
        code.append(f"    b = torch.randn({self._get_shape_tuple(idxs_b)}, device='cuda', dtype=torch.float16)")
        code.append(f"    c_cutile = torch.zeros({self._get_shape_tuple(idxs_out)}, device='cuda', dtype=torch.float16)")
        code.append("")
        code.append("    # --- 1. PyTorch Ground Truth ---")
        code.append("    c_ref = torch.einsum(EQUATION, a, b)")
        code.append("")
        code.append("    # --- 2. Run cuTile Kernel ---")
        code.append("    # Launch configuration")
        code.append("    stream = torch.cuda.current_stream().cuda_stream")
        code.append("    grid = (1, 1, 1)")
        code.append("")
        code.append("    print('Launching cuTile kernel...')")
        code.append("    ct.launch(stream, grid, generated_contraction, (a, b, c_cutile))")
        code.append("    torch.cuda.synchronize() # Wait for completion")
        code.append("")
        code.append("    # --- 3. Compare ---")
        code.append("    passed = torch.allclose(c_ref, c_cutile, atol=1e-3, rtol=1e-3)")
        code.append("    print(f'Match: {passed}')")
        code.append("    if not passed:")
        code.append("        print(f'Ref sample: {c_ref.flatten()[:3]}')")
        code.append("        print(f'Cutile sample: {c_cutile.flatten()[:3]}')")
        code.append("")
        code.append("if __name__ == '__main__':")
        code.append("    main()")

        # Write to file
        with open(self.filename, "w") as f:
            f.write("\n".join(code))
        
        print(f"Code successfully generated in: {self.filename}")
        print("Executing generated script...")
        print("-" * 30)
        
        # Execute the file
        subprocess.run([sys.executable, self.filename])

# --- USER CONFIGURATION ---
eqn = "bchw,ck->bhwk"
# Dimensions adjusted to be small multiples of 16/32 for safe tile sizing if needed
dims = {'b': 2, 'c': 16, 'h': 4, 'w': 4, 'k': 32}

# Run Generator
gen = CuTileSolutionGenerator(eqn, dims)
gen.generate_and_run()