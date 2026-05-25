# Contributing to PallasBench

We welcome contributions! Here's how to get involved.

## Adding New Tasks

1. Create a new `.py` file in the appropriate `pallasbench/kernels/levelN/` directory
2. Implement both `pallas_kernel` (the Pallas implementation) and export module-level metadata:
   - `pallas_kernel`: the callable Pallas function
   - `task_name`: short identifier
   - `input_shapes`: list of tuples defining input dimensions
   - `category`: task category string
   - `level`: 1, 2, or 3
3. Add a corresponding JAX baseline in `pallasbench/baselines/jax_baseline.py`
4. Register the task in `pallasbench/tasks.py`

## Task Naming Convention

- Level 1: `L1/{operation}` (e.g., `L1/relu`, `L1/matmul`)
- Level 2: `L2/{fused_ops}` (e.g., `L2/matmul_relu`, `L2/rmsnorm_residual`)
- Level 3: `L3/{architecture}` (e.g., `L3/flash_attention`)

## Submitting Results

Run the benchmark on your hardware and submit the JSON output:
```bash
python scripts/run_benchmark.py --levels 1 2 3
```

Include in your PR:
- The result JSON file
- Hardware details (TPU generation or GPU model)
- JAX version

## Code Style

- Format with `black`
- Lint with `ruff`
- Type hints where practical
