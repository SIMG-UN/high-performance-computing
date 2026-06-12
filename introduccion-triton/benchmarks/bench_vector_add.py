"""
Benchmark: suma de vectores en Triton vs PyTorch.

Usa triton.testing para medir el ancho de banda efectivo (GB/s) en función
del tamaño del vector y genera una gráfica comparativa.

Ejecuta:  python bench_vector_add.py
La gráfica se guarda como 'vector_add_perf.png' en este directorio.
"""
import os
import sys

import torch
import triton

# Permite importar el kernel desde la carpeta hermana 'kernels/'.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "kernels"))
from importlib import import_module

add = import_module("01_vector_add").add  # nombre de módulo con dígitos -> import dinámico


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(12, 28, 1)],   # de 4K a 128M elementos
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "PyTorch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="vector_add_perf",
        args={},
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device="cuda", dtype=torch.float32)
    y = torch.rand(size, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    else:  # triton
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)

    # 3 accesos a memoria (leer x, leer y, escribir z) * 4 bytes (fp32).
    gbps = lambda t: 3 * x.numel() * x.element_size() / (t * 1e-3) / 1e9
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("Necesitas una GPU con CUDA para correr el benchmark.")

    print("Midiendo... (esto puede tardar unos segundos)\n")
    benchmark.run(print_data=True, show_plots=False, save_path=os.path.dirname(__file__))
    print("\nGráfica guardada en benchmarks/vector_add_perf.png")
