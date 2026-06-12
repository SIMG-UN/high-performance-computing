"""
02 - Softmax fusionado por filas.

Calcula softmax(x) a lo largo del eje 1 (cada fila por separado).

La gracia: PyTorch lanzaría varios kernels (max, resta, exp, suma, división)
y escribiría tensores intermedios en memoria. Aquí lo hacemos TODO en un
único kernel, manteniendo la fila en SRAM. Eso ahorra muchísimo tráfico de
memoria, que es el verdadero cuello de botella en GPU.

Suposición didáctica: cada fila cabe en un solo bloque (BLOCK_SIZE >= n_cols).
Ejecuta:  python 02_fused_softmax.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    out_ptr, in_ptr,
    in_row_stride, out_row_stride,   # cuántos elementos hay que saltar para pasar de fila a fila
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Cada program procesa UNA fila completa.
    row_idx = tl.program_id(0)

    # Puntero al inicio de la fila correspondiente.
    row_start_ptr = in_ptr + row_idx * in_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    in_ptrs = row_start_ptr + col_offsets

    # Máscara: BLOCK_SIZE puede ser mayor que el nº real de columnas.
    mask = col_offsets < n_cols

    # Cargamos la fila. Donde no hay datos ponemos -inf para que no afecte al max.
    row = tl.load(in_ptrs, mask=mask, other=-float("inf"))

    # --- softmax numéricamente estable, todo en registros/SRAM ---
    row_minus_max = row - tl.max(row, axis=0)   # estabilidad numérica
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # Escribimos la fila resultante.
    out_row_ptr = out_ptr + row_idx * out_row_stride
    tl.store(out_row_ptr + col_offsets, softmax_output, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    # Tamaño de bloque = siguiente potencia de 2 >= n_cols (requisito de tl.arange).
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    out = torch.empty_like(x)
    # Un program por fila.
    softmax_kernel[(n_rows,)](
        out, x,
        x.stride(0), out.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("Necesitas una GPU con CUDA para ejecutar este kernel.")

    torch.manual_seed(0)
    x = torch.randn(1823, 781, device="cuda")

    out_triton = softmax(x)
    out_torch = torch.softmax(x, axis=1)

    max_diff = (out_triton - out_torch).abs().max().item()
    print(f"Diferencia máxima Triton vs PyTorch: {max_diff:.3e}")
    assert torch.allclose(out_triton, out_torch, atol=1e-6), "¡Los resultados no coinciden!"
    print("OK ✓  El softmax fusionado en Triton coincide con PyTorch.")
