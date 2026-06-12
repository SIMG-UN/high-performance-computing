"""
01 - Suma de vectores: el "Hola Mundo" de Triton.

Calcula  z = x + y  donde x, y, z son vectores de tamaño n.

Cada *program* del grid se encarga de un bloque de BLOCK_SIZE elementos.
Ejecuta:  python 01_vector_add.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,            # puntero al primer elemento de x
    y_ptr,            # puntero al primer elemento de y
    out_ptr,          # puntero al primer elemento de la salida
    n_elements,       # tamaño de los vectores
    BLOCK_SIZE: tl.constexpr,   # cuántos elementos procesa cada program
):
    # 1) ¿Qué program soy? -> me toca un bloque distinto.
    pid = tl.program_id(axis=0)

    # 2) Índices globales que procesa este program.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 3) Máscara para no salirnos del tensor en el último bloque.
    mask = offsets < n_elements

    # 4) Cargar desde memoria global -> registros.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 5) Computar.
    output = x + y

    # 6) Guardar el resultado en memoria global.
    tl.store(out_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Wrapper de Python que lanza el kernel de Triton."""
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()

    # El grid es 1D: necesitamos ceil(n / BLOCK_SIZE) programs.
    # `meta` da acceso a los argumentos constexpr (aquí BLOCK_SIZE).
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("Necesitas una GPU con CUDA para ejecutar este kernel.")

    torch.manual_seed(0)
    size = 98_432
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")

    out_triton = add(x, y)
    out_torch = x + y

    max_diff = (out_triton - out_torch).abs().max().item()
    print(f"Diferencia máxima Triton vs PyTorch: {max_diff:.3e}")
    assert max_diff == 0, "¡Los resultados no coinciden!"
    print("OK ✓  La suma de vectores en Triton coincide con PyTorch.")
