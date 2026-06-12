"""
03 - Multiplicación de matrices por bloques (C = A @ B).

Este es el caso de uso estrella de Triton. Cada program calcula un BLOQUE
(tile) de la matriz de salida C, recorriendo la dimensión K en pasos de
BLOCK_SIZE_K y acumulando productos parciales con tl.dot (que aprovecha los
Tensor Cores cuando están disponibles).

A: (M, K)   B: (K, N)   C: (M, N)
Ejecuta:  python 03_matmul.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Cada program calcula un tile C[pid_m, pid_n] de tamaño (BLOCK_M, BLOCK_N).
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets de filas (de A/C) y columnas (de B/C) para este tile.
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Punteros a los primeros sub-bloques de A y B.
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Acumulador en fp32 para precisión.
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Recorremos la dimensión K en pasos de BLOCK_SIZE_K.
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)

        acc += tl.dot(a, b)   # producto de bloques (Tensor Cores)

        # Avanzamos los punteros al siguiente sub-bloque en K.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = acc.to(tl.float16)

    # Escribimos el tile de C con máscara en los bordes.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Dimensiones incompatibles"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    # Grid 2D: un program por tile de C.
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=BLOCK_K,
    )
    return c


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("Necesitas una GPU con CUDA para ejecutar este kernel.")

    torch.manual_seed(0)
    a = torch.randn((512, 512), device="cuda", dtype=torch.float16)
    b = torch.randn((512, 512), device="cuda", dtype=torch.float16)

    out_triton = matmul(a, b)
    out_torch = torch.matmul(a, b)

    max_diff = (out_triton - out_torch).abs().max().item()
    print(f"Diferencia máxima Triton vs PyTorch: {max_diff:.3e}")
    # En fp16 toleramos una pequeña diferencia numérica.
    assert torch.allclose(out_triton, out_torch, atol=1e-1, rtol=1e-1), "¡Resultados muy distintos!"
    print("OK ✓  El matmul por bloques en Triton coincide con PyTorch (dentro de la tolerancia fp16).")
