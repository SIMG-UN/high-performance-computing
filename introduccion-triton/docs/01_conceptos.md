# Conceptos fundamentales de Triton

Esta guía cubre las ideas que necesitas para leer y escribir tus primeros kernels.

---

## 1. El decorador `@triton.jit`

Una función de kernel se marca con `@triton.jit`. Triton compila esa función a
código de GPU (PTX en NVIDIA) la primera vez que se invoca (*Just-In-Time*).

```python
import triton
import triton.language as tl

@triton.jit
def mi_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    ...
```

- Los argumentos `*_ptr` son **punteros** a tensores en memoria de GPU.
- Los argumentos marcados como `tl.constexpr` son **constantes en tiempo de
  compilación**; el compilador genera código especializado para cada valor
  (por eso `BLOCK_SIZE` suele ser `constexpr`).

---

## 2. El *grid* y `program_id`

Al lanzar el kernel defines un **grid**: cuántas copias (*programs*) se ejecutan
en paralelo.

```python
grid = (triton.cdiv(n, BLOCK_SIZE),)   # nº de bloques necesarios
mi_kernel[grid](x, y, n, BLOCK_SIZE=1024)
```

- `triton.cdiv(a, b)` = división entera redondeada hacia arriba = `ceil(a/b)`.
- Dentro del kernel, cada *program* sabe quién es con:

```python
pid = tl.program_id(axis=0)   # 0, 1, 2, ... (su índice en el grid)
```

---

## 3. Bloques, *offsets* y máscaras

Cada *program* procesa `BLOCK_SIZE` elementos contiguos. Calculamos los índices
globales que le corresponden:

```python
pid     = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

Como `n` rara vez es múltiplo exacto de `BLOCK_SIZE`, el último bloque se sale
del tensor. Para no leer/escribir fuera de memoria usamos una **máscara**:

```python
mask = offsets < n
x = tl.load(x_ptr + offsets, mask=mask)   # solo carga posiciones válidas
tl.store(y_ptr + offsets, x, mask=mask)    # solo escribe posiciones válidas
```

> **Regla de oro:** toda operación `tl.load`/`tl.store` que pueda salirse del
> rango necesita una máscara.

---

## 4. Carga y guardado de memoria

| Función                                  | Qué hace                                    |
| ---------------------------------------- | ------------------------------------------- |
| `tl.load(ptr + offs, mask=m, other=0.0)` | Lee de memoria global a registros           |
| `tl.store(ptr + offs, val, mask=m)`      | Escribe de registros a memoria global       |
| `tl.arange(0, N)`                        | Vector `[0, 1, ..., N-1]` (N debe ser `constexpr`) |

El argumento `other` define qué valor usar donde la máscara es falsa (útil para
reducciones, p. ej. `other=-inf` antes de un `max`).

---

## 5. Operaciones disponibles

`triton.language` ofrece operaciones vectorizadas estilo NumPy que actúan sobre
todo el bloque a la vez:

- Aritmética: `+ - * /`, `tl.exp`, `tl.log`, `tl.sqrt`, `tl.sin`...
- Reducciones: `tl.sum`, `tl.max`, `tl.min` (sobre un eje del bloque).
- `tl.dot(a, b)` — producto matricial de bloques (usa Tensor Cores si es posible).
- `tl.where(cond, a, b)` — selección elemento a elemento.

---

## 6. *Autotuning*

El tamaño de bloque óptimo depende de la GPU y del problema. Triton puede
probar varias configuraciones automáticamente y quedarse con la más rápida:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n'],   # re-autotunea cuando cambia 'n'
)
@triton.jit
def mi_kernel(...):
    ...
```

- `num_warps`: cuántos *warps* (grupos de 32 hilos) ejecutan cada *program*.
- `key`: lista de argumentos cuyo cambio dispara una nueva búsqueda.

---

## 7. ¿Por qué *fusionar* operaciones?

En PyTorch, `y = (x.exp()).sum()` lanza **varios kernels** y escribe resultados
intermedios en memoria global (lento). Un kernel de Triton puede hacer todo el
cálculo **en un solo paso**, manteniendo los datos intermedios en registros/SRAM.

Esto reduce el tráfico a memoria — que suele ser el cuello de botella real en
GPU — y es la razón por la que Triton brilla en operaciones como *softmax*,
*layernorm* o *attention*.

---

## Checklist mental al escribir un kernel

1. ¿Cuántos *programs* lanzo? → define el `grid`.
2. ¿Qué porción de datos toca a cada *program*? → `program_id` + `offsets`.
3. ¿Puedo salirme del tensor? → añade `mask`.
4. ¿Qué constantes especializo? → márcalas como `tl.constexpr`.
5. ¿Puedo fusionar pasos para evitar ir y volver a memoria? → hazlo.

Siguiente paso: abre [`../kernels/01_vector_add.py`](../kernels/01_vector_add.py).
