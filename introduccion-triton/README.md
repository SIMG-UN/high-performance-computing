# Introducción a Triton

Material introductorio para aprender **[Triton](https://github.com/triton-lang/triton)**, el lenguaje y compilador de OpenAI para escribir *kernels* de GPU de alto rendimiento directamente en Python.

> Repositorio del Semillero de Investigación en *High Performance Computing*.

---

## ¿Qué es Triton?

Triton es un lenguaje embebido en Python que permite escribir código que se ejecuta en la GPU con un rendimiento cercano al de CUDA, pero sin tener que manejar manualmente la mayoría de los detalles de bajo nivel.

La idea central es la siguiente:

| CUDA (C/C++)                                  | Triton (Python)                                       |
| --------------------------------------------- | ----------------------------------------------------- |
| Razonas a nivel de **hilo** (*thread*)        | Razonas a nivel de **bloque** (*tile*)                |
| Gestionas memoria compartida manualmente      | El compilador gestiona la memoria compartida          |
| Coalescing y *bank conflicts* a mano          | El compilador optimiza accesos a memoria              |
| Curva de aprendizaje pronunciada              | Sintaxis tipo NumPy, mucho más accesible              |

En Triton **programas un bloque de datos a la vez** y el compilador se encarga de
repartir el trabajo entre los hilos, optimizar los accesos a memoria y la
paralelización dentro del bloque.

---

## Modelo de programación

```
                 GRID (lanzamiento del kernel)
   ┌─────────────┬─────────────┬─────────────┐
   │  program 0  │  program 1  │  program 2  │  ...
   └─────────────┴─────────────┴─────────────┘
         │              │             │
         ▼              ▼             ▼
   procesa un      procesa un     procesa un
   BLOQUE de       BLOQUE de      BLOQUE de
   datos           datos          datos
```

- El **grid** define cuántas instancias del kernel (*programs*) se lanzan.
- Cada *program* se identifica con `tl.program_id(axis)`.
- Cada *program* procesa un **bloque** (`BLOCK_SIZE` elementos) usando operaciones
  vectorizadas tipo NumPy (`tl.load`, `tl.store`, `tl.arange`, ...).

---

## Requisitos

- **GPU NVIDIA** con CUDA (o AMD ROCm en versiones recientes).
- Python 3.9+
- PyTorch (Triton viene incluido al instalar PyTorch con soporte CUDA).

```bash
pip install -r requirements.txt
```

Verifica que todo funcione:

```bash
python -c "import torch, triton; print('Triton', triton.__version__, '| CUDA', torch.cuda.is_available())"
```

> **¿No tienes GPU?** Puedes usar [Google Colab](https://colab.research.google.com)
> (entorno de ejecución → GPU T4 gratis) y subir los notebooks de `notebooks/`.

---

## Contenido del repositorio

```
introduccion-triton/
├── README.md              ← estás aquí
├── docs/
│   └── 01_conceptos.md     ← teoría: grid, blocks, máscaras, autotuning
├── kernels/
│   ├── 01_vector_add.py    ← "hola mundo": suma de vectores
│   ├── 02_fused_softmax.py ← softmax fusionado en un solo kernel
│   └── 03_matmul.py        ← multiplicación de matrices por bloques
├── benchmarks/
│   └── bench_vector_add.py ← comparación Triton vs PyTorch
└── notebooks/
    ├── README.md                          ← guía para usar Colab
    ├── Sesion_01_Introduccion_a_Triton.ipynb  ← primeros kernels en vivo
    └── Sesion_02_Profiling.ipynb          ← cómo medir y perfilar kernels
```

---

## Ruta de aprendizaje sugerida

1. **Lee** [`docs/01_conceptos.md`](docs/01_conceptos.md) para entender el modelo de programación.
2. **Ejecuta** [`kernels/01_vector_add.py`](kernels/01_vector_add.py) — el equivalente al "Hola Mundo".
3. **Estudia** [`kernels/02_fused_softmax.py`](kernels/02_fused_softmax.py) para ver por qué *fusionar* operaciones acelera el cómputo.
4. **Profundiza** con [`kernels/03_matmul.py`](kernels/03_matmul.py), el caso de uso estrella de Triton.
5. **Mide** con [`benchmarks/bench_vector_add.py`](benchmarks/bench_vector_add.py).
6. **Perfila** con [`notebooks/Sesion_02_Profiling.ipynb`](notebooks/Sesion_02_Profiling.ipynb): aprende a medir bien (`do_bench`), distinguir kernels *memory-bound* de *compute-bound* (*roofline*) y diagnosticar por qué un kernel es lento.

---

## Recursos oficiales

- 📖 [Tutoriales oficiales de Triton](https://triton-lang.org/main/getting-started/tutorials/index.html)
- 💻 [Repositorio en GitHub](https://github.com/triton-lang/triton)
- 📝 [Paper original (Tillet et al., 2019)](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)

---

## Licencia

MIT — uso educativo libre.
