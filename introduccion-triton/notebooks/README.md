# Notebooks — ejecutar Triton en Google Colab

¿No tienes una GPU NVIDIA a mano? Colab ofrece GPUs gratis y ya trae PyTorch +
Triton instalados.

## Pasos

1. Abre [Google Colab](https://colab.research.google.com).
2. Menú **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU (T4)**.
3. Verifica el entorno en una celda:

   ```python
   import torch, triton
   print("Triton", triton.__version__)
   print("GPU:", torch.cuda.get_device_name(0))
   ```

4. Copia el contenido de cualquier kernel de `../kernels/` en una celda y ejecútalo.

## Notebooks disponibles

| Notebook | Contenido |
| -------- | --------- |
| [`Sesion_01_Introduccion_a_Triton.ipynb`](Sesion_01_Introduccion_a_Triton.ipynb) | Modelo de programación, primeros kernels escritos en vivo. |
| [`Sesion_02_Profiling.ipynb`](Sesion_02_Profiling.ipynb) | Medir bien con `do_bench`, ancho de banda vs FLOP/s (*roofline*), inspección del kernel compilado (`n_regs`, *spills*, PTX), `torch.profiler`, Proton y Nsight. **Autocontenido**: corre tal cual en Colab. |

## Sugerencia

Empieza pegando `01_vector_add.py` completo en una sola celda y ejecútala.
Deberías ver el mensaje `OK ✓`. Luego experimenta cambiando `BLOCK_SIZE`
(prueba 64, 256, 1024) y observa si cambia algo.

> Más adelante puedes convertir cada script en su propio `.ipynb` y guardarlos
> aquí para tener el material interactivo.
