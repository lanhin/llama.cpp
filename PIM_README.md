# llama.cpp (PIM branch)

## 1. Build llama.cpp for PIM
Make sure you have your PIM environment (e.g. UPMEM) set correctly already. Then try:
```
cd llama.cpp
make LLAMA_PIM=1
# make LLAMA_PIM=1 -j

# clean:
# make clean
```

## 2. Run llama.cpp with PIM
Prepare your model files as the original README.md shows. A 4-bit-quantified model in gguf format is prefered.

```
./llama-cli -m /mnt/LLM-models/chinese-alpaca-2-7b/gguf/chinese-alpaca-7b_q4_0.gguf  --temp 0 -t 1 --no-warmup -p "列举5个北京经典美食。只列举名字，不要介绍。"
```

## 3. llama-ts for tensor test
A set of tensor utility functions have been implemented (as described in `include/trace_driver.h`), and `example/tensor/ts.cpp` is a good starting point to learn how to import tensors from data files and operate them.

Some snippets in `ggml/src/ggml.c` show how to export a tensor into data file, such as:
```c
#include "trace_driver.h"

stuct ggml_tensor * src0 = ...
...
const char* filenamea = "a.tensor";
tensor_export(src0, filenamea);
```

`example/tensor/ts.cpp` will be built as `llama-ts` after the upper `make` command.


## 4. More details
### 4.1 How we control the model layers computed on PIM
There are several macros defined in `include/llama.h` that controls the bahavior of llama-cli:

```c++
#ifdef PIM_KERNEL
#define NR_DPUS 64
#define NR_LAYER 2
#define DPU_BINARY "./dpu/gemv_dpu"
...
#endif // PIM_KERNEL
```

### 4.2 The PIM function(s) implementation
The PIM binary `dpu/gemv_dpu` is built from `dpu/dpu_main.c` by typing:
```shell
cd dpu
./pim_build.sh
```
So check `dpu/dpu_main.c` to find out how the kernel is implemented.
