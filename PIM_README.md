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
./llama-cli -m /mnt/LLM-models/chinese-alpaca-2-7b/gguf/chinese-alpaca-7b_q4_0.gguf \
--temp 0 -t 1 --no-warmup -p "列举5个北京经典美食。只列举名字，不要介绍。"
```

Which may output:
```shell
...
sampler seed: 4294967295
sampler params:
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.000
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampler chain: logits -> logit-bias -> penalties -> greedy
generate: n_ctx = 4096, n_batch = 2048, n_predict = -1, n_keep = 1

 列举5个北京经典美食。只列举名字，不要介绍。1. 烤鸭 2. 炸酱面 3. 豆汁 4. 羊蝎子 5. 驴打滚 [end of text]


llama_perf_sampler_print:    sampling time =       1.02 ms /    49 runs   (    0.02 ms per token, 47804.88 tokens per second)
llama_perf_context_print:        load time =    4097.04 ms
llama_perf_context_print: prompt eval time =    2966.36 ms /    16 tokens (  185.40 ms per token,     5.39 tokens per second)
llama_perf_context_print:        eval time =   12105.60 ms /    32 runs   (  378.30 ms per token,     2.64 tokens per second)
llama_perf_context_print:       total time =   16206.10 ms /    48 tokens

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
#define NR_DPUS 64    //Number of DPUs to execute the kernel
#define NR_LAYER 2    //Number of transformer layers to offload
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
Check `dpu/dpu_main.c` to find out how the kernel is implemented.
