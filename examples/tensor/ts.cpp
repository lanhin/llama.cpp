#include "trace_driver.h"
#include <iostream>
#include <iomanip>

void fp_table_init(void) {
  for (int i = 0; i < (1 << 16); ++i) {
                union {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
            }
}
int main(int argc, char** argv) {
  // init fp table for fp16 dump
  fp_table_init();

  const char* filenamea = "a.tensor";
  const char* filenameb = "b.tensor";
  const char* filenamebq = "b_quant.tensor";
  const char* filenamec = "c.tensor";
  struct ggml_tensor * ts_a = tensor_import(filenamea);
  struct ggml_tensor * ts_b = tensor_import(filenameb);
  struct ggml_tensor * ts_bq = tensor_import(filenamebq);
  struct ggml_tensor * ts_c = tensor_import(filenamec);
  std::cout<<"ts_a:"<<std::endl;
  dump_tensor(ts_a, stdout);
  std::cout<<"ts_b:"<<std::endl;
  dump_tensor(ts_b, stdout);
  std::cout<<"ts_bq:"<<std::endl;
  dump_tensor(ts_bq, stdout);
  std::cout<<"ts_c:"<<std::endl;
  dump_tensor(ts_c, stdout);

  //dump_tensor_first_n(ts_a, 4096, stdout);
  //dump_tensor_first_n(ts_bq, 4096, stdout);

  float first_res = mul_add_q4_0_q8_0(ts_a, ts_bq);
  std::cout<<"first element: "<<std::fixed << std::setprecision(6)<<first_res<<std::endl;
  return 0;
}
