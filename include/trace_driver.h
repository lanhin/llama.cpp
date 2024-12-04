#ifndef TRACE_DRIVER_H
#define TRACE_DRIVER_H
#include "ggml.h"
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml-cpu-impl.h"

void tensor_export(const struct ggml_tensor * tensor, const char * fname);
struct ggml_tensor * tensor_import(struct ggml_context *ctx, const char * fname);
void dump_tensor(const struct ggml_tensor * tensor, FILE * fout);
#endif
