#include <stdlib.h>
#include <stdio.h>
#include "trace_driver.h"

void tensor_export(const struct ggml_tensor * tensor, const char * fname) {
    FILE * fout = ggml_fopen(fname, "wb");
    if (!fout) {
        fprintf(stderr, "%s: failed to open %s\n", __func__, fname);
        return;
    }

    const uint32_t type   = tensor->type;
    const uint32_t op     = tensor->op;
    const int32_t  flags  = tensor->flags;

    fwrite(&type,   sizeof(uint32_t), 1, fout);
    fwrite(&op,     sizeof(uint32_t), 1, fout);
    fwrite(&flags,  sizeof(int32_t),  1, fout);

    for (int j = 0; j < GGML_MAX_DIMS; ++j) {
        const uint64_t ne = tensor->ne[j];
        const uint64_t nb = tensor->nb[j];

        fwrite(&ne, sizeof(uint64_t), 1, fout);
        fwrite(&nb, sizeof(uint64_t), 1, fout);
    }

    fwrite(tensor->name,      sizeof(char), GGML_MAX_NAME,      fout);

    // dump the data
    // TODO: pad this to 32 byte boundary
    {
        const size_t size = ggml_nbytes(tensor);

        fwrite(tensor->data, sizeof(char), size, fout);
    }

    fclose(fout);
}

struct ggml_tensor * tensor_import(const char * fname) {
  char * data = NULL;
  FILE * fin = ggml_fopen(fname, "rb");
  if (!fin) {
    fprintf(stderr, "%s: failed to open %s\n", __func__, fname);
    return NULL;
  }

  size_t fsize = 0;

  fseek(fin, 0, SEEK_END);
  fsize = ftell(fin);
  fseek(fin, 0, SEEK_SET);

  data = (char*)malloc((fsize+4) * sizeof(char));

  const size_t ret = fread(data, sizeof(char), fsize, fin);
  if (ret != fsize) {
    fprintf(stderr, "%s: failed to read %s\n", __func__, fname);
    fclose(fin);
    return NULL;
  }

  fclose(fin);

  uint32_t type;
  uint32_t op;
  int32_t  flags;
  char* ptr = data;

  type   = *(const uint32_t *) ptr; ptr += sizeof(type);
  op     = *(const uint32_t *) ptr; ptr += sizeof(op);
  flags  = *(const int32_t  *) ptr; ptr += sizeof(flags);

  int64_t ne[GGML_MAX_DIMS];
  size_t  nb[GGML_MAX_DIMS];

  for (int j = 0; j < GGML_MAX_DIMS; ++j) {
    uint64_t ne_cur;
    uint64_t nb_cur;

    ne_cur = *(const uint64_t *) ptr; ptr += sizeof(ne_cur);
    nb_cur = *(const uint64_t *) ptr; ptr += sizeof(nb_cur);

    ne[j] = ne_cur;
    nb[j] = nb_cur;
  }

  struct ggml_tensor * tensor = (struct ggml_tensor *)malloc(sizeof(struct ggml_tensor));

  tensor->type = type;
  tensor->op    = (enum ggml_op) op;
  tensor->flags = flags;

  for (int j = 0; j < GGML_MAX_DIMS; ++j) {
    tensor->ne[j] = ne[j];
    tensor->nb[j] = nb[j];
  }

  memcpy(tensor->name, ptr, GGML_MAX_NAME);
  ptr += GGML_MAX_NAME;

  tensor->data = (void *) ptr; ptr += ggml_nbytes(tensor);

  return tensor;
}

void dump_tensor_first_n(const struct ggml_tensor * tensor, int n, FILE * fout) {
    const int64_t * ne = tensor->ne;
    const size_t  * nb = tensor->nb;
    fprintf(fout, "%-6s %-12s %8d %ld %ld %ld %ld %16zu %16zu %16zu %16zu %16p %32s\n",
            ggml_type_name(tensor->type),
            ggml_op_name  (tensor->op),
            ggml_n_dims(tensor),
            ne[0], ne[1], ne[2], ne[3],
            nb[0], nb[1], nb[2], nb[3],
            tensor->data,
            tensor->name);

    int elements_to_dump = n;
    if (elements_to_dump > ggml_nelements(tensor)) {
      elements_to_dump = ggml_nelements(tensor);
    }
    int blck_size = ggml_blck_size(tensor->type);
    int blcks_to_dump = elements_to_dump = (elements_to_dump + blck_size - 1)/ blck_size;

    for (int i=0; i < blcks_to_dump; i++) {
      switch(tensor->type){
      case GGML_TYPE_F32:
      case GGML_TYPE_F16:
      case GGML_TYPE_BF16:
        {
          fprintf(fout, "[%d] = %f    ", i, (double)ggml_get_f32_1d(tensor, i));
          if (i % 4 == 3) {
            fprintf(fout, "\n");
          }
        }
        break;
      case GGML_TYPE_Q4_0:
        {
          const block_q4_0 * x = tensor->data;
          fprintf(fout, "i = %d, delta = %f, qs: ", i, GGML_FP16_TO_FP32(x[i].d));
          for (int j=0; j < QK4_0/2; j++) {
            const int v0 = (x[i].qs[j] & 0x0f) - 8;
            const int v1 = (x[i].qs[j]  >> 4) - 8;
            fprintf(fout, "%d, %d, ", v0, v1);
          }
          fprintf(fout, "\n");
        }
        break;
      case GGML_TYPE_Q8_0:
        {
          const block_q8_0 * x = tensor->data;
          fprintf(fout, "i = %d, delta = %f, qs:", i, GGML_FP16_TO_FP32(x[i].d));
          for (int j=0; j < QK8_0; j++) {
            const int v0 = (x[i].qs[j]);
            fprintf(fout, "%d,", v0);
          }
          fprintf(fout, "\n");
        }
        break;
      default:
        {
          fprintf(fout, "Unsupported tensor type: %d\n", tensor->type);
        }
      }
    }
}

void dump_tensor(const struct ggml_tensor * tensor, FILE * fout) {
    const int64_t * ne = tensor->ne;
    const size_t  * nb = tensor->nb;
    fprintf(fout, "%-6s %-12s %8d %ld %ld %ld %ld %16zu %16zu %16zu %16zu %16p %32s\n",
            ggml_type_name(tensor->type),
            ggml_op_name  (tensor->op),
            ggml_n_dims(tensor),
            ne[0], ne[1], ne[2], ne[3],
            nb[0], nb[1], nb[2], nb[3],
            tensor->data,
            tensor->name);

    int ele_stride = 128;
    ele_stride = ele_stride / ggml_blck_size(tensor->type);
    if (ele_stride < ggml_nelements(tensor) / ggml_blck_size(tensor->type) / 64) {
      ele_stride = ggml_nelements(tensor) / ggml_blck_size(tensor->type) / 64;
    }
    fprintf(fout, "Element with stride %d:\n", ele_stride);
    for (int i=0; i < ggml_nelements(tensor) / ggml_blck_size(tensor->type); i += ele_stride) {
      switch(tensor->type){
      case GGML_TYPE_F32:
      case GGML_TYPE_F16:
      case GGML_TYPE_BF16:
        {
          fprintf(fout, "[%d] = %f    ", i, (double)ggml_get_f32_1d(tensor, i));
          if ((i / ele_stride) % 4 == 3) {
            fprintf(fout, "\n");
          }
        }
        break;
      case GGML_TYPE_Q4_0:
        {
          const block_q4_0 * x = tensor->data;
          fprintf(fout, "i = %d, delta = %u->%f, qs: ", i, x[i].d, GGML_FP16_TO_FP32(x[i].d));
          for (int j=0; j < QK4_0/2; j++) {
            const int v0 = (x[i].qs[j] & 0x0f) - 8;
            const int v1 = (x[i].qs[j]  >> 4) - 8;
            fprintf(fout, "%d, %d, ", v0, v1);
          }
          fprintf(fout, "\n");
        }
        break;
      case GGML_TYPE_Q8_0:
        {
          const block_q8_0 * x = tensor->data;
          fprintf(fout, "i = %d, delta = %u->%f, qs:", i, x[i].d, GGML_FP16_TO_FP32(x[i].d));
          for (int j=0; j < QK8_0; j++) {
            const int v0 = (x[i].qs[j]);
            fprintf(fout, "%d,", v0);
          }
          fprintf(fout, "\n");
        }
        break;
      default:
        {
          fprintf(fout, "Unsupported tensor type: %d\n", tensor->type);
        }
      }
    }
    fprintf(fout, "\n");
}

float mul_add_q4_0_q8_0(struct ggml_tensor * a, struct ggml_tensor *b) {
  GGML_ASSERT(a->type == GGML_TYPE_Q4_0);
  GGML_ASSERT(b->type == GGML_TYPE_Q8_0);
  const block_q4_0 *x = a->data;
  const block_q8_0 *y = b->data;

  int nb = 4096/32;
  float res = 0.0;
  for (int ib = 0; ib < nb; ++ib) {
    int sumi0 = 0;
    int sumi1 = 0;

    for (int j = 0; j < QK4_0/2; ++j) {
      const int v0 = (x[ib].qs[j] & 0x0F) - 8;
      const int v1 = (x[ib].qs[j] >>   4) - 8;

      sumi0 += (v0 * y[ib].qs[j]);
      sumi1 += (v1 * y[ib].qs[j + QK4_0/2]);
    }

    int sumi = sumi0 + sumi1;
    res += sumi*GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d);
  }
  return res;
}
