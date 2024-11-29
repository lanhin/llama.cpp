/*
 * Matrix vector multiplication with multiple tasklet
 *
 */
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <seqread.h>

//#include "../support/common.h"
#define GGML_MAX_DIMS 4

/*
struct pim_meta {
	// 每个类型的tensor在DPU中的metadata
	uint16_t layer_num;
	uint16_t weight_type;
	uint16_t rows_per_dpu;
	uint16_t rest_rows;
	uint32_t size_per_row;
	uint32_t layer_len;
	// 每一层串行计算，dpu的输入也只需要一个地址空间
	uint32_t input_offset;
	uint32_t input_len; 
	// 每一层串行计算，dpu的返回值只需一个地址空间
	uint32_t response_offset;
	uint32_t response_len;
	int64_t ne[GGML_MAX_DIMS];
	size_t	nb[GGML_MAX_DIMS];
};
*/

typedef struct	{
    int32_t type;
    int32_t layerid;
    int64_t ne[GGML_MAX_DIMS];
}pim_matrix_des; // 8 Byte align

	
struct pim_meta {
	// 每个类型的tensor在DPU中的metadata
	uint16_t layer_num;
	uint16_t weight_type;
	uint16_t rows_per_dpu;
	uint16_t rest_rows;
	uint32_t size_per_row;
	uint32_t layer_len;
    // 每一层串行计算，dpu的输入也只需要一个地址空间
    uint32_t input_offset;
    uint32_t input_len;
    // 每一层串行计算，dpu的返回值只需一个地址空间
    uint32_t response_offset;
    uint32_t response_len;

    pim_matrix_des weight_des;
};

enum ggml_type {
        GGML_TYPE_F32     = 0,
        GGML_TYPE_F16     = 1,
        GGML_TYPE_Q4_0    = 2,
        GGML_TYPE_Q4_1    = 3,
        // GGML_TYPE_Q4_2 = 4, support has been removed
        // GGML_TYPE_Q4_3 = 5, support has been removed
        GGML_TYPE_Q5_0    = 6,
        GGML_TYPE_Q5_1    = 7,
        GGML_TYPE_Q8_0    = 8,
        GGML_TYPE_Q8_1    = 9,
        GGML_TYPE_Q2_K    = 10,
        GGML_TYPE_Q3_K    = 11,
        GGML_TYPE_Q4_K    = 12,
        GGML_TYPE_Q5_K    = 13,
        GGML_TYPE_Q6_K    = 14,
        GGML_TYPE_Q8_K    = 15,
        GGML_TYPE_IQ2_XXS = 16,
        GGML_TYPE_IQ2_XS  = 17,
        GGML_TYPE_IQ3_XXS = 18,
        GGML_TYPE_IQ1_S   = 19,
        GGML_TYPE_IQ4_NL  = 20,
        GGML_TYPE_IQ3_S   = 21,
        GGML_TYPE_IQ2_S   = 22,
        GGML_TYPE_IQ4_XS  = 23,
        GGML_TYPE_I8      = 24,
        GGML_TYPE_I16     = 25,
        GGML_TYPE_I32     = 26,
        GGML_TYPE_I64     = 27,
        GGML_TYPE_F64     = 28,
        GGML_TYPE_IQ1_M   = 29,
        GGML_TYPE_BF16    = 30,
        GGML_TYPE_Q4_0_4_4 = 31,
        GGML_TYPE_Q4_0_4_8 = 32,
        GGML_TYPE_Q4_0_8_8 = 33,
        GGML_TYPE_TQ1_0   = 34,
        GGML_TYPE_TQ2_0   = 35,
        GGML_TYPE_COUNT,
    };


#define QK4_0 32
typedef struct {
    float   d;          // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;

#define QK8_0 32
typedef struct {
    float   d;          // delta
    int8_t  qs[QK8_0];  // quants
} block_q8_0;


//float __mram_noinit table_f32_f16[1 << 16];
float *ptable_f32_f16 = NULL;

inline static float lookup_fp16_to_fp32(uint16_t f) {
	uint16_t s;
	memcpy(&s, &f, sizeof(uint16_t));
	return ptable_f32_f16[s];
}
#define FP16_TO_FP32(x) lookup_fp16_to_fp32(x)


//__host struct pim_meta DPU_INPUT_ARGUMENTS;
//__host struct pim_meta DPU_INPUT_ARGUMENTS;

/*
// GEMV
static void gemv(T *bufferC, T *bufferA, T *bufferB, int pos) {
	//for (unsigned int i = 0; i < BLOCK_SIZE / sizeof(T); i++) {
	for (unsigned int i = 0; i < (BLOCK_SIZE>>1); i++) {
		bufferC[pos] += bufferA[i] * bufferB[i];
		//bufferC[pos] = bufferA[i] + bufferB[i];
	}
	return;
}

static void gemv2(unsigned short *bufferC, unsigned short *bufferA, unsigned short *bufferB, int pos) {
	for (unsigned int i = 0; i < BLOCK_SIZE / sizeof(T); i++) {
		//bufferC[pos] += bufferA[i] * bufferB[i];
		bufferC[pos] += bufferA[i] * bufferB[i];
	}
	return;
}

static void gemv3(unsigned char *bufferC, unsigned char *bufferA, unsigned char *bufferB, int pos) {
	for (unsigned int i = 0; i < BLOCK_SIZE / sizeof(T); i++) {
		//bufferC[pos] += bufferA[i] * bufferB[i];
		bufferC[pos] = bufferA[i] * bufferB[i];
	}
	return;
}
*/

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

/*
DPU MRAM Memory:
/*
|--Quantify-tbl--  |--DPU0-weight-Metadata--  |--layer0-subweight0--pading--  |--layer1-subweight0--pading--  |...|--layer31-subweight0--pading--  |--input-output-metadata--|--input-token--|---output0--pading--| 
|--Quantify-tbl--  |--DPU1-weight-Metadata--  |--layer0-subweight1--pading--  |--layer1-subweight1--pading--  |...|--layer31-subweight1--pading--  |--input-output-metadata--|--input-token--|---output1--pading--|
......
|--Quantify-tbl--  |--DPU127-weight-Metadata--|--layer0-subweight127--pading--|--layer1-subweight127--pading--|...|--layer31-subweight127--pading--|--input-output-metadata--|--input-token--|---output127--pading--|
*/

// main
int main() {
	unsigned int tasklet_id = me();
#if PRINT
	// printf("tasklet_id = %u\n", tasklet_id);
#endif
	if (tasklet_id == 0){ // Initialize once the cycle counter
		mem_reset(); // Reset the heap
	}
	// Barrier
	barrier_wait(&my_barrier);
	
/*
	uint16_t layer_num = DPU_INPUT_ARGUMENTS.layer_num;
	uint16_t weight_type = DPU_INPUT_ARGUMENTS.weight_type;
	uint16_t rows_per_dpu = DPU_INPUT_ARGUMENTS.rows_per_dpu;
	uint16_t size_per_row = DPU_INPUT_ARGUMENTS.size_per_row;
	uint16_t rest_rows = DPU_INPUT_ARGUMENTS.rest_rows;
	uint32_t response_offset = DPU_INPUT_ARGUMENTS.response_offset;;
	uint32_t response_len = DPU_INPUT_ARGUMENTS.response_len;;
	uint32_t input_offset = DPU_INPUT_ARGUMENTS.input_offset;;
	uint32_t input_len = DPU_INPUT_ARGUMENTS.input_len ;; 
*/
    //fp32->fp16 table
    ptable_f32_f16 = (float *)DPU_MRAM_HEAP_POINTER;
    uint32_t table_f32_f16_len = 1 << 16;
	uint32_t offset = table_f32_f16_len;

    //weight metadata
    uint32_t weightmetadatabase = (uint32_t) (DPU_MRAM_HEAP_POINTER + offset);
    struct pim_meta *cache_meta = (struct pim_meta *) mem_alloc(sizeof(struct pim_meta));
    mram_read((__mram_ptr void const*) (weightmetadatabase), cache_meta, sizeof(struct pim_meta));

	printf("layer_num: %d, weight_type=%d,rows_per_dpu=%d,size_per_rosw=%d,rest_rows=%d,response_offset=%d,response_len=%d,input_offset=%d,input_len=%d\n",
		cache_meta->layer_num,cache_meta->weight_type,cache_meta->rows_per_dpu,cache_meta->size_per_row,cache_meta->rest_rows,cache_meta->response_offset,
		cache_meta->response_len,cache_meta->input_offset,cache_meta->input_len);

#if 0
	int layer = 0;// layer is fixed
	int nb = 128;
	int qk = 32;
	int token_rows = 1;
	int token_row_size = (4096*sizeof(block_q8_0)/32);
#else
    //input metadata
    uint32_t inputmetadatabase = weightmetadatabase + sizeof(struct pim_meta) + cache_meta->layer_len * cache_meta->layer_num;
    pim_matrix_des *pinputcache = (pim_matrix_des *) mem_alloc(sizeof(pim_matrix_des));
    mram_read((__mram_ptr void const*) (inputmetadatabase), pinputcache, sizeof(pim_matrix_des));
	int token_rows = pinputcache->ne[1];
    int nb = 128;
	int qk = 32;
    int token_row_size = (4096*sizeof(block_q8_0)/32);
	//printdf("");
#endif

    //weight info: GGML_TYPE_Q4_0 default
    if (cache_meta->weight_type == ((uint16_t)GGML_TYPE_Q4_0)) {
		block_q4_0 *pqlayer0weight = (block_q4_0 *)(weightmetadatabase + sizeof(struct pim_meta));

	    float *psumf = (float *)mem_alloc(sizeof(float)*token_rows*cache_meta->rows_per_dpu);

	    for(int l = 0;l < token_rows;l++) {
		    for(int k = 0;k < cache_meta->rows_per_dpu;k++) {
				block_q4_0 *pqlayer0weight = (block_q4_0 *)(weightmetadatabase + sizeof(struct pim_meta) + cache_meta->size_per_row*k);
				block_q8_0 *pinput = (block_q8_0 *)(weightmetadatabase + cache_meta->input_offset + l*token_row_size);
			    for (int i = 0; i < nb; i++) {
			        int sumi = 0;

			        for (int j = 0; j < qk/2; ++j) {
			            const int v0 = (pqlayer0weight[i].qs[j] & 0x0F) - 8;
			            const int v1 = (pqlayer0weight[i].qs[j] >>   4) - 8;

			            sumi += (v0 * pinput[i].qs[j]) + (v1 * pinput[i].qs[j + qk/2]);
			        }

			        psumf[l*cache_meta->rows_per_dpu + k] += sumi*FP16_TO_FP32(pqlayer0weight[i].d)*FP16_TO_FP32(pinput[i].d);
			    }
		    }
	    }
    }
	
/*
    uint32_t mram_base_addr_weight = (uint32_t) (DPU_MRAM_HEAP_POINTER);
	#define BLOCK_SIZE 2048

	unsigned char *cache_A = (unsigned char* *) mem_alloc(BLOCK_SIZE);

   
	for (unsigned int i = 0; i < BLOCK_SIZE; i ++) {
        mram_read((__mram_ptr void const*) (mram_base_addr_weight), cache_A, BLOCK_SIZE);
	}
*/	
	return 0;
}


