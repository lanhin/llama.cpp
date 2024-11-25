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
struct pim_meta {
	// 每个类型的tensor在DPU中的metadata
	uint16_t layer_num;
	uint16_t weight_type;
	uint16_t rows_per_dpu;
	uint16_t rest_rows;
	uint32_t size_per_row;
	uint32_t layer_len;
	// 每一层串行计算，dpu的返回值只需一个地址空间
	uint32_t response_offset;
	uint32_t response_len;
	// 每一层串行计算，dpu的输入也只需要一个地址空间
	uint32_t input_offset;
	uint32_t input_len; 
	int64_t ne[GGML_MAX_DIMS];
	size_t	nb[GGML_MAX_DIMS];
};



#define roundup(n, m) ((n / m) * m + m)

__host struct pim_meta DPU_INPUT_ARGUMENTS;
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

    uint32_t metadatabase = (uint32_t) (DPU_MRAM_HEAP_POINTER);
    struct pim_meta *cache_meta = (struct pim_meta *) mem_alloc(sizeof(struct pim_meta));
    mram_read((__mram_ptr void const*) (metadatabase), cache_meta, sizeof(struct pim_meta));

	printf("layer_num: %d, weight_type=%d,rows_per_dpu=%d,size_per_row=%d,rest_rows=%d,response_offset=%d,response_len=%d,input_offset=%d,input_len=%d\n",
		cache_meta->layer_num,cache_meta->weight_type,cache_meta->rows_per_dpu,cache_meta->size_per_row,cache_meta->rest_rows,cache_meta->response_offset,
		cache_meta->response_len,cache_meta->input_offset,cache_meta->input_len);
	
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


