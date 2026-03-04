#pragma once

#include <cuda_runtime.h>

namespace fastertransformer {

// Build K-hot matrix Y from expert indices: Y[token][expert] = 1.0 if expert was selected for that token
// expert_ids: [num_tokens * k], each element is an expert index
// Y_matrix:   [num_tokens x num_experts], output float matrix (zeroed before scatter)
void cf_build_khot_matrix(
    const int* expert_ids, float* Y_matrix, int num_tokens, int k, int num_experts, cudaStream_t stream);

// EMA update: H = alpha * Y + (1 - alpha) * H, element-wise
// H_matrix: [num_tokens x num_experts], in-place updated
// Y_matrix: [num_tokens x num_experts], current routing feedback
void cf_ema_update(float* H_matrix, const float* Y_matrix, float alpha, int total_elements, cudaStream_t stream);

// Column-wise sum: reduce P matrix [num_tokens x num_experts] along batch dim
// Output: score_vec [num_experts]
void cf_column_sum(const float* P_matrix, float* score_vec, int num_tokens, int num_experts, cudaStream_t stream);

// GPU-side Top-K selection from score_vec [num_experts] → topk_ids [topk]
// Avoids D2H copy and cudaStreamSynchronize. Uses simple single-block selection.
// score_vec:  [num_experts] input scores on GPU
// topk_ids:   [topk] output expert IDs on GPU (device memory)
// topk:       number of experts to select
void cf_topk_gpu(const float* score_vec, int* topk_ids, int num_experts, int topk, cudaStream_t stream);

// Online S-matrix update: S = beta * S + (1-beta) * normalize(Y^T * Y)
// Uses cuBLAS for Y^T * Y, then normalizes each row to sum to 1.
// This is a launcher that calls the row-normalization kernel after cuBLAS sgemm.
// raw_YtY:    [num_experts x num_experts] temporary buffer for Y^T * Y result
// S_matrix:   [num_experts x num_experts] in-place updated similarity matrix
// beta:       EMA decay for S update (e.g., 0.95)
void cf_normalize_and_update_S(const float* raw_YtY, float* S_matrix, float beta, int num_experts, cudaStream_t stream);

}  // namespace fastertransformer
