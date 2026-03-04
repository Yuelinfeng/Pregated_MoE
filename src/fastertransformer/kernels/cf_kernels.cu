#include "cf_kernels.h"
#include <cstring>

namespace fastertransformer {

// ========================== Kernel 1: Build K-hot matrix ==========================
// expert_ids[num_tokens * k] -> Y_matrix[num_tokens x num_experts]
__global__ void
cf_build_khot_matrix_kernel(const int* expert_ids, float* Y_matrix, int num_tokens, int k, int num_experts)
{
    // Each thread handles one element of expert_ids
    const int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_tokens * k;
    if (idx >= total)
        return;

    const int token_idx  = idx % num_tokens;  // which token
    const int expert_idx = expert_ids[idx];   // which expert was selected

    if (expert_idx >= 0 && expert_idx < num_experts) {
        // Atomic add to handle k > 1 case (multiple experts per token)
        atomicAdd(&Y_matrix[token_idx * num_experts + expert_idx], 1.0f);
    }
}

void cf_build_khot_matrix(
    const int* expert_ids, float* Y_matrix, int num_tokens, int k, int num_experts, cudaStream_t stream)
{
    // Zero out Y_matrix first
    cudaMemsetAsync(Y_matrix, 0, sizeof(float) * num_tokens * num_experts, stream);

    const int total   = num_tokens * k;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;
    cf_build_khot_matrix_kernel<<<blocks, threads, 0, stream>>>(expert_ids, Y_matrix, num_tokens, k, num_experts);
}

// ========================== Kernel 2: EMA Update ==========================
// H[i] = alpha * Y[i] + (1 - alpha) * H[i]
__global__ void cf_ema_update_kernel(float* H_matrix, const float* Y_matrix, float alpha, int total_elements)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements)
        return;

    H_matrix[idx] = alpha * Y_matrix[idx] + (1.0f - alpha) * H_matrix[idx];
}

void cf_ema_update(float* H_matrix, const float* Y_matrix, float alpha, int total_elements, cudaStream_t stream)
{
    const int threads = 256;
    const int blocks  = (total_elements + threads - 1) / threads;
    cf_ema_update_kernel<<<blocks, threads, 0, stream>>>(H_matrix, Y_matrix, alpha, total_elements);
}

// ========================== Kernel 3: Column Sum ==========================
// Reduce P[num_tokens x num_experts] along rows -> score_vec[num_experts]
__global__ void cf_column_sum_kernel(const float* P_matrix, float* score_vec, int num_tokens, int num_experts)
{
    // Each thread handles one expert (column)
    const int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts)
        return;

    float sum = 0.0f;
    for (int t = 0; t < num_tokens; ++t) {
        sum += P_matrix[t * num_experts + expert];
    }
    score_vec[expert] = sum;
}

void cf_column_sum(const float* P_matrix, float* score_vec, int num_tokens, int num_experts, cudaStream_t stream)
{
    const int threads = 256;
    const int blocks  = (num_experts + threads - 1) / threads;
    cf_column_sum_kernel<<<blocks, threads, 0, stream>>>(P_matrix, score_vec, num_tokens, num_experts);
}

}  // namespace fastertransformer

// ========================== Kernel 4: GPU-side Top-K ==========================
// Single-block kernel: loads score_vec into shared memory, finds top-k by repeated argmax
// This avoids D2H copy + cudaStreamSynchronize + CPU sort.
// num_experts is typically ≤ 256, so fits in shared memory easily.
namespace fastertransformer {

__global__ void cf_topk_gpu_kernel(const float* score_vec, int* topk_ids, int num_experts, int topk)
{
    extern __shared__ float smem[];  // [num_experts] scores + [num_experts] mask
    float*                  scores = smem;
    float*                  mask   = smem + num_experts;

    // Load scores into shared memory, initialize mask to 1.0 (active)
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        scores[i] = score_vec[i];
        mask[i]   = 1.0f;
    }
    __syncthreads();

    // Only thread 0 does the sequential top-k selection (num_experts is small ~64-256)
    if (threadIdx.x == 0) {
        for (int t = 0; t < topk; ++t) {
            float best_val = -1e30f;
            int   best_idx = 0;
            for (int e = 0; e < num_experts; ++e) {
                if (mask[e] > 0.5f && scores[e] > best_val) {
                    best_val = scores[e];
                    best_idx = e;
                }
            }
            topk_ids[t]    = best_idx;
            mask[best_idx] = 0.0f;  // mark as selected
        }
    }
}

void cf_topk_gpu(const float* score_vec, int* topk_ids, int num_experts, int topk, cudaStream_t stream)
{
    // Single block, enough threads to load shared memory
    int    threads   = min(256, num_experts);
    size_t smem_size = sizeof(float) * num_experts * 2;  // scores + mask
    cf_topk_gpu_kernel<<<1, threads, smem_size, stream>>>(score_vec, topk_ids, num_experts, topk);
}

// ========================== Kernel 5: Normalize Y^TY and EMA update S ==========================
// For each row of raw_YtY, compute row sum, divide each element by row sum (row-normalize),
// then blend into S via EMA: S[i][j] = beta * S[i][j] + (1 - beta) * normalized_YtY[i][j]
__global__ void cf_normalize_and_update_S_kernel(const float* raw_YtY, float* S_matrix, float beta, int num_experts)
{
    const int row = blockIdx.x;  // one block per row (expert)
    if (row >= num_experts)
        return;

    // Step 1: Compute row sum for normalization
    float row_sum = 0.0f;
    for (int j = threadIdx.x; j < num_experts; j += blockDim.x) {
        row_sum += raw_YtY[row * num_experts + j];
    }

    // Warp-level reduction for row_sum
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        row_sum += __shfl_down_sync(0xFFFFFFFF, row_sum, offset);
    }

    // Use shared memory to collect warp results
    __shared__ float warp_sums[8];  // support up to 256 threads = 8 warps
    int              warp_id = threadIdx.x / warpSize;
    int              lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) {
        warp_sums[warp_id] = row_sum;
    }
    __syncthreads();

    // Thread 0 sums warp results
    if (threadIdx.x == 0) {
        float total     = 0.0f;
        int   num_warps = (blockDim.x + warpSize - 1) / warpSize;
        for (int w = 0; w < num_warps; ++w) {
            total += warp_sums[w];
        }
        warp_sums[0] = total;  // broadcast back
    }
    __syncthreads();

    float total_sum = warp_sums[0];

    // Step 2: Normalize and EMA blend
    if (total_sum > 1e-8f) {
        float inv_sum = 1.0f / total_sum;
        for (int j = threadIdx.x; j < num_experts; j += blockDim.x) {
            float normalized = raw_YtY[row * num_experts + j] * inv_sum;
            int   idx        = row * num_experts + j;
            S_matrix[idx]    = beta * S_matrix[idx] + (1.0f - beta) * normalized;
        }
    }
    // If row_sum ≈ 0, this expert was never activated; leave S row unchanged.
}

void cf_normalize_and_update_S(const float* raw_YtY, float* S_matrix, float beta, int num_experts, cudaStream_t stream)
{
    int threads = min(256, num_experts);
    cf_normalize_and_update_S_kernel<<<num_experts, threads, 0, stream>>>(raw_YtY, S_matrix, beta, num_experts);
}

}  // namespace fastertransformer
