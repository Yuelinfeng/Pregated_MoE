#include "fetcher.h"

#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "src/fastertransformer/kernels/cf_kernels.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/profiling.h"
#include "src/fastertransformer/utils/random.h"
#include <algorithm>
#include <chrono>
#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <numeric>
#include <sstream>

namespace fastertransformer {

// the linker asks me to do so

template class FetcherContext<float>;
template class FetcherContext<half>;

template class FetcherContext<float, cutlass::fp4_t>;
template class FetcherContext<float, cutlass::nf4_t>;
template class FetcherContext<float, cutlass::uint4b_t>;
template class FetcherContext<float, cutlass::int4b_t>;

template class FetcherContext<half, cutlass::fp4_t>;
template class FetcherContext<half, cutlass::nf4_t>;
template class FetcherContext<half, cutlass::uint4b_t>;
template class FetcherContext<half, cutlass::int4b_t>;

template class FetcherContext<float, uint8_t>;
template class FetcherContext<half, uint8_t>;

#ifdef ENABLE_BF16
template class FetcherContext<__nv_bfloat16>;

template class FetcherContext<float, __nv_bfloat16>;
template class FetcherContext<half, __nv_bfloat16>;

template class FetcherContext<__nv_bfloat16, float>;
template class FetcherContext<__nv_bfloat16, half>;

template class FetcherContext<__nv_bfloat16, cutlass::fp4_t>;
template class FetcherContext<__nv_bfloat16, cutlass::nf4_t>;
template class FetcherContext<__nv_bfloat16, cutlass::uint4b_t>;
template class FetcherContext<__nv_bfloat16, cutlass::int4b_t>;

template class FetcherContext<__nv_bfloat16, uint8_t>;
#endif

int64_t calc_sparse_time             = 0;  // microseconds
int64_t cpy_expert_array_to_cpu_time = 0;
int64_t total_row_cpy                = 0;
int64_t layer_1_fetch_time           = 0;

// 1. copy to expert_for_source_row_fetching
// 2. calc expert_sparse_idx_working
// 3. launch fetch on the stream, from source to working

// fetch(prefetch=false) (按需拉取)： 把当前的参数抓到 GPU 工作区。
// fetch(prefetch=true) (层级预取)： 把未来的参数提前放到 GPU 工作区。
template<class ActT, class WeightT, class BiasT>
void FetcherContext<ActT, WeightT, BiasT>::fetch(const int* permuted_experts, bool prefetch)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (last_time && prefetch) {
        FT_LOG_TRACE("Abandon prefetching at final layer");
        return;
    }

    check_cuda_error(cudaMemcpy(permuted_experts_, permuted_experts, sizeof(int) * num_rows_, cudaMemcpyDeviceToHost));

    auto new_end        = std::unique(permuted_experts_, permuted_experts_ + num_rows_);
    num_active_experts_ = new_end - permuted_experts_;

    if (GlobalConfig::instance().profiling) {
        Profiling::instance().activeExperts(num_active_experts_);
    }

    if (GlobalConfig::instance().profiling) {
        Profiling::instance().insert(stream, EventType::MEM_START);
    }

    /*
    当 fetch_all 被设为 1 （即 true）时，fetcher 本不应拉取所有专家，
    但现在会无视当前 token 们发出的路由需求（num_active_experts_），
    强行将需要激活拉取的专家数量放大到所有专家总数（num_experts_），
    以强行实施全体覆盖
    */
    bool fetch_all          = GlobalConfig::instance().fetch_all;
    int  forced_num_experts = GlobalConfig::instance().forced_num_experts;
    num_active_experts_     = forced_num_experts ? forced_num_experts : num_active_experts_;

    // === CF-MoE: 走 CF 专属预测路径 ===
    if (this->mode == FetchType::CF_PREFETCH) {
        cf_predict_and_fetch(prefetch);
        return;
    }

    // 决定当前循环应该处理几个专家，fetch_all为true则强行赋值为专家总数
    int _active_experts_count = fetch_all ? num_experts_ : num_active_experts_;

    static constexpr bool scales_required =
        std::is_same<WeightT, uint8_t>::value || std::is_same<WeightT, cutlass::uint4b_t>::value
        || std::is_same<WeightT, cutlass::fp4_t>::value || std::is_same<WeightT, cutlass::nf4_t>::value;

    for (int i = 0; i < _active_experts_count; i++) {
        /*
        如果 forced_num_experts 或 fetch_all 为真，则直接使用循环变量 i 作为专家索引（强制从 0 连续取到
        _active_experts_count-1）； 否则，使用 permuted_experts_[i] 作为专家索引（按照当前 token 的真实路由结果来取）。
        */
        int expert = (forced_num_experts || fetch_all) ? i : permuted_experts_[i];

        const char* fetch_weight_src = prefetch ? next_weight_src_ : current_weight_src_;
        std::string layer_name       = prefetch ? next_layer_name_ : current_layer_name_;

        if (scales_required) {
            futures_.push_back(GroupedMemoryArena::instance().allocate(
                layer_name + "expert" + std::to_string(expert),
                {reinterpret_cast<char*>(intermediate_working_) + i * intermediate_w_size_per_expert_,
                 reinterpret_cast<char*>(output_working_) + i * output_w_size_per_expert_,
                 reinterpret_cast<char*>(intermediate_scale_working_) + i * intermediate_scale_size_per_expert_,
                 reinterpret_cast<char*>(output_scale_working_) + i * output_scale_size_per_expert_},
                fetch_weight_src + expert * weight_size_per_expert_));
        }
        else {
            futures_.push_back(GroupedMemoryArena::instance().allocate(
                layer_name + "expert" + std::to_string(expert),
                {reinterpret_cast<char*>(intermediate_working_) + i * intermediate_w_size_per_expert_,
                 reinterpret_cast<char*>(output_working_) + i * output_w_size_per_expert_},
                fetch_weight_src + expert * weight_size_per_expert_));
        }
    }
}

int64_t fetcher_sync_wait_time = 0;  // microseconds

template<class ActT, class WeightT, class BiasT>
void FetcherContext<ActT, WeightT, BiasT>::sync()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    for (auto& future : futures_) {
        future.wait();
    }
    if (GlobalConfig::instance().profiling) {
        Profiling::instance().insert(stream, EventType::MEM_END);
    }
    futures_.clear();
    check_cuda_error(cudaStreamSynchronize(stream));

    // update dst from working (swap them)
    std::swap(intermediate_dst_, intermediate_working_);
    std::swap(output_dst_, output_working_);
    std::swap(intermediate_bias_dst_, intermediate_bias_working_);
    std::swap(intermediate_scale_dst_, intermediate_scale_working_);
    std::swap(output_scale_dst_, output_scale_working_);
}

// called in FfnLayer.cc
//
template<class ActT, class WeightT, class BiasT>
void FetcherContext<ActT, WeightT, BiasT>::set_source(const char* next_weight_src, const char* current_weight_src)
{
    next_weight_src_    = next_weight_src;
    current_weight_src_ = current_weight_src;
}

template<class ActT, class WeightT, class BiasT>
void FetcherContext<ActT, WeightT, BiasT>::set_layer(const std::string& next_layer_name,
                                                     const std::string& current_layer_name,
                                                     bool               is_first_moe,
                                                     bool               is_last_moe)
{
    next_layer_name_    = next_layer_name;
    current_layer_name_ = current_layer_name;
    first_time          = is_first_moe;
    last_time           = is_last_moe;
}

template<class ActT, class WeightT, class BiasT>
void FetcherContext<ActT, WeightT, BiasT>::get_weights(int&            num_active_experts,
                                                       const WeightT*& fc1_expert_weights,
                                                       const WeightT*& fc2_expert_weights,
                                                       const BiasT*&   fc1_expert_biases,
                                                       const ActT*&    fc1_scales,
                                                       const ActT*&    fc2_scales) const
{
    num_active_experts = num_active_experts_;
    fc1_expert_weights = intermediate_dst_;
    fc2_expert_weights = output_dst_;
    fc1_expert_biases  = intermediate_bias_dst_;
    if (scales_required) {
        fc1_scales = intermediate_scale_dst_;
        fc2_scales = output_scale_dst_;
    }
}

int64_t expert_for_row_backup_time = 0;  // microseconds

template<class ActT, class WeightT, class BiasT>
FetcherContext<ActT, WeightT, BiasT>::~FetcherContext()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_LOG_TRACE("futures left: %d", futures_.size());
    freeBuffer();
    check_cuda_error(cudaStreamDestroy(stream));
}

template<class ActT, class WeightT, class BiasT>
FetcherContext<ActT, WeightT, BiasT>::FetcherContext(FetchType mode,
                                                     int       num_experts,
                                                     size_t    intermediate_w_size_per_expert,
                                                     size_t    output_w_size_per_expert,
                                                     size_t    intermediate_b_size_per_expert,
                                                     size_t    intermediate_scale_size_per_expert,
                                                     size_t    output_scale_size_per_expert,
                                                     size_t    arena_size):
    mode(mode),
    first_time(true),
    num_experts_(num_experts),
    intermediate_w_size_per_expert_(cutlass::get_real_size<WeightT>(intermediate_w_size_per_expert)),
    output_w_size_per_expert_(cutlass::get_real_size<WeightT>(output_w_size_per_expert)),
    intermediate_b_size_per_expert_(cutlass::get_real_size<BiasT>(intermediate_b_size_per_expert)),
    intermediate_scale_size_per_expert_(cutlass::get_real_size<ActT>(intermediate_scale_size_per_expert)),
    output_scale_size_per_expert_(cutlass::get_real_size<ActT>(output_scale_size_per_expert)),
    is_allocate_buffer_(false)
{
    // create cuda stream
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    check_cuda_error(cudaStreamCreate(&this->stream));
    weight_size_per_expert_ = intermediate_w_size_per_expert_ + output_w_size_per_expert_
                              + intermediate_scale_size_per_expert_ + output_scale_size_per_expert_;
    if (scales_required) {
        GroupedMemoryArena::instance().initIfUninit(arena_size,
                                                    {intermediate_w_size_per_expert_,
                                                     output_w_size_per_expert_,
                                                     intermediate_scale_size_per_expert_,
                                                     output_scale_size_per_expert_},
                                                    stream);
    }
    else {
        GroupedMemoryArena::instance().initIfUninit(
            arena_size, {intermediate_w_size_per_expert_, output_w_size_per_expert_}, stream);
    }
    Profiling::instance().reset();
}

template<class ActT, class WeightT, class BiasT>
void FetcherContext<ActT, WeightT, BiasT>::allocateBuffer(IAllocator* allocator, size_t num_rows)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_) {
        return;
    }

    allocator_ = allocator;
    num_rows_  = num_rows;

    // TODO: refactor with reMalloc
    intermediate_dst_ =
        (WeightT*)allocator_->reMalloc(intermediate_dst_, intermediate_w_size_per_expert_ * num_experts_);
    output_dst_ = (WeightT*)allocator_->reMalloc(output_dst_, output_w_size_per_expert_ * num_experts_);
    intermediate_bias_dst_ =
        (BiasT*)allocator_->reMalloc(intermediate_bias_dst_, intermediate_b_size_per_expert_ * num_experts_);
    intermediate_working_ =
        (WeightT*)allocator_->reMalloc(intermediate_working_, intermediate_w_size_per_expert_ * num_experts_);
    output_working_ = (WeightT*)allocator_->reMalloc(output_working_, output_w_size_per_expert_ * num_experts_);
    intermediate_bias_working_ =
        (BiasT*)allocator_->reMalloc(intermediate_bias_working_, intermediate_b_size_per_expert_ * num_experts_);
    if (scales_required) {
        intermediate_scale_dst_ =
            (ActT*)allocator_->reMalloc(intermediate_scale_dst_, intermediate_scale_size_per_expert_ * num_experts_);
        output_scale_dst_ =
            (ActT*)allocator_->reMalloc(output_scale_dst_, output_scale_size_per_expert_ * num_experts_);
        intermediate_scale_working_ = (ActT*)allocator_->reMalloc(intermediate_scale_working_,
                                                                  intermediate_scale_size_per_expert_ * num_experts_);
        output_scale_working_ =
            (ActT*)allocator_->reMalloc(output_scale_working_, output_scale_size_per_expert_ * num_experts_);
    }

    permuted_experts_ = (int*)allocator_->reMalloc(permuted_experts_, sizeof(int) * num_rows, false, true);

    // === CF-MoE: 分配额外的 CF 状态缓冲区 ===
    if (mode == FetchType::CF_PREFETCH && !cf_initialized_) {
        cf_alpha_ = GlobalConfig::instance().cf_alpha;
        cf_topk_  = GlobalConfig::instance().cf_topk;
        if (cf_topk_ <= 0)
            cf_topk_ = num_experts_;  // 默认预取所有

        size_t HPS_size = sizeof(float) * num_rows * num_experts_;
        size_t S_size   = sizeof(float) * num_experts_ * num_experts_;
        size_t vec_size = sizeof(float) * num_experts_;

        cf_H_matrix_   = (float*)allocator_->reMalloc(cf_H_matrix_, HPS_size);
        cf_S_matrix_   = (float*)allocator_->reMalloc(cf_S_matrix_, S_size);
        cf_P_matrix_   = (float*)allocator_->reMalloc(cf_P_matrix_, HPS_size);
        cf_Y_matrix_   = (float*)allocator_->reMalloc(cf_Y_matrix_, HPS_size);
        cf_score_vec_  = (float*)allocator_->reMalloc(cf_score_vec_, vec_size);
        cf_YtY_buffer_ = (float*)allocator_->reMalloc(cf_YtY_buffer_, S_size);

        // GPU-side top-k output buffer
        check_cuda_error(cudaMalloc(&cf_predicted_ids_gpu_, sizeof(int) * cf_topk_));

        // CPU-side buffers
        cf_score_vec_cpu_ = new float[num_experts_];
        cf_predicted_ids_ = new int[cf_topk_];

        // Initialize H to zeros, S to identity matrix
        check_cuda_error(cudaMemset(cf_H_matrix_, 0, HPS_size));
        check_cuda_error(cudaMemset(cf_P_matrix_, 0, HPS_size));
        check_cuda_error(cudaMemset(cf_Y_matrix_, 0, HPS_size));

        // Initialize S as identity: S[i][i] = 1.0
        std::vector<float> S_init(num_experts_ * num_experts_, 0.0f);
        for (size_t i = 0; i < num_experts_; ++i) {
            S_init[i * num_experts_ + i] = 1.0f;
        }
        check_cuda_error(cudaMemcpy(cf_S_matrix_, S_init.data(), S_size, cudaMemcpyHostToDevice));

        // Create cuBLAS handle
        cublasCreate(&cf_cublas_handle_);
        cublasSetStream(cf_cublas_handle_, stream);

        cf_initialized_ = true;
        FT_LOG_INFO("CF-MoE initialized: alpha=%.2f, topk=%d, num_experts=%d, num_rows=%d",
                    cf_alpha_,
                    cf_topk_,
                    (int)num_experts_,
                    (int)num_rows_);
    }

    is_allocate_buffer_ = true;

    if (GlobalConfig::instance().profiling) {
        Profiling::instance().recordMemoryUsage();
    }
}

template<class ActT, class WeightT, class BiasT>
void FetcherContext<ActT, WeightT, BiasT>::freeBuffer()
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (is_allocate_buffer_) {
        allocator_->free((void**)&intermediate_dst_);
        allocator_->free((void**)&output_dst_);
        allocator_->free((void**)&intermediate_bias_dst_);
        allocator_->free((void**)&intermediate_working_);
        allocator_->free((void**)&output_working_);
        allocator_->free((void**)&intermediate_bias_working_);
        if (scales_required) {
            allocator_->free((void**)&intermediate_scale_dst_);
            allocator_->free((void**)&output_scale_dst_);
            allocator_->free((void**)&intermediate_scale_working_);
            allocator_->free((void**)&output_scale_working_);
        }

        allocator_->free((void**)&permuted_experts_, true);

        // === CF-MoE: 释放 CF 缓冲区 ===
        if (cf_initialized_) {
            allocator_->free((void**)&cf_H_matrix_);
            allocator_->free((void**)&cf_S_matrix_);
            allocator_->free((void**)&cf_P_matrix_);
            allocator_->free((void**)&cf_Y_matrix_);
            allocator_->free((void**)&cf_score_vec_);
            allocator_->free((void**)&cf_YtY_buffer_);
            if (cf_predicted_ids_gpu_) {
                cudaFree(cf_predicted_ids_gpu_);
                cf_predicted_ids_gpu_ = nullptr;
            }
            delete[] cf_score_vec_cpu_;
            cf_score_vec_cpu_ = nullptr;
            delete[] cf_predicted_ids_;
            cf_predicted_ids_ = nullptr;
            if (cf_cublas_handle_) {
                cublasDestroy(cf_cublas_handle_);
                cf_cublas_handle_ = nullptr;
            }
            cf_initialized_ = false;
        }

        is_allocate_buffer_ = false;
    }
}

}  // namespace fastertransformer
// === CF-MoE Implementation ===
namespace fastertransformer {

// cf_predict_and_fetch: 执行 P = H * S 预测, 提取 Top-K, 发起 DMA 搬运
template<class ActT, class WeightT, class BiasT>
void FetcherContext<ActT, WeightT, BiasT>::cf_predict_and_fetch(bool prefetch)
{
    FT_LOG_DEBUG("CF-MoE: cf_predict_and_fetch (prefetch=%d)", prefetch);
    if (!cf_initialized_) {
        FT_LOG_WARNING("CF-MoE: not initialized, falling back to permuted_experts fetch");
        return;
    }

    // Step 1: P = H * S via cuBLAS sgemm
    // H: [num_rows_ x num_experts_], S: [num_experts_ x num_experts_], P: [num_rows_ x num_experts_]
    // cuBLAS is column-major, so we compute P^T = S^T * H^T
    // which is equivalent to: C(num_experts_ x num_rows_) = A(num_experts_ x num_experts_) * B(num_experts_ x
    // num_rows_)
    const float alpha_blas = 1.0f;
    const float beta_blas  = 0.0f;
    cublasSgemm(cf_cublas_handle_,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                num_experts_,  // m
                num_rows_,     // n
                num_experts_,  // k
                &alpha_blas,
                cf_S_matrix_,  // A: [num_experts_ x num_experts_] col-major = S^T
                num_experts_,  // lda
                cf_H_matrix_,  // B: [num_experts_ x num_rows_] col-major = H^T
                num_experts_,  // ldb
                &beta_blas,
                cf_P_matrix_,   // C: [num_experts_ x num_rows_] col-major = P^T
                num_experts_);  // ldc

    // Step 2: Column sum of P along batch dimension -> score_vec [num_experts_]
    cf_column_sum(cf_P_matrix_, cf_score_vec_, num_rows_, num_experts_, stream);

    // Step 3: GPU-side Top-K selection (P1 fix: no cudaStreamSynchronize!)
    int actual_topk = std::min(cf_topk_, (int)num_experts_);
    cf_topk_gpu(cf_score_vec_, cf_predicted_ids_gpu_, num_experts_, actual_topk, stream);

    // Copy only the topk IDs (tiny: e.g. 2*4=8 bytes) from GPU to CPU
    check_cuda_error(cudaMemcpyAsync(
        cf_predicted_ids_, cf_predicted_ids_gpu_, sizeof(int) * actual_topk, cudaMemcpyDeviceToHost, stream));
    check_cuda_error(cudaStreamSynchronize(stream));  // Sync only for the tiny topk ID copy

    num_active_experts_ = actual_topk;

    FT_LOG_DEBUG("CF-MoE: predicted top-%d experts (GPU topk)", actual_topk);

    // NOTE: MEM_START profiling event is already inserted by the parent fetch() method

    // Step 4: Dispatch DMA transfers for predicted experts
    static constexpr bool scales_required =
        std::is_same<WeightT, uint8_t>::value || std::is_same<WeightT, cutlass::uint4b_t>::value
        || std::is_same<WeightT, cutlass::fp4_t>::value || std::is_same<WeightT, cutlass::nf4_t>::value;

    for (int i = 0; i < actual_topk; ++i) {
        int         expert           = cf_predicted_ids_[i];
        const char* fetch_weight_src = prefetch ? next_weight_src_ : current_weight_src_;
        std::string layer_name       = prefetch ? next_layer_name_ : current_layer_name_;

        if (scales_required) {
            futures_.push_back(GroupedMemoryArena::instance().allocate(
                layer_name + "expert" + std::to_string(expert),
                {reinterpret_cast<char*>(intermediate_working_) + i * intermediate_w_size_per_expert_,
                 reinterpret_cast<char*>(output_working_) + i * output_w_size_per_expert_,
                 reinterpret_cast<char*>(intermediate_scale_working_) + i * intermediate_scale_size_per_expert_,
                 reinterpret_cast<char*>(output_scale_working_) + i * output_scale_size_per_expert_},
                fetch_weight_src + expert * weight_size_per_expert_));
        }
        else {
            futures_.push_back(GroupedMemoryArena::instance().allocate(
                layer_name + "expert" + std::to_string(expert),
                {reinterpret_cast<char*>(intermediate_working_) + i * intermediate_w_size_per_expert_,
                 reinterpret_cast<char*>(output_working_) + i * output_w_size_per_expert_},
                fetch_weight_src + expert * weight_size_per_expert_));
        }
    }
}

// update_history: 接收真实路由结果, 构建 K-hot 矩阵 Y, 执行 EMA 更新 H
template<class ActT, class WeightT, class BiasT>
void FetcherContext<ActT, WeightT, BiasT>::update_history(const int*   real_expert_ids,
                                                          int          num_tokens,
                                                          int          k,
                                                          cudaStream_t ext_stream)
{
    FT_LOG_DEBUG("CF-MoE: update_history (num_tokens=%d, k=%d)", num_tokens, k);
    if (!cf_initialized_)
        return;

    // Build K-hot matrix Y from real_expert_ids
    cf_build_khot_matrix(real_expert_ids, cf_Y_matrix_, num_tokens, k, num_experts_, ext_stream);

    // EMA update: H = alpha * Y + (1 - alpha) * H
    int total_elements = num_tokens * num_experts_;
    cf_ema_update(cf_H_matrix_, cf_Y_matrix_, cf_alpha_, total_elements, ext_stream);

    // === P2 fix: Online S-matrix update via Y^T * Y ===
    // Compute raw_YtY = Y^T * Y: [E x N] * [N x E] = [E x E]
    // Using cuBLAS column-major: C = A^T * B where A=Y, B=Y
    const float alpha_s = 1.0f;
    const float beta_s  = 0.0f;
    cublasSgemm(cf_cublas_handle_,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                num_experts_,  // m
                num_experts_,  // n
                num_tokens,    // k
                &alpha_s,
                cf_Y_matrix_,  // A: [num_tokens x num_experts_] row-major = [num_experts_ x num_tokens] col-major
                num_experts_,  // lda (leading dim of A in col-major storage)
                cf_Y_matrix_,  // B: same as A
                num_experts_,  // ldb
                &beta_s,
                cf_YtY_buffer_,  // C: [num_experts_ x num_experts_]
                num_experts_);   // ldc

    // Row-normalize YtY and EMA blend into S: S = 0.95*S + 0.05*normalize(YtY)
    cf_normalize_and_update_S(cf_YtY_buffer_, cf_S_matrix_, 0.95f, num_experts_, ext_stream);
}

}  // namespace fastertransformer
