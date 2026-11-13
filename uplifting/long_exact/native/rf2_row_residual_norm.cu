// Keep the original pair residual and emit the contiguous LayerNorm of its
// spatial transpose. Reuse the verified P10 FP32 Welford and affine helpers.
#include "rfd_residual_norm.cu"

namespace {

template <bool SourceMajor>
__global__ void row_residual_norm(const float4* __restrict__ lhs,
                                  const float4* __restrict__ rhs,
                                  const float4* __restrict__ gamma,
                                  const float4* __restrict__ beta,
                                  float4* __restrict__ residual,
                                  float4* __restrict__ normalized,
                                  float* __restrict__ mean_out,
                                  float* __restrict__ inverse_out,
                                  unsigned length, float eps, bool rhs_transposed) {
  const unsigned lane = threadIdx.x % warp;
  const unsigned row = blockIdx.x * rows_per_block + threadIdx.x / warp;
  if (row >= length * length) {
    return;  // Uniform for the whole warp.
  }
  const unsigned swapped = (row % length) * length + row / length;
  const unsigned source = SourceMajor ? row : swapped;
  const unsigned target = SourceMajor ? swapped : row;
  const unsigned source_offset = source * warp + lane;
  const unsigned rhs_offset = (rhs_transposed ? target : source) * warp + lane;
  const float4 a = lhs[source_offset];
  const float4 b = rhs[rhs_offset];
  const float4 value = make_float4(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y),
                                   __fadd_rn(a.z, b.z), __fadd_rn(a.w, b.w));
  residual[source_offset] = value;

  Moments moments = observe<1>({0.0f, 0.0f}, value.x);
  moments = observe<2>(moments, value.y);
  moments = observe<3>(moments, value.z);
  moments = observe<4>(moments, value.w);
  moments = combine<4, 4>(moments, warp_peer<16>(moments));
  moments = combine<8, 8>(moments, warp_peer<8>(moments));
  moments = combine<16, 16>(moments, warp_peer<4>(moments));
  moments = combine<32, 32>(moments, warp_peer<2>(moments));
  moments = combine<64, 64>(moments, warp_peer<1>(moments));
  if (lane == 0) {
    const Moments empty{0.0f, 0.0f};
    moments = combine<128, 0>(moments, empty);
    moments = combine<128, 0>(moments, empty);
  }
  const float mean = __shfl_sync(0xffffffffu, moments.mean, 0);
  const float m2 = __shfl_sync(0xffffffffu, moments.m2, 0);
  const float inverse = rsqrtf(__fadd_rn(__fmul_rn(m2, 1.0f / 128.0f), eps));
  const float4 scale = gamma[lane];
  const float4 shift = beta[lane];
  normalized[target * warp + lane] = make_float4(
      affine(value.x, scale.x, shift.x, mean, inverse),
      affine(value.y, scale.y, shift.y, mean, inverse),
      affine(value.z, scale.z, shift.z, mean, inverse),
      affine(value.w, scale.w, shift.w, mean, inverse));
  if (lane == 0 && mean_out != nullptr) {
    mean_out[target] = mean;
    inverse_out[target] = inverse;
  }
}

}  // namespace

extern "C" int rf2_row_residual_norm_launch(const void* lhs, const void* rhs,
                                             const void* gamma, const void* beta,
                                             void* residual, void* normalized,
                                             void* mean, void* inverse,
                                             uint32_t length, float eps,
                                             int rhs_transposed, int source_major,
                                             void* stream_ptr) {
  if (length < 2 || length > 5792) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto a = static_cast<const float4*>(lhs);
  const auto b = static_cast<const float4*>(rhs);
  const auto g = static_cast<const float4*>(gamma);
  const auto h = static_cast<const float4*>(beta);
  auto r = static_cast<float4*>(residual);
  auto n = static_cast<float4*>(normalized);
  auto m = static_cast<float*>(mean);
  auto s = static_cast<float*>(inverse);
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const unsigned blocks = (length * length + rows_per_block - 1) / rows_per_block;
  if (source_major) {
    row_residual_norm<true><<<blocks, threads, 0, stream>>>(
        a, b, g, h, r, n, m, s, length, eps, rhs_transposed != 0);
  } else {
    row_residual_norm<false><<<blocks, threads, 0, stream>>>(
        a, b, g, h, r, n, m, s, length, eps, rhs_transposed != 0);
  }
  return static_cast<int>(cudaPeekAtLastError());
}
