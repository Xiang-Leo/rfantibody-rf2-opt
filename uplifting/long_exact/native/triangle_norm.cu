// Reuse the P8 Welford primitives and expose a separate channel-major launcher.
#define layer_norm_launch triangle_norm_contiguous_reference
#include "layer_norm.cu"
#undef layer_norm_launch

namespace {

constexpr unsigned tile_rows = 32;

template <int Width>
__global__ void triangle_norm_kernel(const float* __restrict__ x,
                                      const float4* __restrict__ gamma,
                                      const float4* __restrict__ beta,
                                      float4* __restrict__ y,
                                      float* __restrict__ mean,
                                      float* __restrict__ rstd,
                                      uint32_t rows, float eps) {
  // Each warp reads consecutive cells from one BMM output channel. Padding
  // lets the normalization groups then read across the channel planes.
  __shared__ float transposed[Width][tile_rows + 1];
  const unsigned first = blockIdx.x * tile_rows;
  for (unsigned index = threadIdx.x; index < Width * tile_rows; index += block_threads) {
    const unsigned channel = index / tile_rows;
    const unsigned local_row = index % tile_rows;
    const unsigned row = first + local_row;
    transposed[channel][local_row] = row < rows ? x[channel * rows + row] : 0.0f;
  }
  __syncthreads();

  constexpr unsigned group = Width / 4;
  constexpr unsigned rows_per_pass = block_threads / group;
  const unsigned lane = threadIdx.x % group;
  const unsigned channel = 4 * lane;
  const float4 scale = gamma[lane];
  const float4 shift = beta[lane];
  for (unsigned local_row = threadIdx.x / group; local_row < tile_rows;
       local_row += rows_per_pass) {
    const float4 value = make_float4(
        transposed[channel][local_row], transposed[channel + 1][local_row],
        transposed[channel + 2][local_row], transposed[channel + 3][local_row]);
    Moments moments = observe<1>({0.0f, 0.0f}, value.x);
    moments = observe<2>(moments, value.y);
    moments = observe<3>(moments, value.z);
    moments = observe<4>(moments, value.w);
    moments = merge<group, 16>(moments);
    moments = merge<group, 8>(moments);
    moments = merge<group, 4>(moments);
    moments = merge<group, 2>(moments);
    moments = merge<group, 1>(moments);
    if (lane == 0) {
      const Moments empty{0.0f, 0.0f};
      moments = combine<Width, 0>(moments, empty);
      moments = combine<Width, 0>(moments, empty);
    }
    const float row_mean = __shfl_sync(0xffffffffu, moments.mean, 0, group);
    const float row_m2 = __shfl_sync(0xffffffffu, moments.m2, 0, group);
    const float inverse = rsqrtf(__fadd_rn(__fmul_rn(row_m2, 1.0f / Width), eps));
    const unsigned row = first + local_row;
    if (row < rows) {
      y[row * group + lane] = make_float4(
          affine(value.x, scale.x, shift.x, row_mean, inverse),
          affine(value.y, scale.y, shift.y, row_mean, inverse),
          affine(value.z, scale.z, shift.z, row_mean, inverse),
          affine(value.w, scale.w, shift.w, row_mean, inverse));
      if (lane == 0 && mean != nullptr) {
        mean[row] = row_mean;
        rstd[row] = inverse;
      }
    }
  }
}

template <int Width>
void launch_triangle(const void* x, const void* gamma, const void* beta,
                     void* y, void* mean, void* rstd, uint32_t rows,
                     float eps, cudaStream_t stream) {
  triangle_norm_kernel<Width><<<(rows + tile_rows - 1) / tile_rows, block_threads, 0, stream>>>(
      static_cast<const float*>(x), static_cast<const float4*>(gamma),
      static_cast<const float4*>(beta), static_cast<float4*>(y), static_cast<float*>(mean),
      static_cast<float*>(rstd), rows, eps);
}

}  // namespace

extern "C" int triangle_norm_launch(const void* x, const void* gamma, const void* beta,
                                    void* y, void* mean, void* rstd, uint32_t rows,
                                    uint32_t width, float eps, void* stream_ptr) {
  if (rows == 0 || (mean == nullptr) != (rstd == nullptr)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  switch (width) {
    case 32: launch_triangle<32>(x, gamma, beta, y, mean, rstd, rows, eps, stream); break;
    case 64: launch_triangle<64>(x, gamma, beta, y, mean, rstd, rows, eps, stream); break;
    case 128: launch_triangle<128>(x, gamma, beta, y, mean, rstd, rows, eps, stream); break;
    default: return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(cudaPeekAtLastError());
}
