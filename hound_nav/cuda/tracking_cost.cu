// Fused TrackingCost forward for Jetson (BeamNG SimpleCarCost semantics).
// Footprint collision matches IGHAStar check_crop (full BB raster, not 4 tires).
// No temporal causality: one thread per (m,k,t); atomicAdd into per-rollout K.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

namespace {

__device__ __forceinline__ int meters_to_px(
  float meters, float map_size, float map_res, int map_size_px)
{
  int px = static_cast<int>((meters + map_size * 0.5f) / map_res);
  if (px < 0) {
    px = 0;
  }
  if (px > map_size_px - 1) {
    px = map_size_px - 1;
  }
  return px;
}

__device__ __forceinline__ float clampf(float x, float lo, float hi)
{
  return fminf(fmaxf(x, lo), hi);
}

/**
 * IGHAStar check_crop footprint raster on a robot-centered BEV.
 * bev_cost is IGHA free-ness (255 free .. 0 lethal). Per-cell MPC cost is
 * (255 - raw) / 255 (0 free .. 1 lethal) at lookup — map is not pre-inverted.
 * Returns mean of squared cell costs over the whole vehicle patch.
 * Out-of-map cells count as fully lethal (same spirit as planner OOB → invalid).
 */
__device__ float footprint_state_cost(
  float x, float y, float cy, float sy,
  const float * __restrict__ bev_cost,
  int map_size_px, float map_size, float map_res,
  float car_l2, float car_w2)
{
  const float res_inv = 1.0f / map_res;
  int patch_length_px = static_cast<int>(2.0f * car_l2 * res_inv);
  int patch_width_px = static_cast<int>(2.0f * car_w2 * res_inv);
  if (patch_length_px < 1) {
    patch_length_px = 1;
  }
  if (patch_width_px < 1) {
    patch_width_px = 1;
  }

  const float half = map_size * 0.5f;
  float sum_sq = 0.0f;
  int n = 0;

  for (int i = 0; i < patch_length_px; ++i) {
    const float offset_x = static_cast<float>(i) * map_res - car_l2;
    for (int j = 0; j < patch_width_px; ++j) {
      const float offset_y = static_cast<float>(j) * map_res - car_w2;
      // Body → world (same as check_crop_cpu / check_validity_batch_kernel)
      const float wx = offset_x * cy - offset_y * sy + x;
      const float wy = offset_x * sy + offset_y * cy + y;

      float c;
      if (wx < -half || wx >= half || wy < -half || wy >= half) {
        c = 1.0f;
      } else {
        const int ix = meters_to_px(wx, map_size, map_res, map_size_px);
        const int iy = meters_to_px(wy, map_size, map_res, map_size_px);
        const float raw = bev_cost[iy * map_size_px + ix];
        c = (255.0f - raw) * (1.0f / 255.0f);
      }
      sum_sq += c * c;
      ++n;
    }
  }
  return sum_sq / static_cast<float>(n);
}

/**
 * One thread per (m,k,t). Layout tid = ((m*K + k)*T + t).
 * Accumulates mean_M sum_T via atomicAdd(· * inv_M).
 */
__global__ void tracking_cost_kernel(
  const float * __restrict__ state,
  const float * __restrict__ path,
  const float * __restrict__ bev_cost,
  const float * __restrict__ scaling,
  float * __restrict__ out_cost,
  float * __restrict__ out_cons,
  int M, int K, int T, int NX,
  int N,
  int map_size_px,
  float map_size, float map_res,
  float car_l2, float car_w2,
  float pos_w, float heading_w, float speed_w,
  float roll_ditch_w, float lethal_w,
  float critical_RI, float critical_vert_acc,
  float gravity,
  float inv_M)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) {
    return;
  }

  const int t = tid % T;
  const int k = (tid / T) % K;

  const int base = tid * NX;
  const float x = state[base + 0];
  const float y = state[base + 1];
  const float roll = state[base + 3];
  const float pitch = state[base + 4];
  const float yaw = state[base + 5];
  const float vx = state[base + 6];
  const float vy = state[base + 7];
  const float ay = state[base + 10];
  const float az = state[base + 11];

  float sy, cy;
  sincosf(yaw, &sy, &cy);

  const float V = sqrtf(vx * vx + vy * vy) * ((vx >= 0.0f) ? 1.0f : -1.0f);
  const float beta = atan2f(vy, vx);
  const float beta2_sq = beta * beta;

  const float state_cost = footprint_state_cost(
    x, y, cy, sy, bev_cost,
    map_size_px, map_size, map_res, car_l2, car_w2);

  float sr, cr;
  sincosf(roll, &sr, &cr);
  float sp, cp;
  sincosf(pitch, &sp, &cp);
  (void)sr;
  (void)sp;

  const float ri = (fabsf(az) > 1.0e-3f) ? fabsf(ay / az) : 1.0e3f;
  const float roll_ditch =
    (clampf(fabsf(az - gravity * cr * cp) - critical_vert_acc, 0.0f, 10.0f) / 10.0f
     + clampf(ri - critical_RI, 0.0f, 1.0f))
    * roll_ditch_w;

  float constraint = lethal_w * state_cost;

  const float path_x = path[t * 4 + 0];
  const float path_y = path[t * 4 + 1];
  const float path_yaw = path[t * 4 + 2];
  const float path_vel = path[t * 4 + 3];
  float path_sy, path_cy;
  sincosf(path_yaw, &path_sy, &path_cy);

  const float x_err = x - path_x;
  const float y_err = y - path_y;
  const float cy_err = cy - path_cy;
  const float sy_err = sy - path_sy;
  const float yaw_err = cy_err * cy_err + sy_err * sy_err;
  const float vel_err = (V - path_vel) * (V - path_vel);
  const float pos_err = x_err * x_err + y_err * y_err;

  float running =
    pos_w * pos_err + heading_w * yaw_err + speed_w * vel_err + beta2_sq * 1.5f;
  running = running * scaling[t] + roll_ditch;

  if (pos_err < 1.0f) {
    constraint = 0.0f;
  }

  atomicAdd(out_cost + k, (running + constraint) * inv_M);
  atomicAdd(out_cons + k, constraint * inv_M);
}

}  // namespace

void tracking_cost_launcher(
  torch::Tensor state,
  torch::Tensor path,
  torch::Tensor bev_cost,
  torch::Tensor scaling,
  torch::Tensor out_cost,
  torch::Tensor out_cons,
  int M, int K, int T, int NX,
  int map_size_px,
  float map_size, float map_res,
  float car_l2, float car_w2,
  float pos_w, float heading_w, float speed_w,
  float roll_ditch_w, float lethal_w,
  float critical_RI, float critical_vert_acc,
  float gravity,
  int block_dim, int /*grid_dim_unused*/)
{
  TORCH_CHECK(state.is_cuda() && state.scalar_type() == torch::kFloat32, "state");
  TORCH_CHECK(path.is_cuda() && path.scalar_type() == torch::kFloat32, "path");
  TORCH_CHECK(bev_cost.is_cuda() && bev_cost.scalar_type() == torch::kFloat32, "bev");
  TORCH_CHECK(scaling.is_cuda() && scaling.scalar_type() == torch::kFloat32, "scaling");
  TORCH_CHECK(out_cost.is_cuda() && out_cost.scalar_type() == torch::kFloat32, "out");
  TORCH_CHECK(state.is_contiguous() && path.is_contiguous() && bev_cost.is_contiguous());
  TORCH_CHECK(scaling.is_contiguous() && out_cost.is_contiguous() && out_cons.is_contiguous());

  out_cost.zero_();
  out_cons.zero_();

  const int N = M * K * T;
  const int grid = (N + block_dim - 1) / block_dim;
  const float inv_M = 1.0f / static_cast<float>(M);

  tracking_cost_kernel<<<grid, block_dim>>>(
    state.data_ptr<float>(),
    path.data_ptr<float>(),
    bev_cost.data_ptr<float>(),
    scaling.data_ptr<float>(),
    out_cost.data_ptr<float>(),
    out_cons.data_ptr<float>(),
    M, K, T, NX, N,
    map_size_px, map_size, map_res,
    car_l2, car_w2,
    pos_w, heading_w, speed_w,
    roll_ditch_w, lethal_w,
    critical_RI, critical_vert_acc,
    gravity,
    inv_M);
}
