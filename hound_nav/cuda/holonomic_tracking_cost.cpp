#include <torch/extension.h>

void holonomic_tracking_cost_launcher(
  torch::Tensor state, torch::Tensor path, torch::Tensor bev_cost, torch::Tensor scaling,
  torch::Tensor out_cost, torch::Tensor out_cons,
  int M, int K, int T, int NX, int path_dim, int map_size_px,
  float map_size, float map_res, float car_l2, float car_w2,
  float pos_w, float heading_w, float speed_w, float vy_w, float omega_w,
  float roll_ditch_w, float lethal_w, float critical_RI, float critical_vert_acc,
  float gravity, int block_dim, int grid_dim);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tracking_cost", &holonomic_tracking_cost_launcher, "holonomic tracking cost");
}
