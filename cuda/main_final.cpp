#define _USE_MATH_DEFINES

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>
#include <cuda.h>

namespace grid {

#define EPS 1e-7

template <typename T>
class Point {
 public:
  T x, y, z;

  Point(T xc, T yc, T zc) : x(xc), y(yc), z(zc) {}
  Point(const Point& other) : x(other.x), y(other.y), z(other.z) {}

  // Pointwise operators for matrix & matrix
  Point operator+(const Point& other) const {
    return Point(x + other.x, y + other.y, z + other.z);
  }
  Point operator-(const Point& other) const {
    return Point(x - other.x, y - other.y, z - other.z);
  }
  Point operator*(const Point& other) const {
    return Point(x * other.x, y * other.y, z * other.z);
  }
  Point operator/(const Point& other) const {
    if (std::fabs(other.x) < EPS || std::fabs(other.y) < EPS ||
        std::fabs(other.z) < EPS) {
      throw "Attempt to divide Point by Point with 0 coordinate";
    }
    return Point(x / other.x, y / other.y, z / other.z);
  }
  
  // Pointwise operators for point & number
  Point operator+(T num) const {
    return Point(x + num, y + num, z + num);
  }
  Point operator-(T num) const {
    return Point(x - num, y - num, z - num);
  }
  Point operator*(T multiplier) const {
    return Point(x * multiplier, y * multiplier, z * multiplier);
  }
  Point operator/(T divider) const {
    if (std::fabs(divider) < EPS) {
      throw "Attempt to divide Point by 0";
    }
    return Point(x / divider, y / divider, z / divider);
  }
  
  bool operator==(const Point& other) const {
    return std::fabs(x - other.x) < EPS && std::fabs(y - other.y) < EPS &&
           std::fabs(z - other.z) < EPS;
  }
  bool operator!=(const Point& other) const {
    return std::fabs(x - other.x) > EPS || std::fabs(y - other.y) > EPS ||
           std::fabs(z - other.z) > EPS;
  }

  T Min() const {
    if (x < y && x < z) {
      return x;
    }
    if (y < x && y < z) {
      return y;
    }
    return z;
  }

  std::string DebugString() const {
    std::stringstream ss;
    ss << "Point {x: " << x << ", y: " << y << ", z: " << z << "}";
    return ss.str();
  }

  void Print() const {
    std::cout << DebugString() << "\n";
  }
};

// 3 dimensional grid (x, y, z) with corresponding indices (i, j, k). Area of
// coverage is (0, xL] x (0, yL] x (0, zL].
class Grid3D {
 public:
  Grid3D(Point<double> p0, Point<double> pN, Point<int> n,
         Point<double> delta)
    : data_size_(n.x * n.y * n.z), num_points_(n), starts_(0, 0, 0),
      edge_sizes_(n.y * n.z, n.x * n.z, n.x * n.y), p0_(p0), pN_(pN),
      delta_(delta) {
    cudaMalloc(&data_, data_size_ * sizeof(double));
    

    starts_.y = edge_sizes_.x;
    starts_.z = starts_.y + edge_sizes_.y;
    
    cudaMalloc(&left_, (starts_.z + edge_sizes_.z) * sizeof(double));
    
    cudaMalloc(&right_, (starts_.z + edge_sizes_.z) * sizeof(double));
    
  }

  ~Grid3D() {
    cudaFree(data_);
    cudaFree(left_);
    cudaFree(right_);
  }

  /*
  double at(int i, int j, int k) const;
  double& set(int i, int j, int k);

  Grid3D operator+(const Grid3D& other) const;
  void operator+=(const Grid3D& other);

  Grid3D operator-(const Grid3D& other) const;
  void operator-=(const Grid3D& other);

  Grid3D operator*(double multiplier) const;
  void operator*=(double multiplier);

  void PrintGrid() const;
  void PrintDelta() const;
  std::string DebugString() const;
  
  double Sum() const;
  double Max() const;
  void ApplyAbs();
  */

  void SetData(const std::vector<double>& new_data);

  Point<double> PointFromIndices(int i, int j, int k);
  int LinearIndex(int i, int j, int k) const;

  double* data() { return data_; }
  double* left() { return left_; }
  double* right() { return right_; }

  int data_size() const { return data_size_; }
  Point<int> size() const { return num_points_; }
  Point<int> starts() const { return starts_; }
  Point<int> edge_sizes() const { return edge_sizes_; }

  Point<double> p0() const { return p0_; }
  Point<double> pN() const { return pN_; }
  Point<double> delta() const { return delta_; }

 private:
  //std::string CompareGridMetaData(const Grid3D& other) const;

  // actual grid
  double* data_;
  int data_size_;
  Point<int> num_points_; // number of points along each axis

  // edges from neighbour processors
  double* left_;
  double* right_;
  Point<int> starts_;
  Point<int> edge_sizes_;

  // meta
  Point<double> p0_; // start borders of intervals
  Point<double> pN_; // end borders of intervals
  Point<double> delta_; // deltas between grid points
};

} // namespace grid

namespace cuda_buffers {

struct CudaInfo {
  size_t data_size;
  int size_x, size_y, size_z;
  double delta_x, delta_y, delta_z;
  double tau;

  CudaInfo(size_t ds, const grid::Point<int>& s, const grid::Point<double>& d,
           double t, bool left, bool right)
    : data_size(ds), size_x(s.x), size_y(s.y), size_z(s.z),
      delta_x(d.x), delta_y(d.y), delta_z(d.z), tau(t) {}
};

CudaInfo* cuda_info_;

bool* are_edges_;

double* left_;
double* right_;

void AllocVariables(grid::Grid3D& in, double tau, bool are_edges[6]) {
  CudaInfo info(in.data_size(), in.size(), in.delta(), tau, are_edges[4],
                are_edges[5]);

  cudaMalloc(&cuda_info_, sizeof(info));
  cudaMemcpy(
    cuda_info_, &info, sizeof(info), cudaMemcpyHostToDevice);
  
  cudaMalloc(&are_edges_, 6 * sizeof(bool));
  cudaMemcpy(are_edges_, &are_edges, 6 * sizeof(bool), cudaMemcpyHostToDevice);

  int edge_full_size = in.edge_sizes().x + in.edge_sizes().y + in.edge_sizes().z;
  cudaMalloc(&left_, edge_full_size * sizeof(double));
  cudaMalloc(&right_, edge_full_size * sizeof(double));
}

void FreeVariables() {
  cudaFree(cuda_info_);
  cudaFree(are_edges_);
}

} // namespace cuda_buffers

namespace cuda_kernels {

// calculates abs(in_data - out_data) pointwise and writes to out_data
__global__ void error_kernel(double* in_data, double* out_data,
                             cuda_buffers::CudaInfo* info) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int idx = x;
  if (y > idx) idx = y;
  if (z > idx) idx = z;
  if (idx >= info->data_size) {
    return;
  }

  double v1 = in_data[idx];
  double v2 = out_data[idx];
  double value;
  if (v1 > v2) {
    value = v1 - v2;
  } else {
    value = v2 - v1;
  }
  out_data[idx] = value;
}

} // namespace cuda_kernels

/*
double CalculateError(const grid::Grid3D& actual,
                      const grid::Grid3D& expected) {
  grid::Grid3D delta = actual - expected;
  delta.ApplyAbs();
  return delta.Max();
}*/

// actual -- pointer to data in gpu device memory
// expected -- grid in host memory
double CalculateErrorCuda(grid::Grid3D& actual, grid::Grid3D& expected) {
  size_t size = actual.data_size();

  dim3 block_dim(1024, 1, 1);
  dim3 grid_dim((size + 1023) / 1024, 1, 1);
  cuda_kernels::error_kernel<<<grid_dim, block_dim>>>(
    actual.data(), expected.data(), cuda_buffers::cuda_info_);
  

  std::vector<double> tmp(size, 0);
  cudaMemcpy(tmp.data(), expected.data(), size * sizeof(double),
             cudaMemcpyDeviceToHost);
  

  double max_value = tmp[0];
  for (size_t i = 1; i < size; ++i) {
    if (tmp[i] > max_value) {
      max_value = tmp[i];
    }
  }
  return max_value;
}

int IndexYZ(int y, int z, const grid::Point<int>& n) {
  return y * n.z + z;
}

int IndexXZ(int x, int z, const grid::Point<int>& n) {
  return x * n.z + z;
}

int IndexXY(int x, int y, const grid::Point<int>& n) {
  return x * n.y + y;
}

// sin(2 * pi * x / X + 3 * pi) * sin(2 * pi * y / Y + 2 * pi) * 
// sin(pi * z / Z) * cos(pi * sqrt(4 / (X * X) + 4 / (Y * Y) + 
// 1 / (Z * Z)) * t + pi)
double AnalyticalU(const grid::Point<double>& p, double t,
                   const grid::Point<double>& pL) {
  using std::sin;
  grid::Point<double> coef(2 * M_PI, 2 * M_PI, M_PI);
  grid::Point<double> addition(3 * M_PI, 2 * M_PI, 0);
  grid::Point<double> insin = coef * p / pL + addition;
  double at = M_PI * std::sqrt(4 / (pL.x * pL.x) + 4 / (pL.y * pL.y) +
                               1 / (pL.z * pL.z));
  return sin(insin.x) * sin(insin.y) * sin(insin.z) * std::cos(at * t + M_PI);
}

void SetGridEtalon(grid::Grid3D& grid, const grid::Point<double>& pL,
                   double t, bool are_edges[6]) {
  grid::Point<int> n = grid.size();
  grid::Point<double> delta = grid.delta();

  std::vector<double> data(grid.data_size(), 0);

  for (int i = 0; i < n.x; ++i) {
    for (int j = 0; j < n.y; ++j) {
      for (int k = 0; k < n.z; ++k) {
        data[grid.LinearIndex(i, j, k)] =
          AnalyticalU(grid.PointFromIndices(i, j, k), t, pL);
      }
    }
  }

  for (int y = 0; y < n.y; ++y) {
    for (int z = 0; z < n.z; ++z) {
      grid::Point<double> zero = grid.PointFromIndices(0, y, z);
      zero.x = delta.x;
      grid::Point<double> last = zero;
      last.x = pL.x - delta.x;
      double value = (AnalyticalU(zero, t, pL) +
                      AnalyticalU(last, t, pL)) / 2;
      if (are_edges[0]) {
        data[grid.LinearIndex(0, y, z)] = value;
      }
      if (are_edges[3]) {
        data[grid.LinearIndex(n.x - 1, y, z)] = value;
      }
    }
  }
  for (int x = 0; x < n.x; ++x) {
    for (int z = 0; z < n.z; ++z) {
      grid::Point<double> zero = grid.PointFromIndices(x, 0, z);
      zero.y = delta.y;
      grid::Point<double> last = zero;
      last.y = pL.y - delta.y;
      double value = (AnalyticalU(zero, t, pL) +
                      AnalyticalU(last, t, pL)) / 2;
      if (are_edges[1]) {
        data[grid.LinearIndex(x, 0, z)] = value;
      }
      if (are_edges[4]) {
        data[grid.LinearIndex(x, n.y - 1, z)] = value;
      }
    }
  }
  for (int x = 0; x < n.x; ++x) {
    for (int y = 0; y < n.y; ++y) {
      if (are_edges[2]) {
        data[grid.LinearIndex(x, y, 0)] = 0;
      }
      if (are_edges[5]) {
        data[grid.LinearIndex(x, y, n.z - 1)] = 0;
      }
    }
  }
  grid.SetData(data);
}

void SetGridT0(grid::Grid3D& grid, const grid::Point<double> pL,
                          bool are_edges[6]) {
  SetGridEtalon(grid, pL, /*t=*/0, are_edges);
}

namespace cuda_kernels {

__global__ void laplassian_kernel(double* in_data, double* out_data,
                                  cuda_buffers::CudaInfo* info,
                                  double* left, double* right);

} // namespace cuda_kernels

// in and out are required to be the same in terms of all fields except data_.
void Laplassian7PointsCuda(grid::Grid3D& in, grid::Grid3D& out) {
  dim3 block_dim(8, 8, 16);
  dim3 grid_dim((in.size().x + 7) / 8,
                (in.size().y + 7) / 8,
                (in.size().z + 15) / 16);
  cuda_kernels::laplassian_kernel<<<grid_dim, block_dim>>>(
    in.data(), out.data(), cuda_buffers::cuda_info_, in.left(), in.right());
  
}

// Process edges. For global edges: periodical law for x & y (equality of
// values and derivatives), constant 0 for z. For local just laplassian etc.
// *_neghbours --> [x, y, z]
void UpdateEdges(grid::Grid3D& out, int left_neighbours[3],
                 int right_neighbours[3], bool are_edges[6], int coords3d[3],
                 int dimensions[3], const MPI_Comm& cartComm);

void CalculateGlobalEdges(grid::Grid3D& out);

namespace cuda_kernels {

__global__ void edges_kernel(double* out_data, double* are_edges, double* left,
                             double* right, cuda_buffers::CudaInfo* info);

__global__ void t1_kernel(double* t0, double* out,
                          cuda_buffers::CudaInfo* info) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int idx = x;
  if (y > idx) idx = y;
  if (z > idx) idx = z;
  if (idx >= info->data_size) {
    return;
  }

  double value = out[idx];
  value *= info->tau * info->tau / 2;
  value += t0[idx];
  out[idx] = value;
}

} // namespace cuda_kernels

// Applies laplassian operator to t0. Edges should be updated by caller.
void SetGridT1(grid::Grid3D& grid_t0, grid::Grid3D& grid_t1) {
  Laplassian7PointsCuda(grid_t0, grid_t1);

  //return (grid_t0 + laplassian * (tau * tau / 2));
  dim3 block_dim(1024, 1, 1);
  dim3 grid_dim((grid_t0.data_size() + 1023) / 1024, 1, 1);
  cuda_kernels::t1_kernel<<<grid_dim, block_dim>>>(
    grid_t0.data(), grid_t1.data(), cuda_buffers::cuda_info_);
  
}

namespace cuda_kernels {

__global__ void step_kernel(double* in_prev_data, double* in_data,
                            double* out_data,
                            cuda_buffers::CudaInfo* cuda_info) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int idx = x;
  if (y > idx) idx = y;
  if (z > idx) idx = z;
  if (idx >= cuda_info->data_size) {
    return;
  }

  double value = out_data[idx];
  value *= cuda_info->tau * cuda_info->tau;
  value += 2 * in_data[idx];
  value -= in_prev_data[idx];
  out_data[idx] = value;
}

} // namespace cuda_kernels

void StepCuda(grid::Grid3D& in_prev, grid::Grid3D& in, grid::Grid3D& out) {
  Laplassian7PointsCuda(in, out);

  dim3 block_dim(1024, 1, 1);
  dim3 grid_dim((in.data_size() + 1023) / 1024, 1, 1);
  cuda_kernels::step_kernel<<<grid_dim, block_dim>>>(
    in_prev.data(), in.data(), out.data(), cuda_buffers::cuda_info_);
  
}

void Calculate3Dimensions(int world_size, int dimensions[3]);

int main(int argc, char* argv[]) {
  // PARAMETER
  double L = 1.0;
  int grid_axis_size = 128; // points along one axis

  const grid::Point<double> pL(L, L, L); // global borders

  MPI_Init(&argc, &argv);
  
  MPI_Barrier(MPI_COMM_WORLD);
  double startTime = MPI_Wtime();

  int rank, world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size); // number of processors
  
  int dimensions[3]; // number of processors along each axis
  Calculate3Dimensions(world_size, dimensions);

  int periods[3] = {true, true, true};
  MPI_Comm cartComm;
  MPI_Cart_create(MPI_COMM_WORLD, /*ndims=*/3, dimensions, periods,
                  /*reorder=*/false, &cartComm);
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // process id
  if (rank == 0) {
    std::stringstream ss;
    ss << "num proc " << world_size << ", axis size " << grid_axis_size << "\n";
    std::cout << ss.str();
  }

  int coords3d[3]; // coordinates of current processor
  MPI_Cart_coords(cartComm, rank, /*maxdims=*/3, coords3d);

  // neighbours {x, y, z}
  int left_neighbours[3] = {-1, -1, -1};
  int right_neighbours[3] = {-1, -1, -1};
  bool are_edges[6] = {false, false, false, false, false, false};
  
  for (int i = 0; i < 3; ++i) {
    int tmp;
    MPI_Cart_shift(cartComm, i, -1, &tmp, left_neighbours + i);
    MPI_Cart_shift(cartComm, i, 1, &tmp, right_neighbours + i);
    if (coords3d[i] == 0) {
      are_edges[i] = true;
    }
    if (coords3d[i] == dimensions[i] - 1) {
      are_edges[i + 3] = true;
    }
  }

  // calculate grid definers
  grid::Point<double> delta(L / (grid_axis_size - 1),
                      L / (grid_axis_size - 1),
                      L / (grid_axis_size - 1));
  grid::Point<int> local_grid_num_points(grid_axis_size / dimensions[0],
                                   grid_axis_size / dimensions[1],
                                   grid_axis_size / dimensions[2]);
  grid::Point<int> mod(grid_axis_size % dimensions[0],
                 grid_axis_size % dimensions[1],
                 grid_axis_size % dimensions[2]);

  grid::Point<int> p0_point(coords3d[0] * local_grid_num_points.x,
                      coords3d[1] * local_grid_num_points.y,
                      coords3d[2] * local_grid_num_points.z);
  if (coords3d[0] < mod.x) {
    local_grid_num_points.x += 1;
  }
  if (coords3d[1] < mod.y) {
    local_grid_num_points.y += 1;
  }
  if (coords3d[2] < mod.z) {
    local_grid_num_points.z += 1;
  }

  p0_point.x += std::min(coords3d[0], mod.x);
  p0_point.y += std::min(coords3d[1], mod.y);
  p0_point.z += std::min(coords3d[2], mod.z);

  grid::Point<double> p0(p0_point.x * delta.x,
                   p0_point.y * delta.y,
                   p0_point.z * delta.z);
  
  grid::Point<int> edges_subtract(1 ? are_edges[3] : 0,
                            1 ? are_edges[4] : 0,
                            1 ? are_edges[5] : 0);
  grid::Point<int> pN_point = p0_point + local_grid_num_points - edges_subtract;
  grid::Point<double> pN(pN_point.x * delta.x,
                   pN_point.y * delta.y,
                   pN_point.z * delta.z);

  // Create grid for t=0
  grid::Grid3D grid_t0(p0, pN, local_grid_num_points, delta);
  SetGridT0(grid_t0, pL, are_edges);

  // Initialize tau & t info
  const int num_time_points = 20;
  std::vector<double> errors(num_time_points, 0);
  double tau = grid_t0.delta().Min();
  tau = tau * tau / 20;

  cuda_buffers::AllocVariables(grid_t0, tau, are_edges);
  
  // Create grid for t=1 and calculate error with etalon
  grid::Grid3D grid_t1(p0, pN, local_grid_num_points, delta);
  SetGridT1(grid_t0, grid_t1);

  // general formula for updates inside u(n+1) = tau^2 * laplassian + 2u(n) - u(n-1).
  // so if u(n) == u(n-1) == grid_t0, tau = tau / sqrt(2), we get
  // u1 = tau^2 / 2 * laplassian + u0
  UpdateEdges(grid_t1, left_neighbours, right_neighbours, are_edges, coords3d,
              dimensions, cartComm);

  grid::Grid3D etalon(p0, pN, local_grid_num_points, delta);

  errors[1] = CalculateErrorCuda(grid_t1, etalon);

  grid::Grid3D grid_t2(p0, pN, local_grid_num_points, delta);

  grid::Grid3D* in_prev = &grid_t0;
  grid::Grid3D* in = &grid_t1;
  grid::Grid3D* out = &grid_t2;

  for (int t = 2; t < num_time_points; ++t) {
    StepCuda(*in_prev, *in, *out);
    UpdateEdges(*out, left_neighbours, right_neighbours, are_edges, coords3d,
                dimensions, cartComm);

    SetGridEtalon(etalon, pL, t, are_edges);
    errors[t] = CalculateErrorCuda(*out, etalon);

    grid::Grid3D* tmp = in_prev;
    in_prev = in;
    in = out;
    out = tmp;
  }

  double reduced_error[num_time_points];
  std::vector<MPI_Request> error_requests(num_time_points);

  // Reduce max of errors on each iter
  int root_rank;
  int zero_coords[3] = {0, 0, 0};
  MPI_Cart_rank(cartComm, zero_coords, &root_rank);
  for (int t = 0; t < num_time_points; ++t) {
    MPI_Reduce(&(errors[t]), reduced_error + t, /*count=*/1, MPI_DOUBLE,
                MPI_MAX, root_rank, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double endTime = MPI_Wtime();

  if (rank == root_rank) {
    std::stringstream ss;
    for (int t = 0; t < num_time_points; ++t) {
      ss << "Iteration " << t << ", error " << std::setprecision(9) <<
            reduced_error[t] << "\n";
    }
    ss << "Processing time : " << endTime - startTime << " seconds\n";
    std::cout << ss.str();
  }

  cuda_buffers::FreeVariables();
  MPI_Finalize();
  return 0;
}

namespace cuda_kernels {

__global__ void laplassian_kernel(double* in_data, double* out_data,
                                  cuda_buffers::CudaInfo* info,
                                  double* left, double* right) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < info->size_x && y < info->size_y && z < info->size_z) {
    // OK
  } else {
    return;
  }

  int yz_size = info->size_y * info->size_z;

  int linear_idx = x * yz_size + y * info->size_z + z;

  if (linear_idx >= info->data_size) {
    return;
  }

  int edge_start_y = info->size_y * info->size_z;
  int edge_start_z = edge_start_y + info->size_x * info->size_z;

  double xyz, x_left, x_right, y_left, y_right, z_left, z_right;
  xyz = 2 * in_data[linear_idx];
  // x left block edge
  if (x == 0) {
    x_left = left[y * info->size_z + z];
  } else {
    x_left = in_data[linear_idx - yz_size];
  }
  // y left block edge
  if (y == 0) {
    y_left = left[edge_start_y + x * info->size_z + z];
  } else {
    y_left = in_data[linear_idx - info->size_z];
  }
  // z left block edge
  if (z == 0) {
    z_left = left[edge_start_z + x * info->size_y + y];
  } else {
    z_left = in_data[linear_idx - 1];
  }
  // x right block edge
  if (x == info->size_x - 1) {
    x_right = right[y * info->size_z + z];
  } else {
    x_right = in_data[linear_idx + yz_size];
  }
  // y right block edge
  if (y == info->size_y - 1) {
    y_right = right[edge_start_y + x * info->size_z + z];
  } else {
    y_right = in_data[linear_idx + info->size_z];
  }
  // z right block edge
  if (z == info->size_z - 1) {
    z_right = right[edge_start_z + x * info->size_y + y];
  } else {
    z_right = in_data[linear_idx + 1];
  }

  double delta_x = info->delta_x * info->delta_x;
  double delta_y = info->delta_y * info->delta_y;
  double delta_z = info->delta_z * info->delta_z;

  out_data[linear_idx] = (x_left - xyz + x_right) / delta_x + 
                         (y_left - xyz + y_right) / delta_y + 
                         (z_left - xyz + z_right) / delta_z;
}

// send second to edge elements if global edges, and edge elements for local edges.
__global__ void write_edges_kernel(double* data, double* left, double* right,
                              cuda_buffers::CudaInfo* info, bool* are_edges) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x == 0 || x == info->size_x - 1 || y == 0 || y == info->size_y - 1 ||
      z == 0 || z == info->size_z - 1) {
    // OK
  } else {
    return;
  }

  int yz_size = info->size_y * info->size_z;
  int xz_size = info->size_x * info->size_z;

  int linear_idx = x * yz_size + y * info->size_z + z;
  if (linear_idx >= info->data_size) { return; }

  int indices[3] = {y * info->size_z + z,
                    yz_size + x * info->size_z + z,
                    yz_size + xz_size + x * info->size_y + y};
  if (x == 0) {
    if (are_edges[0]) { // global left
      left[indices[0]] = data[linear_idx + yz_size];
    } else {
      left[indices[0]] = data[linear_idx];
    }
  }
  if (x == info->size_x) {
    if (are_edges[3]) { // global right
      right[indices[0]] = data[linear_idx - yz_size];
    } else {
      right[indices[0]] = data[linear_idx];
    }
  }
  if (y == 0) {
    if (are_edges[1]) { // global left
      left[indices[1]] = data[linear_idx + info->size_z];
    } else {
      left[indices[1]] = data[linear_idx];
    }
  }
  if (y == info->size_y) {
    if (are_edges[4]) { // global right
      right[indices[1]] = data[linear_idx - info->size_z];
    } else {
      right[indices[1]] = data[linear_idx];
    }
  }
  if (z == 0) {
    if (are_edges[2]) { // global left
      left[indices[2]] = data[linear_idx + 1];
    } else {
      left[indices[2]] = data[linear_idx];
    }
  }
  if (z == info->size_z) {
    if (are_edges[5]) { // global right
      right[indices[2]] = data[linear_idx - 1];
    } else {
      right[indices[2]] = data[linear_idx];
    }
  }
}

} // namespace cuda_kernels

void UpdateEdges(grid::Grid3D& out, int left_neighbours[3],
                 int right_neighbours[3], bool are_edges[6], int coords3d[3],
                 int dimensions[3], const MPI_Comm& cartComm) {
  const grid::Point<int> n = out.size();
  
  std::vector<MPI_Request> send_requests(6); // left first, right last
  std::vector<MPI_Request> recv_requests(6); // left first, right last

  // create bufs
  int sizes[3] = {out.edge_sizes().x, out.edge_sizes().y, out.edge_sizes().z};
  int starts[3] = {out.starts().x, out.starts().y, out.starts().z};

  std::vector<double*> left_send_buf(3);
  std::vector<double*> right_send_buf(3);

  std::vector<double*> left_recv_buf(3);
  std::vector<double*> right_recv_buf(3);
  
  for (int i = 0; i < 3; ++i) {
    left_send_buf[i] = new double[sizes[i]];
    right_send_buf[i] = new double[sizes[i]];

    left_recv_buf[i] = new double[sizes[i]];
    right_recv_buf[i] = new double[sizes[i]];
  }
  // init send bufs.
  dim3 block_dim(8, 8, 16);
  dim3 grid_dim((out.size().x + 7) / 8,
                (out.size().y + 7) / 8,
                (out.size().z + 15) / 16);
  cuda_kernels::write_edges_kernel<<<grid_dim, block_dim>>>(
    out.data(), cuda_buffers::left_, cuda_buffers::right_,
    cuda_buffers::cuda_info_, cuda_buffers::are_edges_);

  for (int i = 0; i < 3; ++i) {
    cudaMemcpy(left_send_buf[i], cuda_buffers::left_ + starts[i],
               sizes[i] * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(right_send_buf[i], cuda_buffers::right_ + starts[i],
               sizes[i] * sizeof(double), cudaMemcpyDeviceToHost);
  }

  // send recv
  for (int i = 0; i < 3; ++ i) {
    // left neighbour
    MPI_Isend(left_send_buf[i], sizes[i], MPI_DOUBLE, left_neighbours[i],
              /*tag=*/0, MPI_COMM_WORLD, &send_requests[i]);
    MPI_Irecv(left_recv_buf[i], sizes[i], MPI_DOUBLE, left_neighbours[i],
              /*tag=*/0, MPI_COMM_WORLD, &recv_requests[i]);
    // right neighbour
    MPI_Isend(right_send_buf[i], sizes[i], MPI_DOUBLE, right_neighbours[i],
              /*tag=*/0, MPI_COMM_WORLD, &send_requests[i + 3]);
    MPI_Irecv(right_recv_buf[i], sizes[i], MPI_DOUBLE, right_neighbours[i],
              /*tag=*/0, MPI_COMM_WORLD, &recv_requests[i + 3]);
  }

  // wait for requests to finish
  for (int i = 0; i < 6; ++i) {
    MPI_Status status;
    MPI_Wait(&send_requests[i], &status);
    MPI_Wait(&recv_requests[i], &status);
  }

  // save received points
  for (int i = 0; i < 3; ++i) {
    cudaMemcpy(out.left() + starts[i], left_recv_buf[i],
               sizes[i] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(out.right() + starts[i], right_recv_buf[i],
               sizes[i] * sizeof(double), cudaMemcpyHostToDevice);
  }
  
  // now we only have to calculate global edges, using newly received info
  CalculateGlobalEdges(out);  

  for (int i = 0; i < 3; ++i) {
    delete[] left_send_buf[i];
    delete[] right_send_buf[i];
    delete[] left_recv_buf[i];
    delete[] right_recv_buf[i];
  }
}

namespace cuda_kernels {

__global__ void edges_kernel(double* out_data, bool* are_edges, double* left,
                             double* right, cuda_buffers::CudaInfo* info) {
  bool needed = false;
  for (int i = 0; i < 6; ++i) {
    needed = needed || are_edges[i];
  }
  if (!needed) {
    return;
  }

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x == 0 || x == info->size_x - 1 || y == 0 || y == info->size_y - 1 ||
      z == 0 || z == info->size_z - 1) {
    // OK
  } else {
    return;
  }

  int yz_size = info->size_y * info->size_z;
  int xz_size = info->size_x * info->size_z;

  int linear_idx = x * yz_size + y * info->size_z + z;

  if (linear_idx >= info->data_size) {
    return;
  }

  if (x == 0 && are_edges[0]) {
    double x_left = left[y * info->size_z + z];
    x_left += out_data[linear_idx + yz_size]; // x+1, y, z
    out_data[linear_idx] = x_left / 2;
  }
  if (x == info->size_x - 1 && are_edges[3]) {
    double x_right = right[y * info->size_z + z];
    x_right += out_data[linear_idx - yz_size]; // x-1, y, z
    out_data[linear_idx] = x_right / 2;
  }
  if (y == 0 && are_edges[1]) {
    double y_left = left[yz_size + x * info->size_z + z];
    y_left += out_data[linear_idx + info->size_z]; // x, y+1, z
    out_data[linear_idx] = y_left / 2;
  }
  if (y == info->size_y - 1 && are_edges[4]) {
    double y_right = right[yz_size + x * info->size_z + z];
    y_right += out_data[linear_idx - info->size_z]; // x, y-1, z
    out_data[linear_idx] = y_right / 2;
  }
  if (z == 0 && are_edges[2]) {
    double z_left = left[yz_size + xz_size + x * info->size_y + y];
    z_left += out_data[linear_idx + 1]; // x, y, z+1
    out_data[linear_idx] = z_left / 2;
  }
  if (z == info->size_z - 1 && are_edges[5]) {
    double z_right = right[yz_size + xz_size + x * info->size_y + y];
    z_right += out_data[linear_idx - 1]; // x, y, z-1
    out_data[linear_idx] = z_right / 2;
  }
}

} // namespace cuda_kernels

void CalculateGlobalEdges(grid::Grid3D& out) {
  dim3 block_dim(8, 8, 16);
  dim3 grid_dim((out.size().x + 7) / 8,
                (out.size().y + 7) / 8,
                (out.size().z + 15) / 16);
  cuda_kernels::edges_kernel<<<grid_dim, block_dim>>>(
    out.data(), cuda_buffers::are_edges_, out.left(), out.right(),
    cuda_buffers::cuda_info_);
  
}

void Calculate3Dimensions(int world_size, int dimensions[3]) {
  switch (world_size) {
    case 1:
      dimensions[0] = 1;
      dimensions[1] = 1;
      dimensions[2] = 1;
      break;
    case 2:
      dimensions[0] = 1;
      dimensions[1] = 1;
      dimensions[2] = 2;
      break;
    case 3:
      dimensions[0] = 1;
      dimensions[1] = 1;
      dimensions[2] = 3;
      break;
    case 4:
      dimensions[0] = 1;
      dimensions[1] = 2;
      dimensions[2] = 2;
      break;
    case 6:
      dimensions[0] = 2;
      dimensions[1] = 1;
      dimensions[2] = 3;
      break;
    case 8:
      dimensions[0] = 2;
      dimensions[1] = 2;
      dimensions[2] = 2;
      break;
    case 16:
      dimensions[0] = 2;
      dimensions[1] = 2;
      dimensions[2] = 4;
      break;
    case 64:
      dimensions[0] = 4;
      dimensions[1] = 4;
      dimensions[2] = 4;
      break;
    case 128:
      dimensions[0] = 4;
      dimensions[1] = 4;
      dimensions[2] = 8;
      break;
    case 256:
      dimensions[0] = 4;
      dimensions[1] = 8;
      dimensions[2] = 8;
      break;
    case 10:
      dimensions[0] = 2;
      dimensions[1] = 5;
      dimensions[2] = 1;
      break;
    case 20:
      dimensions[0] = 2;
      dimensions[1] = 2;
      dimensions[2] = 5;
      break;
    case 40:
      dimensions[0] = 2;
      dimensions[1] = 4;
      dimensions[2] = 5;
      break;
  }
}

namespace grid {

void Grid3D::SetData(const std::vector<double>& new_data) {
  cudaMemcpy(data_, new_data.data(), data_size_ * sizeof(double),
             cudaMemcpyHostToDevice);
  
}

Point<double> Grid3D::PointFromIndices(int i, int j, int k) {
  return Point<double>(p0_.x + i * delta_.x,
                       p0_.y + j * delta_.y,
                       p0_.z + k * delta_.z);
}

int Grid3D::LinearIndex(int i, int j, int k) const {
  return i * num_points_.y * num_points_.z + j * num_points_.z + k;
}

/*
std::string Grid3D::CompareGridMetaData(const Grid3D& other) const {
  if (num_points_ != other.num_points_) {
    return "grids have different number of points";
  }
  if (data_.size() != other.data_.size()) {
    return "grids have different sizes";
  }
  if (p0_ != other.p0_) {
    return "grids have different start points";
  }
  if (pN_ != other.pN_) {
    return "grids have different end points";
  } if (delta_ != other.delta_) {
    return "grids have different delta between points";
  }
  return "OK";
}

double Grid3D::at(int i, int j, int k) const {
  return data_.at(LinearIndex(i, j, k));
}

double& Grid3D::set(int i, int j, int k) {
  return data_.at(LinearIndex(i, j, k));
}

Grid3D Grid3D::operator+(const Grid3D& other) const {
  std::string message = CompareGridMetaData(other);
  if (message != std::string("OK")) {
    throw std::invalid_argument(message);
  }

  Grid3D result(p0_, pN_, num_points_, delta_);
  for (size_t idx = 0; idx < data_.size(); ++idx) {
    result.data_[idx] = data_[idx] + other.data_[idx];
  }
  return result;
}

void Grid3D::operator+=(const Grid3D& other) {
  std::string message = CompareGridMetaData(other);
  if (message != std::string("OK")) {
    throw std::invalid_argument(message);
  }

  for (size_t idx = 0; idx < data_.size(); ++idx) {
    data_[idx] += other.data_[idx];
  }
}

Grid3D Grid3D::operator-(const Grid3D& other) const {
  std::string message = CompareGridMetaData(other);
  if (message != std::string("OK")) {
    throw std::invalid_argument(message);
  }

  Grid3D result(p0_, pN_, num_points_, delta_);
  for (size_t idx = 0; idx < data_.size(); ++idx) {
    result.data_[idx] = data_[idx] - other.data_[idx];
  }
  return result;
}

void Grid3D::operator-=(const Grid3D& other) {
  std::string message = CompareGridMetaData(other);
  if (message != std::string("OK")) {
    throw std::invalid_argument(message);
  }

  for (size_t idx = 0; idx < data_.size(); ++idx) {
    data_[idx] -= other.data_[idx];
  }
}

Grid3D Grid3D::operator*(double multiplier) const {
  Grid3D result(p0_, pN_, num_points_, delta_);
  for (size_t idx = 0; idx < data_.size(); ++idx) {
    result.data_[idx] = data_[idx] * multiplier;
  }
  return result;
}

void Grid3D::operator*=(double multiplier) {
  for (size_t idx = 0; idx < data_.size(); ++idx) {
    data_[idx] *= multiplier;
  }
}

void Grid3D::PrintGrid() const {
  std::cout << "size "  << num_points_.DebugString() << "\n[";
  for (int y = 0; y < num_points_.y; ++y) {
    std::cout << "[";
    for (int z = 0; z < num_points_.z; ++z) {
      std::cout << data_[LinearIndex(32, y, z)] << ", ";
    }
    std::cout << "]\n";
  }
  std::cout << "]\n";
}

void Grid3D::PrintDelta() const {
  std::cout << "delta: ";
  delta_.Print();
}

std::string Grid3D::DebugString() const {
  std::stringstream ss;
  ss << "grid: size " << data_.size() << "\n";
  for (int x = 0; x < num_points_.x; ++x) {
    for (int y = 0; y < num_points_.y; ++y) {
      for (int z = 0; z < num_points_.z; ++z) {
        ss << data_[LinearIndex(x, y, z)] << " ";
      }
      ss << "\n";
    }
    ss << "\n";
  }
  return ss.str();
}

double Grid3D::Sum() const {
  double res = 0.;
  for (size_t i = 0; i < data_.size(); ++i) {
    res += data_[i];
  }
  return res;
}

double Grid3D::Max() const {
  double res = data_[0];
  int coord = 0;
  int counter = 1;
  for (size_t i = 1; i < data_.size(); ++i) {
    if (std::fabs(data_[i] - res) < 1e-7) {
      counter++;
    }
    if (data_[i] > res) {
      res = data_[i];
      coord = i;
      counter = 1;
    }
  }
  for (int i = 1; i < num_points_.x - 1; ++i) {
    for (int j = 1; j < num_points_.y - 1; ++j) {
      for (int k = 1; k < num_points_.z - 1; ++k) {
        int idx = LinearIndex(i, j, k);
        if (std::fabs(data_[idx] - res) < 1e-7) {
          counter++;
        }
        if (data_[idx] > res) {
          res = data_[idx];
          coord = i;
          counter = 1;
        }
      }
    }
  }
  int x = coord / (num_points_.y * num_points_.z);
  int y = (coord % (num_points_.y * num_points_.z)) / num_points_.z;
  int z = (coord % (num_points_.y * num_points_.z)) % num_points_.z;

  {
    std::stringstream ss;
    ss << "MAX COORD " << coord << ", x " << x << ", y " << y << ", z "
       << z << " counter " << counter << " value " << res << "\n";
    //std::cout << ss.str();
  }
  return res;
}

void Grid3D::ApplyAbs() {
  for (size_t i = 0; i < data_.size(); ++i) {
    data_[i] = std::fabs(data_[i]);
  }
}
*/

} // namespace grid