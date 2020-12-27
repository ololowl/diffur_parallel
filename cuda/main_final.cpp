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
// coverage is (0, xL] x (0, yL] x (0, zL). Only cubical grids are suported:
// Number of points is equal for each axis and is passed in the constructor as n.
class Grid3D {
 public:
  std::vector<std::vector<double> > left_xyz;
  std::vector<std::vector<double> > right_xyz;

  Grid3D(Point<double> p0, Point<double> pN, Point<int> axis_num_points,
         Point<double> delta)
    : data_(axis_num_points.x * axis_num_points.y * axis_num_points.z, 0),
      num_points_(axis_num_points), p0_(p0), pN_(pN), delta_(delta) {
    InitVectors();
  }

  Grid3D(const Grid3D& other)
    : left_xyz(other.left_xyz), right_xyz(other.right_xyz), data_(other.data_),
      num_points_(other.num_points_), p0_(other.p0_), pN_(other.pN_),
      delta_(other.delta_) {
  }

  double at(int i, int j, int k) const;
  double& set(int i, int j, int k);

  Grid3D operator+(const Grid3D& other) const;
  void operator+=(const Grid3D& other);

  Grid3D operator-(const Grid3D& other) const;
  void operator-=(const Grid3D& other);

  Grid3D operator*(double multiplier) const;
  void operator*=(double multiplier);

  Point<double> PointFromIndices(int i, int j, int k);

  void PrintGrid() const;
  void PrintDelta() const;

  std::string DebugString() const;

  double Sum() const;
  double Max() const;
  void ApplyAbs();

  void Clear();

  Point<int> size() const { return num_points_; }
  Point<double> delta() const { return delta_; }
  Point<double> p0() const { return p0_; }
  Point<double> pN() const { return pN_; }
  std::vector<double> data() const { return data_; }

 private:
  int LinearIndex(int i, int j, int k) const;
  std::string CompareGridMetaData(const Grid3D& other) const;
  void InitVectors();

  std::vector<double> data_;
  Point<int> num_points_; // number of points along each axis

  Point<double> p0_; // start borders of intervals
  Point<double> pN_; // end borders of intervals
  Point<double> delta_; // deltas between grid points
};

} // namespace grid

namespace error {

double* in_data_;
double* out_data_;
size_t* data_size_;

void alloc_variables(const grid::Grid3D& in) {
  size_t size = in.data().size();
  
  cudaMalloc(&data_size_, sizeof(size_t));
  cudaMemcpy(data_size_, &size, sizeof(size_t), cudaMemcpyHostToDevice);

  cudaMalloc(&in_data_, size * sizeof(double));
  cudaMalloc(&out_data_, size * sizeof(double));
}

void free_variables() {
  cudaFree(in_data_);
  cudaFree(out_data_);
  cudaFree(data_size_);
}

__global__ void error_kernel(double* in_data, double* out_data,
                             size_t* data_size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int idx = x;
  if (y > idx) idx = y;
  if (z > idx) idx = z;
  if (idx >= *data_size) {
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

} // namespace error

double CalculateError(const grid::Grid3D& actual,
                      const grid::Grid3D& expected) {
  /*
  grid::Grid3D delta = actual - expected;
  delta.ApplyAbs();
  return delta.Max();
  */
  size_t size = actual.data().size();

  cudaMemcpy(error::in_data_, actual.data().data(),
             size * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(error::out_data_, expected.data().data(),
             size * sizeof(double), cudaMemcpyHostToDevice);

  dim3 block_dim(1024, 1, 1);
  dim3 grid_dim((size + 1023) / 1024, 1, 1);
  error::error_kernel<<<grid_dim, block_dim>>>(
    error::in_data_, error::out_data_, error::data_size_);

  std::vector<double> tmp(size, 0);
  cudaMemcpy(tmp.data(), error::out_data_,
             size * sizeof(double), cudaMemcpyDeviceToHost);

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

grid::Grid3D CreateGridEtalon(const grid::Point<double>& p0,
                              const grid::Point<double>& pN,
                              const grid::Point<int>& n,
                              const grid::Point<double>& pL, double t,
                              bool are_edges[6],
                              const grid::Point<double>& delta) {
  grid::Grid3D grid(p0, pN, n, delta);
  for (int i = 1; i < n.x - 1; ++i) {
    for (int j = 1; j < n.y - 1; ++j) {
      for (int k = 1; k < n.z - 1; ++k) {
        grid.set(i, j, k) = AnalyticalU(grid.PointFromIndices(i, j, k), t, pL);
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
        grid.set(0, y, z) = value;
      }
      if (are_edges[3]) {
        grid.set(n.x - 1, y, z) = value;
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
        grid.set(x, 0, z) = value;
      }
      if (are_edges[4]) {
        grid.set(x, n.y - 1, z) = value;
      }
    }
  }
  for (int x = 0; x < n.x; ++x) {
    for (int y = 0; y < n.y; ++y) {
      if (are_edges[2]) {
        grid.set(x, y, 0) = 0;
      }
      if (are_edges[5]) {
        grid.set(x, y, n.z - 1) = 0;
      }
    }
  }
  return grid;
}

grid::Grid3D CreateGridT0(const grid::Point<double> p0,
                          const grid::Point<double> pN,
                          const grid::Point<int> n,
                          const grid::Point<double> pL,
                          bool are_edges[6], const grid::Point<double> delta) {
  return CreateGridEtalon(p0, pN, n, pL, /*t=*/0, are_edges, delta);
}

namespace laplassian {

struct CudaInfo {
  size_t data_size;
  int size_x, size_y, size_z;
  double delta_x, delta_y, delta_z;

  CudaInfo(size_t ds, const std::vector<int>& s,
           const grid::Point<double>& d)
    : data_size(ds), size_x(s[0]), size_y(s[1]), size_z(s[2]), delta_x(d.x),
      delta_y(d.y), delta_z(d.z) {}
};

__global__ void laplassian_kernel(double* in_data, double* out_data,
                                  CudaInfo* info,
                                  //double* in_x_left, double* in_x_right,
                                  //double* in_y_left, double* in_y_right,
                                  //double* in_z_left, double* in_z_right
                                  double* left, double* right);

CudaInfo* cuda_info_;
double* in_data_;
double* out_data_;

double* left_;
double* right_;
size_t edge_size_sum_;

double* left_data_;
double* right_data_;

void alloc_variables(const grid::Grid3D& in) {
  std::cout << "LAPL ALLOC\n";
  cudaMalloc(&cuda_info_, sizeof(CudaInfo));
  std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";
  cudaMalloc(&in_data_, sizeof(double) * in.data().size());
  std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";
  cudaMalloc(&out_data_, sizeof(double) * in.data().size());
  std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";

  edge_size_sum_ = in.left_xyz[0].size() +
                   in.left_xyz[1].size() +
                   in.left_xyz[2].size();
  left_ = new double[edge_size_sum_];
  right_ = new double[edge_size_sum_];
  std::cout << "edge_size_sum_ " << edge_size_sum_ << "\n";

  cudaMalloc(&left_data_, edge_size_sum_ * sizeof(double));
  std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";
  cudaMalloc(&right_data_, edge_size_sum_ * sizeof(double));
  std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";

  std::vector<int> edge_size{in.size().x,
                                in.size().y,
                                in.size().z};
  laplassian::CudaInfo info(in.data().size(), edge_size, in.delta());
  std::cout << "sizes " << info.size_x << " " << info.size_y << " "
            << info.size_z << "\n";
  cudaMemcpy(
    laplassian::cuda_info_, &info, sizeof(info), cudaMemcpyHostToDevice);
  std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";
}

void free_variables() {
  cudaFree(cuda_info_);
  cudaFree(in_data_);
  cudaFree(out_data_);

  delete[] left_;
  delete[] right_;

  cudaFree(left_data_);
  cudaFree(right_data_);
}

} // namespace laplassian

// in and out are required to be the same in terms of all fields except data_.
void Laplassian7Points(const grid::Grid3D& in, grid::Grid3D& out) {
  cudaMemcpy(laplassian::in_data_, in.data().data(),
             sizeof(double) * in.data().size(), cudaMemcpyHostToDevice);

  // yz, xz, xy
  int starts[3] = {0,
                   in.size().y * in.size().z,
                   in.size().y * in.size().z + in.size().x * in.size().z};

  for (int i = 0; i < 3; ++i) {
    memcpy(laplassian::left_ + starts[i],
           in.left_xyz[i].data(),
           in.left_xyz[i].size() * sizeof(double));
    memcpy(laplassian::right_ + starts[i],
           in.right_xyz[i].data(),
           in.right_xyz[i].size() * sizeof(double));
  }

  cudaMemcpy(
    laplassian::left_data_, laplassian::left_,
    laplassian::edge_size_sum_ * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(
    laplassian::right_data_, laplassian::right_,
    laplassian::edge_size_sum_ * sizeof(double), cudaMemcpyHostToDevice);

  dim3 block_dim(8, 8, 16);
  dim3 grid_dim((in.size().x + 7) / 8,
                (in.size().y + 7) / 8,
                (in.size().z + 15) / 16);
  laplassian::laplassian_kernel<<<grid_dim, block_dim>>>(
    laplassian::in_data_, laplassian::out_data_, laplassian::cuda_info_,
    //left[0], left[1], left[2], right[0], right[1], right[2]
    laplassian::left_data_, laplassian::right_data_);
  std::cout << "lapl " << cudaGetErrorString(cudaGetLastError()) << "\n";

  cudaMemcpy(out.data().data(), laplassian::out_data_,
             sizeof(double) * out.data().size(), cudaMemcpyDeviceToHost);
}

// Process edges. For global edges: periodical law for x & y (equality of
// values and derivatives), constant 0 for z. For local just laplassian etc.
// *_neghbours --> [x, y, z]
void UpdateEdges(const grid::Grid3D& in_prev, const grid::Grid3D& in,
                 grid::Grid3D& out, double tau, int left_neighbours[3],
                 int right_neighbours[3], bool are_edges[6], int coords3d[3],
                 int dimensions[3], const MPI_Comm& cartComm);

// Calculates edges only for corresponding edges of {global_edges, local_edges}
// set to true.
void CalculateEdges(const grid::Grid3D& in_prev, const grid::Grid3D& in,
                    grid::Grid3D& out, bool are_edges[6], double tau,
                    bool global_edges, bool local_edges);

// Applies laplassian operator to t0. Edges should be updated by caller.
grid::Grid3D CreateGridT1(const grid::Grid3D& grid_t0, double tau) {
  grid::Grid3D grid(grid_t0);
  grid::Grid3D laplassian(
    grid_t0.p0(), grid_t0.pN(), grid_t0.size(), grid_t0.delta());

  Laplassian7Points(grid, laplassian);

  return (grid + laplassian * (tau * tau / 2));
}

namespace step {

struct CudaInfo {
  double tau;
  size_t data_size;

  CudaInfo(double t, size_t ds) : tau(t), data_size(ds) {}
};

double* in_prev_data_;
double* in_data_;
double* out_data_;
CudaInfo* cuda_info_;

void alloc_variables(const grid::Grid3D& in, double tau) {
  CudaInfo info(tau, in.data().size());

  cudaMalloc(&cuda_info_, sizeof(CudaInfo));
  cudaMemcpy(cuda_info_, &info, sizeof(CudaInfo), cudaMemcpyHostToDevice);

  cudaMalloc(&in_prev_data_, info.data_size * sizeof(double));
  cudaMalloc(&in_data_, info.data_size * sizeof(double));
  cudaMalloc(&out_data_, info.data_size * sizeof(double));
}

void free_variables() {
  cudaFree(in_prev_data_);
  cudaFree(in_data_);
  cudaFree(out_data_);
  cudaFree(cuda_info_);
}

__global__ void step_kernel(double* in_prev_data, double* in_data,
                            double* out_data, CudaInfo* cuda_info) {
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

} // namespace step

// arguments: u_{n-1}, u_n, u_{n+1}
// edges have to be processed by caller afterwards
void Step(const grid::Grid3D& in_prev, const grid::Grid3D& in,
          grid::Grid3D& out, double tau) {
  Laplassian7Points(in, out);
  /*
  out *= (tau * tau);
  out += in * 2.;
  out -= in_prev;
  return;
  */
  cudaMemcpy(step::in_prev_data_, in_prev.data().data(),
             in_prev.data().size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(step::in_data_, in.data().data(),
             in.data().size() * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(step::out_data_, out.data().data(),
             out.data().size() * sizeof(double), cudaMemcpyHostToDevice);

  dim3 block_dim(1024, 1, 1);
  dim3 grid_dim((in.data().size() + 1023) / 1024, 1, 1);
  step::step_kernel<<<grid_dim, block_dim>>>(
    step::in_prev_data_, step::in_data_, step::out_data_, step::cuda_info_);

  cudaMemcpy(out.data().data(), step::out_data_,
             out.data().size() * sizeof(double), cudaMemcpyDeviceToHost);
  
}

void alloc_all(const grid::Grid3D& grid_t0, double tau) {
  laplassian::alloc_variables(grid_t0);
  step::alloc_variables(grid_t0, tau);
  error::alloc_variables(grid_t0);
}

void free_all() {
  laplassian::free_variables();
  step::free_variables();
  error::free_variables();
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
  grid::Grid3D grid_t0 =
    CreateGridT0(p0, pN, local_grid_num_points, pL, are_edges, delta);
  // Initialize tau & t info
  const int num_time_points = 20;
  std::vector<double> errors(num_time_points, 0);
  double tau = grid_t0.delta().Min();
  tau = tau * tau / 10;

  alloc_all(grid_t0, tau);

  // Create grid for t=1 and calculate error with etalon
  grid::Grid3D grid_t1 = CreateGridT1(grid_t0, tau);
  // general formula for updates inside u(n+1) = tau^2 * laplassian + 2u(n) - u(n-1).
  // so if u(n) == u(n-1) == grid_t0, tau = tau / sqrt(2), we get
  // u1 = tau^2 / 2 * laplassian + u0
  UpdateEdges(grid_t0, grid_t0, grid_t1, tau / std::sqrt(2), left_neighbours,
              right_neighbours, are_edges, coords3d, dimensions, cartComm);
  errors[1] = CalculateError(
    CreateGridEtalon(p0, pN, local_grid_num_points, pL, tau, are_edges,
                     delta), grid_t1);

  grid::Grid3D grid_t2(p0, pN, local_grid_num_points, delta);

  grid::Grid3D* in_prev = &grid_t0;
  grid::Grid3D* in = &grid_t1;
  grid::Grid3D* out = &grid_t2;

  for (int t = 2; t < num_time_points; ++t) {
    Step(*in_prev, *in, *out, tau);
    UpdateEdges(*in_prev, *in, *out, tau, left_neighbours, right_neighbours,
                are_edges, coords3d, dimensions, cartComm);
    errors[t] = CalculateError(
      CreateGridEtalon(p0, pN, local_grid_num_points, pL, tau * t, are_edges,
                       delta), *out);
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

  free_all();
  MPI_Finalize();
  return 0;
}

namespace laplassian {

__global__ void laplassian_kernel(double* in_data, double* out_data,
                                  CudaInfo* info,
                                  //double* in_x_left, double* in_x_right,
                                  //double* in_y_left, double* in_y_right,
                                  //double* in_z_left, double* in_z_right
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

} // namespace laplassian

void UpdateEdges(const grid::Grid3D& in_prev, const grid::Grid3D& in,
                 grid::Grid3D& out, double tau, int left_neighbours[3],
                 int right_neighbours[3], bool are_edges[6], int coords3d[3],
                 int dimensions[3], const MPI_Comm& cartComm) {
  const grid::Point<int> n = out.size();
  
  std::vector<MPI_Request> send_requests(6); // left first, right last
  std::vector<MPI_Request> recv_requests(6); // left first, right last

  // first calculate local edges, as we already have enough info (they use info
  // only from in)
  //CalculateEdges(in_prev, in, out, are_edges, tau, /*global_edges=*/false,
  //               /*local_edges=*/true);

  // create bufs
  int sizes[3] = {n.y * n.z, n.x * n.z, n.x * n.y};
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
  // init send bufs. we send second to edge elements if global edges, and edge
  // elements for local edges.
  // x
  for (int y = 0; y < n.y; ++y) {
    for (int z = 0; z < n.z; ++z) {
      if (are_edges[0]) {
        left_send_buf[0][IndexYZ(y, z, n)] = out.at(1, y, z);
      } else {
        left_send_buf[0][IndexYZ(y, z, n)] = out.at(0, y, z);
      }
      if (are_edges[3]) {
        right_send_buf[0][IndexYZ(y, z, n)] = out.at(n.x - 2, y, z);
      } else {
        right_send_buf[0][IndexYZ(y, z, n)] = out.at(n.x - 1, y, z);
      }
    }
  }
  // y
  for (int x = 0; x < n.x; ++x) {
    for (int z = 0; z < n.z; ++z) {
      if (are_edges[1]) {
        left_send_buf[1][IndexXZ(x, z, n)] = out.at(x, 1, z);
      } else {
        left_send_buf[1][IndexXZ(x, z, n)] = out.at(x, 0, z);
      }
      if (are_edges[4]) {
        right_send_buf[1][IndexXZ(x, z, n)] = out.at(x, n.y - 2, z);
      } else {
        right_send_buf[1][IndexXZ(x, z, n)] = out.at(x, n.y - 1, z);
      }
    }
  }
  // z
  for (int x = 0; x < n.x; ++x) {
    for (int y = 0; y < n.y; ++y) {
      if (are_edges[2]) {
        left_send_buf[2][IndexXY(x, y, n)] = out.at(x, y, 1);
      } else {
        left_send_buf[2][IndexXY(x, y, n)] = out.at(x, y, 0);
      }
      if (are_edges[5]) {
        right_send_buf[2][IndexXY(x, y, n)] = out.at(x, y, n.z - 2);
      } else {
        right_send_buf[2][IndexXY(x, y, n)] = out.at(x, y, n.z - 1);
      }
    }
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
    for (int idx = 0; idx < sizes[i]; ++idx) {
      out.left_xyz[i][idx] = left_recv_buf[i][idx];
      out.right_xyz[i][idx] = right_recv_buf[i][idx];
    }
  }
  
  // now we only have to calculate global edges, using newly received info
  CalculateEdges(in_prev, in, out, are_edges, tau, /*global_edges=*/true,
                 /*local_edges=*/false);  

  for (int i = 0; i < 3; ++i) {
    delete[] left_send_buf[i];
    delete[] right_send_buf[i];
    delete[] left_recv_buf[i];
    delete[] right_recv_buf[i];
  }
}

void CalculateEdges(const grid::Grid3D& in_prev, const grid::Grid3D& in,
                    grid::Grid3D& out, bool are_edges[6], double tau,
                    bool global_edges, bool local_edges) {
  const grid::Point<int> n = out.size();
  // x
  const grid::Point<double> delta = in.delta() * in.delta();
  
  for (int y = 1; y < n.y - 1; ++y) {
    for (int z = 1; z < n.z - 1; ++z) {
      // left
      int x = 0;
      if (are_edges[0] && global_edges) { // left global edge
        out.set(x, y, z) =
          (out.left_xyz[0][IndexYZ(y, z, n)] + out.at(x + 1, y, z)) / 2;
      } else if (local_edges) { // local
        //double xyz = 2 * in.at(x, y, z);
        //double lapl = 
        //  (in.left_xyz[0][IndexYZ(y, z, n)] - xyz + in.at(x + 1, y, z)) / delta.x +
        //  (in.at(x, y - 1, z) - xyz + in.at(x, y + 1, z)) / delta.y +
        //  (in.at(x, y, z - 1) - xyz + in.at(x, y, z + 1)) / delta.z;
        //out.set(x, y, z) = xyz - in_prev.at(x, y, z) + tau * tau * lapl;
      }
      // right
      x = n.x - 1;
      if (are_edges[3] && global_edges) { // right global edge
        out.set(x, y, z) =
          (out.right_xyz[0][IndexYZ(y, z, n)] + out.at(x - 1, y, z)) / 2;
      } else if (local_edges) { // local
        //double xyz = 2 * in.at(x, y, z);
        //double lapl = 
        //  (in.at(x - 1, y, z) - xyz + in.right_xyz[0][IndexYZ(y, z, n)]) / delta.x +
        //  (in.at(x, y - 1, z) - xyz + in.at(x, y + 1, z)) / delta.y +
        //  (in.at(x, y, z - 1) - xyz + in.at(x, y, z + 1)) / delta.z;
        //out.set(x, y, z) = xyz - in_prev.at(x, y, z) + tau * tau * lapl;
      }
    }
  }
  // y
  for (int x = 1; x < n.x - 1; ++x) {
    for (int z = 1; z < n.z - 1; ++z) {
      // left
      int y = 0;
      if (are_edges[1] && global_edges) { // left global edge
        out.set(x, y, z) =
          (out.left_xyz[1][IndexXZ(x, z, n)] + out.at(x, y + 1, z)) / 2;
      } else if (local_edges) { // local
        //double xyz = 2 * in.at(x, y, z);
        //double lapl = 
        //  (in.at(x - 1, y, z) - xyz + in.at(x + 1, y, z)) / delta.x +
        //  (in.left_xyz[1][IndexXZ(x, z, n)] - xyz + in.at(x, y + 1, z)) / delta.y +
        //  (in.at(x, y, z - 1) - xyz + in.at(x, y, z + 1)) / delta.z;
        //out.set(x, y, z) = xyz - in_prev.at(x, y, z) + tau * tau * lapl;
      }
      // right
      y = n.y - 1;
      if (are_edges[4] && global_edges) { // right global edge
        out.set(x, y, z) =
          (out.right_xyz[1][IndexXZ(x, z, n)] + out.at(x, y - 1, z)) / 2;
      } else if (local_edges) { // local
        //double xyz = 2 * in.at(x, y, z);
        //double lapl = 
        //  (in.at(x - 1, y, z) - xyz + in.at(x + 1, y, z)) / delta.x +
        //  (in.at(x, y - 1, z) - xyz + in.right_xyz[1][IndexXZ(x, z, n)]) / delta.y +
        //  (in.at(x, y, z - 1) - xyz + in.at(x, y, z + 1)) / delta.z;
        //out.set(x, y, z) = xyz - in_prev.at(x, y, z) + tau * tau * lapl;
      }
    }
  }
  // z ( const 0 for global)
  for (int x = 1; x < n.x - 1; ++x) {
    for (int y = 1; y < n.y - 1; ++y) {
      // left
      int z = 0;
      if (are_edges[2] && global_edges) { // left global edge
        out.set(x, y, z) = 0;
      } else if (local_edges) { // local
        //double xyz = 2 * in.at(x, y, z);
        //double lapl = 
        //  (in.at(x - 1, y, z) - xyz + in.at(x + 1, y, z)) / delta.x +
        //  (in.at(x, y - 1, z) - xyz + in.at(x, y + 1, z)) / delta.y +
        //  (in.left_xyz[2][IndexXY(x, y, n)] - xyz + in.at(x, y, z + 1)) / delta.z;
        //out.set(x, y, z) = xyz - in_prev.at(x, y, z) + tau * tau * lapl;
      }
      // right 
      z = n.z - 1;
      if (are_edges[5] && global_edges) { // right global edge
        out.set(x, y, z) = 0;
      } else if (local_edges) { // local
        //double xyz = 2 * in.at(x, y, z);
        //double lapl = 
        //  (in.at(x - 1, y, z) - xyz + in.at(x + 1, y, z)) / delta.x +
        //  (in.at(x, y - 1, z) - xyz + in.at(x, y + 1, z)) / delta.y +
        //  (in.at(x, y, z - 1) - xyz + in.right_xyz[2][IndexXY(x, y, n)]) / delta.z;
        //out.set(x, y, z) = xyz - in_prev.at(x, y, z) + tau * tau * lapl; 
      } 
    }
  }
}

void Calculate3Dimensions(int world_size, int dimensions[3]) {
  switch (world_size) {
    case 1:
      dimensions[0] = 1;
      dimensions[1] = 1;
      dimensions[2] = 1;
      break;
    case 3:
      dimensions[0] = 1;
      dimensions[1] = 1;
      dimensions[2] = 3;
      break;
    case 4:
      dimensions[0] = 2;
      dimensions[1] = 2;
      dimensions[2] = 1;
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

Point<double> Grid3D::PointFromIndices(int i, int j, int k) {
  return Point<double>(p0_.x + i * delta_.x,
                       p0_.y + j * delta_.y,
                       p0_.z + k * delta_.z);
}

void Grid3D::PrintGrid() const {
  /*std::cout << "grid: size " << data_.size() << "\n";
  for (size_t idx = 0; idx < data_.size(); ++idx) {
    if (idx > 0 && idx % num_points_.z == 0) std::cout << "\n";
    if (idx > 0 && idx % (num_points_.z * num_points_.y) == 0) std::cout << "\n";
    std::cout << data_[idx] << " ";
  }*/
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
  /*for (int i = 1; i < num_points_.x - 1; ++i) {
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
  }*/
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

int Grid3D::LinearIndex(int i, int j, int k) const {
  return i * num_points_.y * num_points_.z + j * num_points_.z + k;
}

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

void Grid3D::InitVectors() {
  left_xyz.resize(3);
  right_xyz.resize(3);

  left_xyz[0].resize(num_points_.y * num_points_.z, 0.0);
  left_xyz[1].resize(num_points_.x * num_points_.z, 0.0);
  left_xyz[2].resize(num_points_.x * num_points_.y, 0.0);

  right_xyz[0].resize(num_points_.y * num_points_.z, 0.0);
  right_xyz[1].resize(num_points_.x * num_points_.z, 0.0);
  right_xyz[2].resize(num_points_.x * num_points_.y, 0.0);
}

} // namespace grid