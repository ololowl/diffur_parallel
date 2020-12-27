#define _USE_MATH_DEFINES

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include <mpi.h>
#include <omp.h>

// c++ 11
// #include <random>

#include "grid.h"

const int NUM_TREADS = omp_get_max_threads();

namespace {

double CalculateError(const Grid3D& actual, const Grid3D& expected) {
  Grid3D tmp = actual - expected;
  tmp.ApplyAbs();
  return tmp.Max();
}

void PrintError(int t, double error) {
  std::stringstream ss;
  ss << "Iteration: " << t << ", error: " << std::setprecision(9) << error << "\n";
  std::cout << ss.str();
}

int IndexYZ(int y, int z, const Point<int>& n) {
  return y * n.z + z;
}

int IndexXZ(int x, int z, const Point<int>& n) {
  return x * n.z + z;
}

int IndexXY(int x, int y, const Point<int>& n) {
  return x * n.y + y;
}

} // namespace

// sin(2 * pi * x / X + 3 * pi) * sin(2 * pi * y / Y + 2 * pi) * sin(pi * z / Z) *
// cos(pi * sqrt(4 / (X * X) + 4 / (Y * Y) + 1 / (Z * Z)) * t + pi)
double AnalyticalU(const Point<double>& p, double t, const Point<double>& pL) {
  using std::sin;
  Point<double> coef(2 * M_PI, 2 * M_PI, M_PI);
  Point<double> addition(3 * M_PI, 2 * M_PI, 0);
  Point<double> insin = coef * p / pL + addition;
  double at = M_PI * std::sqrt(4 / (pL.x * pL.x) + 4 / (pL.y * pL.y) +
                               1 / (pL.z * pL.z));
  return sin(insin.x) * sin(insin.y) * sin(insin.z) * std::cos(at * t + M_PI);
}

// - sin(2 * pi * x / X + 3 * pi) * sin(2 * pi * y / Y + 2 * pi) * sin(pi * z / Z) *
// sin(pi * sqrt(4 / (X * X) + 4 / (Y * Y) + 1 / (Z * Z)) * t + pi) * 
// pi * sqrt(4 / (X * X) + 4 / (Y * Y) + 1 / (Z * Z))
double AnalyticalUDerivT(const Point<double>& p, double t,
                         const Point<double>& pL) {
  using std::sin;
  Point<double> coef(2 * M_PI, 2 * M_PI, M_PI);
  Point<double> addition(3 * M_PI, 2 * M_PI, 0);
  Point<double> insin = coef * p / pL + addition;
  double at = M_PI * std::sqrt(4 / (pL.x * pL.x) + 4 / (pL.y * pL.y) +
                               1 / (pL.z * pL.z));
  return - at * sin(insin.x) * sin(insin.y) * sin(insin.z) * sin(at * t + M_PI);
}

Grid3D CreateGridEtalon(const Point<double>& p0, const Point<double>& pN,
                        const Point<int>& n, const Point<double>& pL, double t,
                        bool are_edges[6], const Point<double>& delta) {
  Grid3D grid(p0, pN, n, delta);
  omp_set_num_threads(NUM_TREADS);
  #pragma omp parallel for
  for (int i = 1; i < n.x - 1; ++i) {
    for (int j = 1; j < n.y - 1; ++j) {
      for (int k = 1; k < n.z - 1; ++k) {
        grid.set(i, j, k) = AnalyticalU(grid.PointFromIndices(i, j, k), t, pL);
      }
    }
  }

  #pragma omp parallel for
  for (int y = 0; y < n.y; ++y) {
    for (int z = 0; z < n.z; ++z) {
      Point<double> zero = grid.PointFromIndices(0, y, z);
      zero.x = delta.x;
      Point<double> last = zero;
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
  #pragma omp parallel for
  for (int x = 0; x < n.x; ++x) {
    for (int z = 0; z < n.z; ++z) {
      Point<double> zero = grid.PointFromIndices(x, 0, z);
      zero.y = delta.y;
      Point<double> last = zero;
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
  #pragma omp parallel for
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

Grid3D CreateGridT0(const Point<double> p0, const Point<double> pN,
                    const Point<int> n, const Point<double> pL,
                    bool are_edges[6], const Point<double> delta) {
  return CreateGridEtalon(p0, pN, n, pL, /*t=*/0, are_edges, delta);
}

// in and out are required to be the same in terms of all fields except data_.
void Laplassian7Points(const Grid3D& in, Grid3D& out) {
  const Point<double> delta = in.delta() * in.delta();
  const Point<int> n = in.size();

  omp_set_num_threads(NUM_TREADS);
  #pragma omp parallel for
  for (int x = 1; x < n.x - 1; ++x) {
    for (int y = 1; y < n.y - 1; ++y) {
      for (int z = 1; z < n.z - 1; ++z) {
        const double xyz = 2 * in.at(x, y, z);
        out.set(x, y, z) = (in.at(x-1, y, z) - xyz + in.at(x+1, y, z)) / delta.x +
                           (in.at(x, y-1, z) - xyz + in.at(x, y+1, z)) / delta.y +
                           (in.at(x, y, z-1) - xyz + in.at(x, y, z+1)) / delta.z;
      }
    }
  }
}

// Process edges. For global edges: periodical law for x & y (equality of
// values and derivatives), constant 0 for z. For local just laplassian etc.
// *_neghbours --> [x, y, z]
void UpdateEdges(const Grid3D& in_prev, const Grid3D& in, Grid3D& out, double tau,
                 int left_neighbours[3], int right_neighbours[3], bool are_edges[6],
                 int coords3d[3], int dimensions[3], const MPI_Comm& cartComm);

// Calculates edges only for corresponding edges of {global_edges, local_edges}
// set to true.
void CalculateEdges(const Grid3D& in_prev, const Grid3D& in, Grid3D& out,
                    bool are_edges[6], double tau, bool global_edges,
                    bool local_edges);

// Applies laplassian operator to t0. Edges should be updated by caller.
Grid3D CreateGridT1(const Grid3D& grid_t0, double tau) {
  Grid3D grid(grid_t0);
  Grid3D laplassian(grid_t0.p0(), grid_t0.pN(), grid_t0.size(), grid_t0.delta());

  Laplassian7Points(grid, laplassian);

  return (grid + laplassian * (tau * tau / 2));
}

// arguments: u_{n-1}, u_n, u_{n+1}
// edges have to be processed by caller afterwards
void Step(const Grid3D& in_prev, const Grid3D& in, Grid3D& out, double tau) {
  Laplassian7Points(in, out);
  out *= (tau * tau);
  out += in * 2.;
  out -= in_prev;
}

void Calculate3Dimensions(int world_size, int dimensions[3]);

int main(int argc, char* argv[]) {
  // PARAMETER
  double L = 1.0;
  int grid_axis_size = 128; // points along one axis

  const Point<double> pL(L, L, L); // global borders

  MPI_Init(&argc, &argv);
  omp_set_num_threads(NUM_TREADS);
  
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
    ss << "num proc " << world_size << ", axis size " << grid_axis_size <<
          ", num threads " << NUM_TREADS << "\n";
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
  Point<double> delta(L / (grid_axis_size - 1),
                      L / (grid_axis_size - 1),
                      L / (grid_axis_size - 1));
  Point<int> local_grid_num_points(grid_axis_size / dimensions[0],
                                   grid_axis_size / dimensions[1],
                                   grid_axis_size / dimensions[2]);
  Point<int> mod(grid_axis_size % dimensions[0],
                 grid_axis_size % dimensions[1],
                 grid_axis_size % dimensions[2]);

  Point<int> p0_point(coords3d[0] * local_grid_num_points.x,
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

  Point<double> p0(p0_point.x * delta.x,
                   p0_point.y * delta.y,
                   p0_point.z * delta.z);
  
  Point<int> edges_subtract(1 ? are_edges[3] : 0,
                            1 ? are_edges[4] : 0,
                            1 ? are_edges[5] : 0);
  Point<int> pN_point = p0_point + local_grid_num_points - edges_subtract;
  Point<double> pN(pN_point.x * delta.x,
                   pN_point.y * delta.y,
                   pN_point.z * delta.z);

  // Create grid for t=0
  Grid3D grid_t0 = CreateGridT0(p0, pN, local_grid_num_points, pL, are_edges,
                                delta);
  // Initialize tau & t info
  const int num_time_points = 20;
  std::vector<double> errors(num_time_points, 0);
  double tau = grid_t0.delta().Min();
  tau = tau * tau / 10;

  // Create grid for t=1 and calculate error with etalon
  Grid3D grid_t1 = CreateGridT1(grid_t0, tau);
  // general formula for updates inside u(n+1) = tau^2 * laplassian + 2u(n) - u(n-1).
  // so if u(n) == u(n-1) == grid_t0, tau = tau / sqrt(2), we get
  // u1 = tau^2 / 2 * laplassian + u0
  UpdateEdges(grid_t0, grid_t0, grid_t1, tau / std::sqrt(2), left_neighbours,
              right_neighbours, are_edges, coords3d, dimensions, cartComm);
  errors[1] = CalculateError(
    CreateGridEtalon(p0, pN, local_grid_num_points, pL, tau, are_edges,
                     delta), grid_t1);

  Grid3D grid_t2(p0, pN, local_grid_num_points, delta);

  Grid3D* in_prev = &grid_t0;
  Grid3D* in = &grid_t1;
  Grid3D* out = &grid_t2;

  for (int t = 2; t < num_time_points; ++t) {
    Step(*in_prev, *in, *out, tau);
    UpdateEdges(*in_prev, *in, *out, tau, left_neighbours, right_neighbours,
                are_edges, coords3d, dimensions, cartComm);
    errors[t] = CalculateError(
      CreateGridEtalon(p0, pN, local_grid_num_points, pL, tau * t, are_edges,
                       delta), *out);
    Grid3D* tmp = in_prev;
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
  MPI_Finalize();
  return 0;
}

void UpdateEdges(const Grid3D& in_prev, const Grid3D& in, Grid3D& out, double tau,
                 int left_neighbours[3], int right_neighbours[3], bool are_edges[6],
                 int coords3d[3], int dimensions[3], const MPI_Comm& cartComm) {
  const Point<int> n = out.size();
  
  std::vector<MPI_Request> send_requests(6); // left first, right last
  std::vector<MPI_Request> recv_requests(6); // left first, right last

  omp_set_num_threads(NUM_TREADS);

  // first calculate local edges, as we already have enough info (they use info
  // only from in)
  CalculateEdges(in_prev, in, out, are_edges, tau, /*global_edges=*/false,
                 /*local_edges=*/true);

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
  #pragma omp parallel for
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
  #pragma omp parallel for
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
  #pragma omp parallel for
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

void CalculateEdges(const Grid3D& in_prev, const Grid3D& in, Grid3D& out,
                    bool are_edges[6], double tau, bool global_edges,
                    bool local_edges) {
  const Point<int> n = out.size();
  // x
  const Point<double> delta = in.delta() * in.delta();
  
  #pragma omp parallel for
  for (int y = 1; y < n.y - 1; ++y) {
    for (int z = 1; z < n.z - 1; ++z) {
      // left
      int x = 0;
      if (are_edges[0] && global_edges) { // left global edge
        out.set(x, y, z) = (out.left_xyz[0][IndexYZ(y, z, n)] + out.at(x + 1, y, z)) / 2;
      } else if (local_edges) { // local
        double xyz = 2 * in.at(x, y, z);
        double lapl = 
          (in.left_xyz[0][IndexYZ(y, z, n)] - xyz + in.at(x + 1, y, z)) / delta.x +
          (in.at(x, y - 1, z) - xyz + in.at(x, y + 1, z)) / delta.y +
          (in.at(x, y, z - 1) - xyz + in.at(x, y, z + 1)) / delta.z;
        out.set(x, y, z) = xyz - in_prev.at(x, y, z) + tau * tau * lapl;
      }
      // right
      x = n.x - 1;
      if (are_edges[3] && global_edges) { // right global edge
        out.set(x, y, z) = (out.right_xyz[0][IndexYZ(y, z, n)] + out.at(x - 1, y, z)) / 2;
      } else if (local_edges) { // local
        double xyz = 2 * in.at(x, y, z);
        double lapl = 
          (in.at(x - 1, y, z) - xyz + in.right_xyz[0][IndexYZ(y, z, n)]) / delta.x +
          (in.at(x, y - 1, z) - xyz + in.at(x, y + 1, z)) / delta.y +
          (in.at(x, y, z - 1) - xyz + in.at(x, y, z + 1)) / delta.z;
        out.set(x, y, z) = xyz - in_prev.at(x, y, z) + tau * tau * lapl;
      }
    }
  }
  // y
  #pragma omp parallel for
  for (int x = 1; x < n.x - 1; ++x) {
    for (int z = 1; z < n.z - 1; ++z) {
      // left
      int y = 0;
      if (are_edges[1] && global_edges) { // left global edge
        out.set(x, y, z) = (out.left_xyz[1][IndexXZ(x, z, n)] + out.at(x, y + 1, z)) / 2;
      } else if (local_edges) { // local
        double xyz = 2 * in.at(x, y, z);
        double lapl = 
          (in.at(x - 1, y, z) - xyz + in.at(x + 1, y, z)) / delta.x +
          (in.left_xyz[1][IndexXZ(x, z, n)] - xyz + in.at(x, y + 1, z)) / delta.y +
          (in.at(x, y, z - 1) - xyz + in.at(x, y, z + 1)) / delta.z;
        out.set(x, y, z) = xyz - in_prev.at(x, y, z) + tau * tau * lapl;
      }
      // right
      y = n.y - 1;
      if (are_edges[4] && global_edges) { // right global edge
        out.set(x, y, z) = (out.right_xyz[1][IndexXZ(x, z, n)] + out.at(x, y - 1, z)) / 2;
      } else if (local_edges) { // local
        double xyz = 2 * in.at(x, y, z);
        double lapl = 
          (in.at(x - 1, y, z) - xyz + in.at(x + 1, y, z)) / delta.x +
          (in.at(x, y - 1, z) - xyz + in.right_xyz[1][IndexXZ(x, z, n)]) / delta.y +
          (in.at(x, y, z - 1) - xyz + in.at(x, y, z + 1)) / delta.z;
        out.set(x, y, z) = xyz - in_prev.at(x, y, z) + tau * tau * lapl;
      }
    }
  }
  // z ( const 0 for global)
  #pragma omp parallel for
  for (int x = 1; x < n.x - 1; ++x) {
    for (int y = 1; y < n.y - 1; ++y) {
      // left
      int z = 0;
      if (are_edges[2] && global_edges) { // left global edge
        out.set(x, y, z) = 0;
      } else if (local_edges) { // local
        double xyz = 2 * in.at(x, y, z);
        double lapl = 
          (in.at(x - 1, y, z) - xyz + in.at(x + 1, y, z)) / delta.x +
          (in.at(x, y - 1, z) - xyz + in.at(x, y + 1, z)) / delta.y +
          (in.left_xyz[2][IndexXY(x, y, n)] - xyz + in.at(x, y, z + 1)) / delta.z;
        out.set(x, y, z) = xyz - in_prev.at(x, y, z) + tau * tau * lapl;
      }
      // right 
      z = n.z - 1;
      if (are_edges[5] && global_edges) { // right global edge
        out.set(x, y, z) = 0;
      } else if (local_edges) { // local
        double xyz = 2 * in.at(x, y, z);
        double lapl = 
          (in.at(x - 1, y, z) - xyz + in.at(x + 1, y, z)) / delta.x +
          (in.at(x, y - 1, z) - xyz + in.at(x, y + 1, z)) / delta.y +
          (in.at(x, y, z - 1) - xyz + in.right_xyz[2][IndexXY(x, y, n)]) / delta.z;
        out.set(x, y, z) = xyz - in_prev.at(x, y, z) + tau * tau * lapl; 
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


/*
  std::stringstream neighbours;
  neighbours << "rank " << rank << ", left " << left_neighbours[0] << " ";
  neighbours << left_neighbours[1] << " " << left_neighbours[2] << ", right ";
  neighbours << right_neighbours[0] << " " << right_neighbours[1] << " ";
  neighbours << right_neighbours[2] << "\n";
  std::cout << neighbours.str();

double AnalyticalUSimple(Point p, double t, Point pL) {
  using std::sin;
  double sinx = 2 * M_PI * p.x / pL.x + 3 * M_PI;
  double siny = 2 * M_PI * p.y / pL.y + 2 * M_PI;
  double sinz = M_PI * p.z / pL.z;
  double at = M_PI * std::sqrt(4 / (pL.x * pL.x) + 4 / (pL.y * pL.y) +
                               1 / (pL.z * pL.z));
  return sin(sinx) * sin(siny) * sin(sinz) * std::cos(at * t + M_PI);
}

double AnalyticalUDerivTSimple(Point p, double t, Point pL) {
  using std::sin;
  double sinx = 2 * M_PI * p.x / pL.x + 3 * M_PI;
  double siny = 2 * M_PI * p.y / pL.y + 2 * M_PI;
  double sinz = M_PI * p.z / pL.z;
  double at = M_PI * std::sqrt(4 / (pL.x * pL.x) + 4 / (pL.y * pL.y) +
                               1 / (pL.z * pL.z));
  return - M_PI * at * sin(sinx) * sin(siny) * sin(sinz) * sin(at * t + M_PI);
}

void StressTest(double border) {
    std::mt19937 gen(1543); //Standard mersenne_twister_engine seeded with 1543
    std::uniform_real_distribution<> dis(0, border);
    for (int n = 0; n < 1000; ++n) {
        Point p(dis(gen), dis(gen), dis(gen));
        double t = dis(gen);
        Point pL(dis(gen), dis(gen), dis(gen));
        double actual = AnalyticalUDerivT(p, t, pL);
        double expected = AnalyticalUDerivTSimple(p, t, pL);
        if (std::fabs(actual - expected) > 1e-5) {
          std::cout << "for point ";
          p.Print();
          std::cout << " t " << t << " pL ";
          pL.Print();
          std::cout << " actual (" << actual << ") != expected (" << expected << ")\n";
        }
    }
    std::cout << "SUCCESS\n";
}
*/