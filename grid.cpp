#include "grid.h"

#include <iostream>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

void SetVectorToZeros(std::vector<double>& v) {
  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = 0.;
  }
}

} // namespace

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