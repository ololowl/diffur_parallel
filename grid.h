#ifndef GRID_H
#define GRID_H

#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

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

#endif // GRID_H