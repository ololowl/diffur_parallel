#include "grid.h"

#include <iostream>
#include <sstream>

template<typename T>
void CheckEqualPoints(const Point<T>& actual, const Point<T>& expected, std::string header = "") {
  if (actual != expected) {
    std::stringstream ss;
    ss << header << "\nexpected " << expected.DebugString() << ", actual "
       << actual.DebugString();
    std::cerr << ss.str() << "\n";
    throw "";
  }
}

void TestPoint() {
  Point<double> point1(1, 2, 3);
  Point<double> point2(4, 2, 6);

  CheckEqualPoints(point1 + point2, Point<double>(5, 4, 9), "Sum values don't match");
  CheckEqualPoints(point1 - point2, Point<double>(-3, 0, -3), "Sub values don't match");
  CheckEqualPoints(point1 * point2, Point<double>(4, 4, 18), "Mul values don't match");
  CheckEqualPoints(point1 / point2, Point<double>(0.25, 1, 0.5), "Div values don't match");

  CheckEqualPoints(point1 + 2., Point<double>(3, 4, 5), "Sum with num values don't match");
  CheckEqualPoints(point1 - 2., Point<double>(-1, 0, 1), "Sub with num values don't match");
  CheckEqualPoints(point1 * 2., Point<double>(2, 4, 6), "Mul with num values don't match");
  CheckEqualPoints(point1 / 2., Point<double>(0.5, 1, 1.5), "Div with num values don't match");

  std::cerr << "Point test passed!\n";
}

void TestGrid() {
  Point<double> p0(2, 3, 5), pN(5, 4, 8);
  Grid3D grid(p0, pN, 3);
  
  double t = 1;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        grid.set(i, j, k) = t;
        ++t;
      }
    }
  }

  std::vector<double> expected;
  double sum = 0;
  for (int i = 1; i < 28; ++i) {
    expected.push_back(i);
    sum += i;
  }

  if (grid.data() != expected) {
    std::cerr << "Data values are not equal\n" << grid.DebugString() << "\n";
    throw "";
  }
  if (grid.Sum() != sum) {
    std::cerr << "Sum of grid doesn't match expected\n";
    throw "";
  }
  if (grid.Max() != 27) {
    std::cerr << "Max of grid doesn't match expected\n";
    throw "";
  }

  for (int i = 0; i < 3; i += 2) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; k += 2) {
        grid.set(i, j, k) = - grid.at(i, j, k);
        ++t;
      }
    }
  }
  grid.ApplyAbs();
  if (grid.data() != expected) {
    std::cerr << "Data values are not equal\n" << grid.DebugString() << "\n";
    throw "";
  }

  Point<double> actual = grid.PointFromIndices(1, 2, 0);
  if (actual != Point<double>(3.5, 4, 5)) {
    std::cerr << "PointFromIndices not correct\n";
    std::cerr << "actual " << actual.DebugString() << "\n";
    std::cerr << "expected {x: 3.5, y: 4, z: 5}\n";
    throw "";
  }

  std::cerr << "Grid test passed!\n";
}

int main() {
  TestPoint();
  TestGrid();
  return 0;
}