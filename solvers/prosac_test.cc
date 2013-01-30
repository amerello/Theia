// Copyright (C) 2013  Chris Sweeney <cmsweeney@cs.ucsb.edu>
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of the University of California, Santa Barbara nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL CHRIS SWEENEY BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <chrono>
#include <math.h>
#include <random>

#include "gtest/gtest.h"
#include "solvers/estimator.h"
#include "solvers/prosac.h"

namespace solvers {
namespace {
struct Point {
  double x;
  double y;
  Point() {}
  Point(double _x, double _y) : x(_x), y(_y) {}
};

// y = mx + b
struct Line {
  double m;
  double b;
  Line() {}
  Line(double _m, double _b) : m(_m), b(_b) {}
};

class LineEstimator : public Estimator<Point, Line> {
 public:
  LineEstimator() {}
  ~LineEstimator() {}

  bool EstimateModel(const vector<Point>& data, Line* model) const {
    model->m = (data[1].y - data[0].y)/(data[1].x - data[0].x);
    model->b = data[1].y - model->m*data[1].x;
    return true;
  }

  double Error(const Point& point, const Line& line) const {
    double a = -1.0*line.m;
    double b = 1.0;
    double c = -1.0*line.b;

    return fabs(a*point.x + b*point.y + c)/(sqrt(pow(a*a + b*b, 2)));
  }
};

// Returns a random double between dMin and dMax
double RandDouble(double dMin, double dMax) {
  double d = static_cast<double>(rand()) / RAND_MAX;
  return dMin + d * (dMax - dMin);
}
}  // namespace

TEST(ProsacTest, LineFitting) {
  // Create a set of points along y=x with a small random pertubation.
  // construct a trivial random generator engine from a time-based seed:
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<double> gauss_distribution(0.0, 0.5);
  std::normal_distribution<double> small_distribution(0.0, 0.05);
  const int num_points = 10000;
  vector<Point> input_points(num_points);
  vector<double> confidence(num_points);

  for (int i = 0; i < num_points; ++i) {
    if (i < 300) {
      double noise_x = small_distribution(generator);
      double noise_y = small_distribution(generator);
      input_points[i] = Point(i + noise_x, i + noise_y);      
      confidence[i] = 0.95;
    } else {
      double noise_x = gauss_distribution(generator);
      double noise_y = gauss_distribution(generator);
      input_points[i] = Point(i + noise_x, i + noise_y);
      confidence[i] = 0.1;
    }
  }

  LineEstimator line_estimator;
  Line line;
  Prosac<Point, Line> prosac_line(2, 1.0, 800, 100000);
  prosac_line.Estimate(input_points, line_estimator, &line);
  ASSERT_LT(fabs(line.m - 1.0), 0.1);
}

}  // namespace solvers
