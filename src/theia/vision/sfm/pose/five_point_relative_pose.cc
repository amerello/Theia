// Copyright (C) 2013 The Regents of the University of California (Regents).
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
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

#include "theia/vision/sfm/pose/five_point_relative_pose.h"

#include <Eigen/Dense>
#include <glog/logging.h>

#include <cmath>
#include <ctime>
#include <vector>

#include "theia/math/polynomial.h"
#include "theia/vision/sfm/pose/util.h"

namespace theia {

using Eigen::Map;
using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Matrix;
using Eigen::RowVector3d;
using Eigen::RowVector4d;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;

typedef Matrix<double, 3, 3, Eigen::RowMajor> RowMatrix3d;

namespace {

// Multiply two degree one polynomials of variables x, y, z.
// E.g. p1 = a[0]x + a[1]y + a[2]z + a[3]
// x^2 y^2 z^2 xy xz yz x y z 1
Matrix<double, 1, 10> MultiplyDegOnePoly(const RowVector4d& a,
                                         const RowVector4d& b) {
  Matrix<double, 1, 10> output;
  output(0) = a(0) * b(0);
  output(1) = a(1) * b(1);
  output(2) = a(2) * b(2);
  output(3) = a(0) * b(1) + a(1) * b(0);
  output(4) = a(0) * b(2) + a(2) * b(0);
  output(5) = a(1) * b(2) + a(2) * b(1);
  output(6) = a(0) * b(3) + a(3) * b(0);
  output(7) = a(1) * b(3) + a(3) * b(1);
  output(8) = a(2) * b(3) + a(3) * b(2);
  output(9) = a(3) * b(3);
  return output;
}

// Multiply a 2 deg poly (in x, y, z) and a one deg poly.
// x^3 y^3 x^2y xy^2 x^2z x^2 y^2z y^2 xyz xy | z^2x zx x z^2y zy y z^3 z^2 z 1
// NOTE: after the | all are variables along z.
Matrix<double, 1, 20> MultiplyDegTwoDegOnePoly(const Matrix<double, 1, 10>& a,
                                               const RowVector4d& b) {
  Matrix<double, 1, 20> output;
  output(0) = a(0) * b(0);
  output(1) = a(1) * b(1);
  output(2) = a(0) * b(1) + a(3) * b(0);
  output(3) = a(1) * b(0) + a(3) * b(1);
  output(4) = a(0) * b(2) + a(4) * b(0);
  output(5) = a(0) * b(3) + a(6) * b(0);
  output(6) = a(1) * b(2) + a(5) * b(1);
  output(7) = a(1) * b(3) + a(7) * b(1);
  output(8) = a(3) * b(2) + a(4) * b(1) + a(5) * b(0);
  output(9) = a(3) * b(3) + a(6) * b(1) + a(7) * b(0);
  output(10) = a(2) * b(0) + a(4) * b(2);
  output(11) = a(4) * b(3) + a(8) * b(0) + a(6) * b(2);
  output(12) = a(6) * b(3) + a(9) * b(0);
  output(13) = a(2) * b(1) + a(5) * b(2);
  output(14) = a(5) * b(3) + a(8) * b(1) + a(7) * b(2);
  output(15) = a(7) * b(3) + a(9) * b(1);
  output(16) = a(2) * b(2);
  output(17) = a(2) * b(3) + a(8) * b(2);
  output(18) = a(8) * b(3) + a(9) * b(2);
  output(19) = a(9) * b(3);
  return output;
}

// Shorthand for multiplying the Essential matrix with its transpose according
// to Eq. 20 in Nister paper.
Matrix<double, 1, 10> EETranspose(
    const Matrix<double, 9, 4>& null_matrix, int i, int j) {
  return MultiplyDegOnePoly(null_matrix.row(3 * i), null_matrix.row(3 * j)) +
      MultiplyDegOnePoly(null_matrix.row(3 * i + 1),
                         null_matrix.row(3 * j + 1)) +
      MultiplyDegOnePoly(null_matrix.row(3 * i + 2),
                         null_matrix.row(3 * j + 2));
}

// Builds the 10x20 constraint matrix according to Section 3.2.2 of Nister
// paper. Constraints are built based on the singularity of the Essential
// matrix, and the trace equation (Eq. 6). This builds the 10x20 matrix such
// that the columns correspond to: x^3, yx^2, y^2x, y^3, zx^2, zyx, zy^2, z^2x,
// z^2y, z^3, x^2, yx, y^2, zx, zy, z^2, x, y, z, 1.
Matrix<double, 10, 20> BuildConstraintMatrix(
    const Matrix<double, 9, 4>& null_space) {
  Matrix<double, 10, 20> constraint_matrix;
  // Singularity constraint.
  constraint_matrix.row(0) =
      MultiplyDegTwoDegOnePoly(
          MultiplyDegOnePoly(null_space.row(1), null_space.row(5)) -
          MultiplyDegOnePoly(null_space.row(2), null_space.row(4)),
          null_space.row(6)) +
      MultiplyDegTwoDegOnePoly(
          MultiplyDegOnePoly(null_space.row(2), null_space.row(3)) -
          MultiplyDegOnePoly(null_space.row(0), null_space.row(5)),
          null_space.row(7)) +
      MultiplyDegTwoDegOnePoly(
          MultiplyDegOnePoly(null_space.row(0), null_space.row(4)) -
          MultiplyDegOnePoly(null_space.row(1), null_space.row(3)),
          null_space.row(8));

  // Trace Constraint. Only need to compute the upper triangular part of the
  // symmetric polynomial matrix
  Matrix<double, 1, 10> symmetric_poly[3][3];
  symmetric_poly[0][0] = EETranspose(null_space, 0, 0);
  symmetric_poly[1][1] = EETranspose(null_space, 1, 1);
  symmetric_poly[2][2] = EETranspose(null_space, 2, 2);

  Matrix<double, 1, 10> half_trace = 0.5*(symmetric_poly[0][0] +
                                          symmetric_poly[1][1] +
                                          symmetric_poly[2][2]);

  symmetric_poly[0][0] -= half_trace;
  symmetric_poly[1][1] -= half_trace;
  symmetric_poly[2][2] -= half_trace;
  symmetric_poly[0][1] = EETranspose(null_space, 0, 1);
  symmetric_poly[0][2] = EETranspose(null_space, 0, 2);
  symmetric_poly[1][0] = symmetric_poly[0][1];
  symmetric_poly[1][2] = EETranspose(null_space, 1, 2);
  symmetric_poly[2][0] = symmetric_poly[0][2];
  symmetric_poly[2][1] = symmetric_poly[1][2];

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      constraint_matrix.row(3*i + j + 1) =
          MultiplyDegTwoDegOnePoly(symmetric_poly[i][0],
                                   null_space.row(j)) +
          MultiplyDegTwoDegOnePoly(symmetric_poly[i][1],
                                   null_space.row(3 + j)) +
          MultiplyDegTwoDegOnePoly(symmetric_poly[i][2],
                                   null_space.row(6 + j));
    }
  }
  return constraint_matrix;
}

}  // namespace

// Implementation of Nister from "An Efficient Solution to the Five-Point
// Relative Pose Problem"
bool FivePointRelativePose(const Vector2d image1_points[5],
                           const Vector2d image2_points[5],
                           std::vector<Matrix3d>* essential_matrices) {
  // Step 1. Create the 5x9 matrix containing epipolar constraints.
  //   Essential matrix is a linear combination of the 4 vectors spanning the
  //   null space of this matrix (found by SVD).
  Matrix<double, 5, 9> epipolar_constraint;
  for (int i = 0; i < 5; i++) {
    // Fill matrix with the epipolar constraint from q'_t*E*q = 0. Where q is
    // from the first image, and q' is from the second. Eq. 8 in the Nister
    // paper.
    epipolar_constraint.row(i) <<
        image1_points[i].x() * image2_points[i].x(),
        image1_points[i].y() * image2_points[i].x(),
        image2_points[i].x(),
        image1_points[i].x() * image2_points[i].y(),
        image1_points[i].y() * image2_points[i].y(),
        image2_points[i].y(),
        image1_points[i].x(),
        image1_points[i].y(),
        1.0;
  }

  const Eigen::FullPivLU<Matrix<double, 5, 9> > lu(epipolar_constraint);
  if (lu.dimensionOfKernel() != 4) {
    return false;
  }
  const Matrix<double, 9, 4>& null_space = lu.kernel();

  // Step 2. Expansion of the epipolar constraints Eq. 5 and 6 from Nister
  // paper.
  const Matrix<double, 10, 20> constraint_matrix =
      BuildConstraintMatrix(null_space);

  // Step 3. Eliminate part of the matrix to isolate polynomials in z.
  const Matrix<double, 10, 10>& eliminated_matrix =
      constraint_matrix.block<10, 10>(0, 0).partialPivLu()
          .solve(constraint_matrix.block<10, 10>(0, 10));

  // Step 4. Create the matrix B whose elements are polynomials in z.
  VectorXd gj_matrix[3][3];
  for (int i = 0; i < 3; i++) {
    const int row = 2 * i + 4;
    gj_matrix[i][0] =
        Vector4d(-eliminated_matrix(row + 1, 0),
                 eliminated_matrix(row, 0) - eliminated_matrix(row + 1, 1),
                 eliminated_matrix(row, 1) - eliminated_matrix(row + 1, 2),
                 eliminated_matrix(row, 2));
    gj_matrix[i][1] =
        Vector4d(-eliminated_matrix(row + 1, 3),
                 eliminated_matrix(row, 3) - eliminated_matrix(row + 1, 4),
                 eliminated_matrix(row, 4) - eliminated_matrix(row + 1, 5),
                 eliminated_matrix(row, 5));
    gj_matrix[i][2] = VectorXd::Zero(5);
    gj_matrix[i][2][0] = -eliminated_matrix(row + 1, 6);
    gj_matrix[i][2][1] =
        eliminated_matrix(row, 6) - eliminated_matrix(row + 1, 7);
    gj_matrix[i][2][2] =
        eliminated_matrix(row, 7) - eliminated_matrix(row + 1, 8);
    gj_matrix[i][2][3] =
        eliminated_matrix(row, 8) - eliminated_matrix(row + 1, 9);
    gj_matrix[i][2][4] = eliminated_matrix(row, 9);
  }

  const VectorXd cofactor1 =
      MultiplyPolynomials(gj_matrix[0][1], gj_matrix[1][2]) -
      MultiplyPolynomials(gj_matrix[0][2], gj_matrix[1][1]);
  const VectorXd cofactor2 =
      MultiplyPolynomials(gj_matrix[0][2], gj_matrix[1][0]) -
      MultiplyPolynomials(gj_matrix[0][0], gj_matrix[1][2]);
  const VectorXd cofactor3 =
      MultiplyPolynomials(gj_matrix[0][0], gj_matrix[1][1]) -
      MultiplyPolynomials(gj_matrix[0][1], gj_matrix[1][0]);

  // Form determinant of B as a 10th degree polynomial in z.
  const VectorXd determinant_poly =
      MultiplyPolynomials(cofactor1, gj_matrix[2][0]) +
      MultiplyPolynomials(cofactor2, gj_matrix[2][1]) +
      MultiplyPolynomials(cofactor3, gj_matrix[2][2]);

  // Step 5. Extract real roots of the 10th degree polynomial.
  Eigen::VectorXd roots;
  FindRealPolynomialRootsJenkinsTraub(determinant_poly, &roots);

  essential_matrices->reserve(roots.size());
  static const double kTolerance = 1e-12;
  for (int i = 0; i < roots.size(); i++) {
    // We only want non-zero roots
    if (fabs(roots(i)) < kTolerance)
      continue;

    const double x = EvaluatePolynomial(cofactor1, roots(i)) /
                     EvaluatePolynomial(cofactor3, roots(i));
    const double y = EvaluatePolynomial(cofactor2, roots(i)) /
                     EvaluatePolynomial(cofactor3, roots(i));

    Matrix<double, 9, 1> temp_sum =
        x * null_space.col(0) + y * null_space.col(1) +
        roots(i) * null_space.col(2) + null_space.col(3);
    // Need to do it like this because temp_sum is a row vector and recasting
    // it as a 3x3 will load it column-major.
    Matrix3d candidate_essential_mat;
    candidate_essential_mat <<
        temp_sum(0), temp_sum(1), temp_sum(2),
        temp_sum(3), temp_sum(4), temp_sum(5),
        temp_sum(6), temp_sum(7), temp_sum(8),
    essential_matrices->emplace_back(candidate_essential_mat);
  }
  return (roots.size() > 0);
}

}  // namespace theia
