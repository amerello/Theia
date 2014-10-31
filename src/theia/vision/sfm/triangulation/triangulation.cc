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

#include "theia/vision/sfm/triangulation/triangulation.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <glog/logging.h>
#include <vector>

#include "theia/vision/sfm/feature_correspondence.h"
#include "theia/vision/sfm/pose/fundamental_matrix_util.h"
#include "theia/vision/sfm/pose/util.h"

namespace theia {
namespace {

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Vector2d;
using Eigen::Vector3d;

// Given either a fundamental or essential matrix and two corresponding images
// points such that ematrix * point_right produces a line in the left image,
// this method finds corrected image points such that
// corrected_point_left^t * ematrix * corrected_point_right = 0.
void FindOptimalImagePoints(const Matrix3d& ematrix,
                            const Vector2d& point_left,
                            const Vector2d& point_right,
                            Vector2d* corrected_point_left,
                            Vector2d* corrected_point_right) {
  const Vector3d point1 = point_left.homogeneous();
  const Vector3d point2 = point_right.homogeneous();

  // A helper matrix to isolate certain coordinates.
  Matrix<double, 2, 3> s_matrix;
  s_matrix <<
      1, 0, 0,
      0, 1, 0;

  const Eigen::Matrix2d e_submatrix = ematrix.topLeftCorner<2, 2>();

  // The epipolar line from one image point in the other image.
  Vector2d epipolar_line1 = s_matrix * ematrix * point2;
  Vector2d epipolar_line2 = s_matrix * ematrix.transpose() * point1;

  const double a = epipolar_line1.transpose() * e_submatrix * epipolar_line2;
  const double b =
      (epipolar_line1.squaredNorm() + epipolar_line2.squaredNorm()) / 2.0;
  const double c = point1.transpose() * ematrix * point2;

  const double d = sqrt(b * b - a * c);

  double lambda = c / (b + d);
  epipolar_line1 -= e_submatrix * lambda * epipolar_line1;
  epipolar_line2 -= e_submatrix.transpose() * lambda * epipolar_line2;

  lambda *=
      (2.0 * d) / (epipolar_line1.squaredNorm() + epipolar_line2.squaredNorm());

  *corrected_point_left =
      (point1 - s_matrix.transpose() * lambda * epipolar_line1).hnormalized();
  *corrected_point_right =
      (point2 - s_matrix.transpose() * lambda * epipolar_line2).hnormalized();
}

}  // namespace

// Triangulates 2 posed views
bool Triangulate(const Matrix3x4d& pose1,
                 const Matrix3x4d& pose2,
                 const Vector2d& point_left,
                 const Vector2d& point_right,
                 Vector3d* triangulated_point) {
  Matrix3d fmatrix;
  FundamentalMatrixFromProjectionMatrices(pose1.data(),
                                          pose2.data(),
                                          fmatrix.data());

  Vector2d corrected_point_left, corrected_point_right;
  FindOptimalImagePoints(fmatrix, point_left, point_right,
                         &corrected_point_left, &corrected_point_right);

  // Now the two points are guaranteed to intersect. We can use the DLT method
  // since it is easy to construct.
  return TriangulateDLT(pose1 ,
                        pose2,
                        corrected_point_left,
                        corrected_point_right,
                        triangulated_point);
}

// Triangulates 2 posed views
bool TriangulateDLT(const Matrix3x4d& pose_left,
                    const Matrix3x4d& pose_right,
                    const Vector2d& point_left,
                    const Vector2d& point_right,
                    Vector3d* triangulated_point) {
  Matrix4d design_matrix;
  design_matrix.row(0) = point_left[0] * pose_left.row(2) - pose_left.row(0);
  design_matrix.row(1) = point_left[1] * pose_left.row(2) - pose_left.row(1);
  design_matrix.row(2) = point_right[0] * pose_right.row(2) - pose_right.row(0);
  design_matrix.row(3) = point_right[1] * pose_right.row(2) - pose_right.row(1);

  // Extract nullspace.
  Eigen::Vector4d homog_triangulated_point =
      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
  if (homog_triangulated_point[3] == 0) {
    return false;
  }

  *triangulated_point = homog_triangulated_point.hnormalized();
  return true;
}

// Triangulates N views by computing SVD that minimizes the error.
bool TriangulateNViewSVD(const std::vector<Matrix3x4d>& poses,
                         const std::vector<Vector2d>& points,
                         Vector3d* triangulated_point) {
  CHECK_EQ(poses.size(), points.size());

  MatrixXd design_matrix(3 * points.size(), 4 + points.size());

  for (int i = 0; i < points.size(); i++) {
    design_matrix.block<3, 4>(3 * i, 0) = -poses[i].matrix();
    design_matrix.block<3, 1>(3 * i, 4 + i) = points[i].homogeneous();
  }

  // Computing SVD on A'A is more efficient and gives the same null-space.
  Eigen::Vector4d homog_triangulated_point =
      (design_matrix.transpose() * design_matrix).jacobiSvd(Eigen::ComputeFullV)
          .matrixV().rightCols<1>().head(4);
  if (homog_triangulated_point[3] == 0) {
    return false;
  }

  *triangulated_point = homog_triangulated_point.hnormalized();
  return true;
}

bool TriangulateNView(const std::vector<Matrix3x4d>& poses,
                      const std::vector<Vector2d>& points,
                      Vector3d* triangulated_point) {
  CHECK_EQ(poses.size(), points.size());

  Matrix4d design_matrix = Matrix4d::Zero();
  for (int i = 0; i < points.size(); i++) {
    const Vector3d norm_point = points[i].homogeneous().normalized();
    const Eigen::Matrix<double, 3, 4> cost_term =
        poses[i].matrix() -
        norm_point * norm_point.transpose() * poses[i].matrix();
    design_matrix = design_matrix + cost_term.transpose() * cost_term;
  }

  Eigen::SelfAdjointEigenSolver<Matrix4d> eigen_solver(design_matrix);
  Eigen::Vector4d homog_triangulated_point = eigen_solver.eigenvectors().col(0);
  if (homog_triangulated_point[3] == 0) {
    return false;
  }

  *triangulated_point = homog_triangulated_point.hnormalized();
  return true;
}

bool IsTriangulatedPointInFrontOfCameras(
    const FeatureCorrespondence& correspondence,
    const Matrix3d& rotation,
    const Vector3d& position) {

  const Vector3d dir1 = correspondence.feature1.homogeneous();
  const Vector3d dir2 =
      rotation.transpose() * correspondence.feature2.homogeneous();

  const double dir1_sq = dir1.squaredNorm();
  const double dir2_sq = dir2.squaredNorm();
  const double dir1_dir2 = dir1.dot(dir2);
  const double dir1_pos = dir1.dot(position);
  const double dir2_pos = dir2.dot(position);

  return (dir2_sq * dir1_pos - dir1_dir2 * dir2_pos > 0 &&
          dir1_dir2 * dir1_pos - dir1_sq * dir2_pos > 0);
}

}  // namespace theia
