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

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <vector>

#include "gtest/gtest.h"

#include "theia/math/util.h"
#include "theia/test/benchmark.h"
#include "theia/util/random.h"
#include "theia/util/util.h"
#include "theia/vision/sfm/feature_correspondence.h"
#include "theia/vision/sfm/triangulation/triangulation.h"
#include "theia/vision/sfm/pose/test_util.h"

namespace theia {
using Eigen::MatrixXd;
using Eigen::Matrix3d;
using Eigen::Quaterniond;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;

double ReprojectionError(const Matrix3x4d& pose,
                         const Vector4d& world_point,
                         const Vector2d& image_point) {
  const Vector3d reprojected_point = pose * world_point;
  const double sq_reproj_error =
      (reprojected_point.hnormalized() - image_point).squaredNorm();
  return sq_reproj_error;
}

void TestTriangulationDLTBasic(const Vector3d& point_3d,
                               const Quaterniond& rel_rotation,
                               const Vector3d& rel_translation,
                               const double projection_noise,
                               const double max_reprojection_error) {
  InitRandomGenerator();

  const Matrix3x4d pose_left = Matrix3x4d::Identity();
  Matrix3x4d pose_right;
  pose_right << rel_rotation.toRotationMatrix(), rel_translation;

  // Reproject point into both image 2, assume image 1 is identity rotation at
  // the origin.
  Vector2d image_point_1 =
      (pose_left * point_3d.homogeneous()).eval().hnormalized();
  Vector2d image_point_2 =
      (pose_right * point_3d.homogeneous()).eval().hnormalized();

  // Add projection noise if required.
  if (projection_noise) {
    AddNoiseToProjection(projection_noise, &image_point_1);
    AddNoiseToProjection(projection_noise, &image_point_2);
  }

  // Triangulate with DLT.
  Vector4d dlt_triangulated_point;
  EXPECT_TRUE(
      TriangulateDLT(pose_left.matrix(), pose_right.matrix(), image_point_1,
                     image_point_2, &dlt_triangulated_point));

  // Check the reprojection error.
  EXPECT_LE(
      ReprojectionError(pose_left, dlt_triangulated_point, image_point_1),
      max_reprojection_error);
  EXPECT_LE(
      ReprojectionError(pose_right, dlt_triangulated_point, image_point_2),
      max_reprojection_error);
}

void TestTriangulationBasic(const Vector3d& point_3d,
                            const Quaterniond& rel_rotation,
                            const Vector3d& rel_translation,
                            const double projection_noise,
                            const double max_reprojection_error) {
  InitRandomGenerator();

  Matrix3x4d pose1;
  pose1 <<
      rel_rotation.toRotationMatrix(), rel_translation.normalized();
  const Matrix3x4d pose2 = Matrix3x4d::Identity();

  // Reproject point into both image 2, assume image 1 is identity rotation at
  // the origin.
  Vector2d image_point1 =
      (pose1 * point_3d.homogeneous()).eval().hnormalized();
  Vector2d image_point2 =
      (pose2 * point_3d.homogeneous()).eval().hnormalized();

  // Add projection noise if required.
  if (projection_noise) {
    AddNoiseToProjection(projection_noise, &image_point1);
    AddNoiseToProjection(projection_noise, &image_point2);
  }

  // Triangulate with Optimal.
  Vector4d triangulated_point;
  EXPECT_TRUE(Triangulate(pose1.matrix(), pose2.matrix(), image_point1,
                          image_point2, &triangulated_point));

  // Check the reprojection error.
  EXPECT_LE(
      ReprojectionError(pose1, triangulated_point, image_point1),
      max_reprojection_error);
  EXPECT_LE(
      ReprojectionError(pose2, triangulated_point, image_point2),
      max_reprojection_error);
}

void TestTriangulationManyPoints(const double projection_noise,
                                 const double max_reprojection_error) {
  using Eigen::AngleAxisd;

  static const int num_views = 8;

  // Sets some test rotations and translations.
  static const Quaterniond kRotations[num_views] = {
    Quaterniond(
        AngleAxisd(DegToRad(7.0), Vector3d(0.0, 0.0, 1.0).normalized())),
    Quaterniond(
        AngleAxisd(DegToRad(12.0), Vector3d(0.0, 1.0, 0.0).normalized())),
    Quaterniond(
        AngleAxisd(DegToRad(15.0), Vector3d(1.0, 0.0, 0.0).normalized())),
    Quaterniond(
        AngleAxisd(DegToRad(20.0), Vector3d(1.0, 0.0, 1.0).normalized())),
    Quaterniond(
        AngleAxisd(DegToRad(11.0), Vector3d(0.0, 1.0, 1.0).normalized())),
    Quaterniond(
        AngleAxisd(DegToRad(0.0), Vector3d(1.0, 1.0, 1.0).normalized())),
    Quaterniond(
        AngleAxisd(DegToRad(5.0), Vector3d(0.0, 1.0, 1.0).normalized())),
    Quaterniond(AngleAxisd(DegToRad(0.0), Vector3d(1.0, 1.0, 1.0).normalized()))
  };

  static const Vector3d kTranslations[num_views] = {
    Vector3d(1.0, 1.0, 1.0),
    Vector3d(3.0, 2.0, 13.0),
    Vector3d(4.0, 5.0, 11.0),
    Vector3d(1.0, 2.0, 15.0),
    Vector3d(3.0, 1.5, 91.0),
    Vector3d(1.0, 7.0, 11.0),
    Vector3d(0.0, 0.0, 0.0),  // Tests no translation.
    Vector3d(0.0, 0.0, 0.0)  // Tests no translation and no rotation.
  };

  // Set up model points.
  static const double kTestPoints[][3] = {
    { -1.62, -2.99, 6.12 }, { 4.42, -1.53, 9.83 }, { 1.45, -0.59, 5.29 },
    { 1.89, -1.10, 8.22 }, { -0.21, 2.38, 5.63 }, { 0.61, -0.97, 7.49 },
    { 0.48, 0.70, 8.94 }, { 1.65, -2.56, 8.63 }, { 2.44, -0.20, 7.78 },
    { 2.84, -2.58, 7.35 }, { -1.35, -2.84, 7.33 }, { -0.42, 1.54, 8.86 },
    { 2.56, 1.72, 7.86 }, { 1.75, -1.39, 5.73 }, { 2.08, -3.91, 8.37 },
    { -0.91, 1.36, 9.16 }, { 2.84, 1.54, 8.74 }, { -1.01, 3.02, 8.18 },
    { -3.73, -0.62, 7.81 }, { -2.98, -1.88, 6.23 }, { 2.39, -0.19, 6.47 },
    { -0.63, -1.05, 7.11 }, { -1.76, -0.55, 5.18 }, { -3.19, 3.27, 8.18 },
    { 0.31, -2.77, 7.54 }, { 0.54, -3.77, 9.77 },
  };

  // Set up pose matrices.
  std::vector<Matrix3x4d> poses(num_views);
  for (int i = 0; i < num_views; i++) {
    poses[i] << kRotations[i].toRotationMatrix(), kTranslations[i];
  }

  for (int j = 0; j < ARRAYSIZE(kTestPoints); j++) {
    // Reproject model point into the images.
    std::vector<Vector2d> image_points(num_views);
    const Vector3d model_point(kTestPoints[j][0], kTestPoints[j][1],
                               kTestPoints[j][2]);
    for (int i = 0; i < num_views; i++) {
      image_points[i] =
          (poses[i] * model_point.homogeneous()).eval().hnormalized();
    }

    // Add projection noise if required.
    if (projection_noise) {
      for (int i = 0; i < num_views; i++) {
        AddNoiseToProjection(projection_noise, &image_points[i]);
      }
    }

    Vector4d triangulated_point;
    ASSERT_TRUE(TriangulateNView(poses, image_points, &triangulated_point));

    // Check the reprojection error.
    for (int i = 0; i < num_views; i++) {
      EXPECT_LE(ReprojectionError(poses[i],
                                  triangulated_point, image_points[i]),
                max_reprojection_error);
    }
  }
}

TEST(Triangluation, BasicTest) {
  static const double kProjectionNoise = 0.0;
  static const double kReprojectionTolerance = 1e-12;

  // Set up model points.
  const Vector3d points_3d[2] = { Vector3d(5.0, 20.0, 23.0),
                                  Vector3d(-6.0, 16.0, 33.0) };

  // Set up rotations.
  const Quaterniond kRotation(Eigen::AngleAxisd(0.15, Vector3d(0.0, 1.0, 0.0)));

  // Set up translations.
  const Vector3d kTranslation(-3.0, 1.5, 11.0);

  // Run the test.
  for (int i = 0; i < 2; i++) {
    TestTriangulationDLTBasic(points_3d[i],
                              kRotation,
                              kTranslation,
                              kProjectionNoise,
                              kReprojectionTolerance);

    TestTriangulationBasic(points_3d[i],
                           kRotation,
                           kTranslation,
                           kProjectionNoise,
                           kReprojectionTolerance);
  }
}

TEST(Triangluation, NoiseTest) {
  static const double kProjectionNoise = 1.0 / 512.0;
  static const double kReprojectionTolerance = 1e-5;

  // Set up model points.
  const Vector3d points_3d[2] = { Vector3d(5.0, 20.0, 23.0),
                                  Vector3d(-6.0, 16.0, 33.0) };

  // Set up rotations.
  const Quaterniond kRotation(Eigen::AngleAxisd(0.15, Vector3d(0.0, 1.0, 0.0)));

  // Set up translations.
  const Vector3d kTranslation(-3.0, 1.5, 11.0);

  // Run the test.
  for (int i = 0; i < 2; i++) {
    TestTriangulationDLTBasic(points_3d[i],
                              kRotation,
                              kTranslation,
                              kProjectionNoise,
                              kReprojectionTolerance);

    TestTriangulationBasic(points_3d[i],
                           kRotation,
                           kTranslation,
                           kProjectionNoise,
                           kReprojectionTolerance);
  }
}

TEST(TriangluationNView, BasicTest) {
  static const double kProjectionNoise = 0.0;
  static const double kReprojectionTolerance = 1e-12;

  // Run the test.
  TestTriangulationManyPoints(kProjectionNoise, kReprojectionTolerance);
}

TEST(TriangluationNView, NoiseTest) {
  static const double kProjectionNoise = 1.0 / 512.0;
  static const double kReprojectionTolerance = 5e-4;

  // Run the test.
  TestTriangulationManyPoints(kProjectionNoise, kReprojectionTolerance);
}

BENCHMARK(TriangulationNView, Benchmark, 100, 1000) {
  static const double kProjectionNoise = 0.0;
  static const double kReprojectionTolerance = 1e-12;

  // Run the test.
  TestTriangulationManyPoints(kProjectionNoise, kReprojectionTolerance);
}

void TestIsTriangulatedPointInFrontOfCameras(
    const Eigen::Vector3d& point3d,
    const Eigen::Matrix3d& rotation,
    const Eigen::Vector3d& translation,
    const bool expected_outcome) {
  FeatureCorrespondence correspondence;
  correspondence.feature1 = point3d.hnormalized();
  correspondence.feature2 =
      (rotation * point3d + translation).hnormalized();
  const Vector3d position = -rotation.transpose() * translation;
  EXPECT_EQ(IsTriangulatedPointInFrontOfCameras(
      correspondence,
      rotation,
      position),
            expected_outcome);
}

TEST(IsTriangulatedPointInFrontOfCameras, InFront) {
  const Matrix3d rotation = Matrix3d::Identity();
  const Vector3d position(-1, 0, 0);
  const Vector3d point(0, 0, 5);
  TestIsTriangulatedPointInFrontOfCameras(point, rotation, position, true);
}

TEST(IsTriangulatedPointInFrontOfCameras, Behind) {
  const Matrix3d rotation = Matrix3d::Identity();
  const Vector3d position(-1, 0, 0);
  const Vector3d point(0, 0, -5);
  TestIsTriangulatedPointInFrontOfCameras(point, rotation, position, false);
}

TEST(IsTriangulatedPointInFrontOfCameras, OneInFrontOneBehind) {
  const Matrix3d rotation = Matrix3d::Identity();
  const Vector3d position(0, 0, -2);
  const Vector3d point(0, 0, 1);
  TestIsTriangulatedPointInFrontOfCameras(point, rotation, position, false);
}

}  // namespace theia
