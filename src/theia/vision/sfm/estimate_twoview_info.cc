// Copyright (C) 2014 The Regents of the University of California (Regents).
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

#include "theia/vision/sfm/estimate_twoview_info.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glog/logging.h>

#include <vector>

#include "theia/solvers/sample_consensus_estimator.h"
#include "theia/vision/sfm/estimators/estimate_relative_pose.h"
#include "theia/vision/sfm/estimators/estimate_uncalibrated_relative_pose.h"
#include "theia/vision/sfm/estimators/relative_pose_estimator.h"
#include "theia/vision/sfm/estimators/uncalibrated_relative_pose_estimator.h"
#include "theia/vision/sfm/feature_correspondence.h"
#include "theia/vision/sfm/pose/util.h"
#include "theia/vision/sfm/triangulation/triangulation.h"
#include "theia/vision/sfm/twoview_info.h"
#include "theia/vision/sfm/types.h"
#include "theia/vision/sfm/view.h"

namespace theia {

using Eigen::AngleAxisd;
using Eigen::Matrix3d;
using Eigen::Vector2d;
using Eigen::Vector3d;

namespace {

int NumTriangulatedFeatures(
    const std::vector<FeatureCorrespondence>& correspondences,
    const Matrix3d& rotation,
    const Vector3d& position) {
  int num_triangulated_points = 0;
  for (const auto& correspondence : correspondences) {
    if (IsTriangulatedPointInFrontOfCameras(correspondence,
                                            rotation,
                                            position)) {
      ++num_triangulated_points;
    }
  }
  return num_triangulated_points;
}

bool EstimateTwoViewInfoCalibrated(
    const EstimateTwoViewInfoOptions& options,
    const View& view1,
    const View& view2,
    const std::vector<FeatureCorrespondence>& correspondences,
    TwoViewInfo* twoview_info,
    std::vector<int>* inlier_indices) {
  // Normalize features w.r.t focal length.
  std::vector<FeatureCorrespondence> normalized_correspondences;
  NormalizeFeatures(view1.Camera(),
                    view2.Camera(),
                    correspondences,
                    &normalized_correspondences);

  // Set the ransac parameters.
  RansacParameters ransac_options;
  ransac_options.failure_probability = 1.0 - options.expected_ransac_confidence;
  ransac_options.min_iterations = options.min_ransac_iterations;
  ransac_options.max_iterations = options.max_ransac_iterations;
  ransac_options.error_thresh =
      options.max_sampson_error_pixels * options.max_sampson_error_pixels /
      (view1.Metadata().focal_length.value *
       view2.Metadata().focal_length.value);
  ransac_options.use_mle = options.use_mle;

  RelativePose relative_pose;
  RansacSummary summary;
  if (!EstimateRelativePose(ransac_options,
                            options.ransac_type,
                            normalized_correspondences,
                            &relative_pose,
                            &summary)) {
    VLOG(2) << "Could not estimate a relative pose for " << view1.Name()
            << " and " << view2.Name();
    return false;
  }

  AngleAxisd rotation(relative_pose.rotation);

  // Set the twoview info.
  twoview_info->rotation_2 = rotation.angle() * rotation.axis();
  twoview_info->position_2 = relative_pose.position;
  twoview_info->focal_length_1 = view1.Camera().FocalLength();
  twoview_info->focal_length_2 = view1.Camera().FocalLength();
  twoview_info->num_3d_points =
      NumTriangulatedFeatures(normalized_correspondences,
                              relative_pose.rotation,
                              relative_pose.position);
  twoview_info->num_matched_features = summary.inliers.size();

  *inlier_indices = summary.inliers;

  return true;
}

bool EstimateTwoViewInfoUncalibrated(
    const EstimateTwoViewInfoOptions& options,
    const View& view1,
    const View& view2,
    const std::vector<FeatureCorrespondence>& correspondences,
    TwoViewInfo* twoview_info,
    std::vector<int>* inlier_indices) {
  // Normalize features w.r.t principal point.
  std::vector<FeatureCorrespondence> centered_correspondences(
      correspondences.size());
  const Vector2d principal_point1(view1.Camera().PrincipalPointX(),
                                  view1.Camera().PrincipalPointY());
  const Vector2d principal_point2(view1.Camera().PrincipalPointX(),
                                  view1.Camera().PrincipalPointY());
  for (int i = 0; i < correspondences.size(); i++) {
    centered_correspondences[i].feature1 =
        correspondences[i].feature1 - principal_point1;
    centered_correspondences[i].feature2 =
        correspondences[i].feature2 - principal_point2;
  }

  // Set the ransac parameters.
  RansacParameters ransac_options;
  ransac_options.failure_probability = 1.0 - options.expected_ransac_confidence;
  ransac_options.min_iterations = options.min_ransac_iterations;
  ransac_options.max_iterations = options.max_ransac_iterations;
  ransac_options.error_thresh =
      options.max_sampson_error_pixels * options.max_sampson_error_pixels;

  UncalibratedRelativePose relative_pose;
  RansacSummary summary;
  if (!EstimateUncalibratedRelativePose(ransac_options,
                                        options.ransac_type,
                                        centered_correspondences,
                                        &relative_pose,
                                        &summary)) {
    VLOG(2) << "Could not estimate a relative pose for " << view1.Name()
            << " and " << view2.Name();
    return false;
  }

  AngleAxisd rotation(relative_pose.rotation);

  // Set the twoview info.
  twoview_info->rotation_2 = rotation.angle() * rotation.axis();
  twoview_info->position_2 = relative_pose.position;
  twoview_info->focal_length_1 = relative_pose.focal_length1;
  twoview_info->focal_length_2 = relative_pose.focal_length2;

  // Normalize the correspondences by focal length.
  for (int i = 0; i < centered_correspondences.size(); i++) {
    centered_correspondences[i].feature1 /= relative_pose.focal_length1;
    centered_correspondences[i].feature2 /= relative_pose.focal_length2;
  }

  // Get the number of 3d points.
  twoview_info->num_3d_points =
      NumTriangulatedFeatures(centered_correspondences,
                              relative_pose.rotation,
                              relative_pose.position);
  twoview_info->num_matched_features = summary.inliers.size();

  *inlier_indices = summary.inliers;

  return true;
}

}  // namespace

bool EstimateTwoViewInfo(
    const EstimateTwoViewInfoOptions& options,
    const View& view1,
    const View& view2,
    const std::vector<FeatureCorrespondence>& correspondences,
    TwoViewInfo* twoview_info,
    std::vector<int>* inlier_indices) {
  CHECK_NOTNULL(twoview_info);
  CHECK_NOTNULL(inlier_indices)->clear();

  const bool view1_focal_length_set = view1.Metadata().focal_length.is_set;
  const bool view2_focal_length_set = view2.Metadata().focal_length.is_set;

  // Early exit if applicable.
  if (correspondences.size() < options.min_triangulated_points) {
    VLOG(2) << "Insufficient number of correspondences to estimate a two view "
               "model!";
    return false;
  }

  // Case where both views are calibrated.
  if (view1_focal_length_set && view2_focal_length_set) {
    return EstimateTwoViewInfoCalibrated(options,
                                         view1,
                                         view2,
                                         correspondences,
                                         twoview_info,
                                         inlier_indices);
  }

  // Only one of the focal lengths is set.
  if (view1_focal_length_set || view2_focal_length_set) {
    LOG(WARNING) << "Solving for two view infos when exactly one view is "
                    "calibrated has not been implemented yet. Treating both "
                    "views as uncalibrated instead.";
    return EstimateTwoViewInfoCalibrated(options,
                                         view1,
                                         view2,
                                         correspondences,
                                         twoview_info,
                                         inlier_indices);
  }

  // Assume both views are uncalibrated.
  return EstimateTwoViewInfoUncalibrated(options,
                                         view1,
                                         view2,
                                         correspondences,
                                         twoview_info,
                                         inlier_indices);
}

}  // namespace theia
