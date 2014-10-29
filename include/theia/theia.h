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

#ifndef THEIA_THEIA_H_
#define THEIA_THEIA_H_

#include "theia/alignment/alignment.h"
#include "theia/image/descriptor/brief_descriptor.h"
#include "theia/image/descriptor/brisk_descriptor.h"
#include "theia/image/descriptor/descriptor_extractor.h"
#include "theia/image/descriptor/freak_descriptor.h"
#include "theia/image/descriptor/sift_descriptor.h"
#include "theia/image/image.h"
#include "theia/image/image_canvas.h"
#include "theia/image/keypoint_detector/agast_detector.h"
#include "theia/image/keypoint_detector/brisk_detector.h"
#include "theia/image/keypoint_detector/brisk_impl.h"
#include "theia/image/keypoint_detector/keypoint.h"
#include "theia/image/keypoint_detector/keypoint_detector.h"
#include "theia/image/keypoint_detector/sift_detector.h"
#include "theia/io/bundler_binary_file.h"
#include "theia/io/bundler_text_file.h"
#include "theia/io/sift_binary_file.h"
#include "theia/io/sift_text_file.h"
#include "theia/math/closed_form_polynomial_solver.h"
#include "theia/math/distribution.h"
#include "theia/math/graph/connected_components.h"
#include "theia/math/matrix/gauss_jordan.h"
#include "theia/math/matrix/rq_decomposition.h"
#include "theia/math/polynomial.h"
#include "theia/math/probability/sequential_probability_ratio.h"
#include "theia/math/sturm_chain.h"
#include "theia/math/util.h"
#include "theia/solvers/arrsac.h"
#include "theia/solvers/estimator.h"
#include "theia/solvers/evsac.h"
#include "theia/solvers/evsac_sampler.h"
#include "theia/solvers/inlier_support.h"
#include "theia/solvers/mle_quality_measurement.h"
#include "theia/solvers/mlesac.h"
#include "theia/solvers/prosac.h"
#include "theia/solvers/prosac_sampler.h"
#include "theia/solvers/quality_measurement.h"
#include "theia/solvers/random_sampler.h"
#include "theia/solvers/ransac.h"
#include "theia/solvers/sample_consensus_estimator.h"
#include "theia/solvers/sampler.h"
#include "theia/util/filesystem.h"
#include "theia/util/hash.h"
#include "theia/util/map_util.h"
#include "theia/util/random.h"
#include "theia/util/stringprintf.h"
#include "theia/util/util.h"
#include "theia/vision/matching/brute_force_feature_matcher.h"
#include "theia/vision/matching/cascade_hasher.h"
#include "theia/vision/matching/cascade_hashing_feature_matcher.h"
#include "theia/vision/matching/distance.h"
#include "theia/vision/matching/feature_matcher.h"
#include "theia/vision/matching/feature_matcher_utils.h"
#include "theia/vision/matching/feature_matcher_options.h"
#include "theia/vision/sfm/camera/camera.h"
#include "theia/vision/sfm/camera/project_point_to_image.h"
#include "theia/vision/sfm/camera/projection_matrix_utils.h"
#include "theia/vision/sfm/camera/radial_distortion.h"
#include "theia/vision/sfm/estimators/essential_matrix_estimator.h"
#include "theia/vision/sfm/estimators/fundamental_matrix_estimator.h"
#include "theia/vision/sfm/feature.h"
#include "theia/vision/sfm/feature_correspondence.h"
#include "theia/vision/sfm/model.h"
#include "theia/vision/sfm/pose/dls_impl.h"
#include "theia/vision/sfm/pose/dls_pnp.h"
#include "theia/vision/sfm/pose/eight_point_fundamental_matrix.h"
#include "theia/vision/sfm/pose/essential_matrix_utils.h"
#include "theia/vision/sfm/pose/estimate_global_rotations.h"
#include "theia/vision/sfm/pose/five_point_focal_length_radial_distortion.h"
#include "theia/vision/sfm/pose/five_point_relative_pose.h"
#include "theia/vision/sfm/pose/four_point_focal_length.h"
#include "theia/vision/sfm/pose/four_point_focal_length_helper.h"
#include "theia/vision/sfm/pose/four_point_homography.h"
#include "theia/vision/sfm/pose/four_point_relative_pose_partial_rotation.h"
#include "theia/vision/sfm/pose/fundamental_matrix_util.h"
#include "theia/vision/sfm/pose/perspective_three_point.h"
#include "theia/vision/sfm/pose/three_point_relative_pose_partial_rotation.h"
#include "theia/vision/sfm/pose/test_util.h"
#include "theia/vision/sfm/pose/util.h"
#include "theia/vision/sfm/track.h"
#include "theia/vision/sfm/track_builder.h"
#include "theia/vision/sfm/transformation/align_point_clouds.h"
#include "theia/vision/sfm/transformation/gdls_similarity_transform.h"
#include "theia/vision/sfm/triangulation/triangulation.h"
#include "theia/vision/sfm/twoview_info.h"
#include "theia/vision/sfm/types.h"
#include "theia/vision/sfm/view.h"
#include "theia/vision/sfm/view_graph/view_graph.h"
#include "theia/vision/sfm/view_metadata.h"

#endif  // THEIA_THEIA_H_
