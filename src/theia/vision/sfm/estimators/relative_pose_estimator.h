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

#ifndef THEIA_VISION_SFM_ESTIMATORS_RELATIVE_POSE_ESTIMATOR_H_
#define THEIA_VISION_SFM_ESTIMATORS_RELATIVE_POSE_ESTIMATOR_H_

#include <Eigen/Core>
#include <vector>

#include "theia/solvers/estimator.h"
#include "theia/util/util.h"
#include "theia/vision/sfm/feature_correspondence.h"

namespace theia {

struct RelativePose {
  Eigen::Matrix3d essential_matrix;
  Eigen::Matrix3d rotation;
  Eigen::Vector3d position;
};

// An estimator for computing the relative pose from 5 feature
// correspondences. The feature correspondences should be normalized
// by the focal length with the principal point at (0, 0).
class RelativePoseEstimator
    : public Estimator<FeatureCorrespondence, RelativePose> {
 public:
  RelativePoseEstimator() {}

  // 5 correspondences are needed to determine an essential matrix and thus a
  // relative pose..
  double SampleSize() const { return 5; }

  // Estimates candidate relative poses from correspondences.
  bool EstimateModel(const std::vector<FeatureCorrespondence>& correspondences,
                     std::vector<RelativePose>* essential_matrices) const;

  // The error for a correspondences given a model. This is the squared sampson
  // error.
  double Error(const FeatureCorrespondence& correspondence,
               const RelativePose& essential_matrix) const;

 private:
  DISALLOW_COPY_AND_ASSIGN(RelativePoseEstimator);
};

}  // namespace theia

#endif  // THEIA_VISION_SFM_ESTIMATORS_RELATIVE_POSE_ESTIMATOR_H_
