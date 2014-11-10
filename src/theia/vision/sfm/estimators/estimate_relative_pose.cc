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

#include "theia/vision/sfm/estimators/estimate_relative_pose.h"

#include <memory>
#include <vector>

#include "theia/solvers/sample_consensus_estimator.h"
#include "theia/vision/sfm/create_and_initialize_ransac_variant.h"
#include "theia/vision/sfm/estimators/relative_pose_estimator.h"
#include "theia/vision/sfm/feature_correspondence.h"

namespace theia {

bool EstimateRelativePose(
    const RansacParameters& ransac_params,
    const RansacType& ransac_type,
    const std::vector<FeatureCorrespondence>& normalized_correspondences,
    RelativePose* relative_pose,
    RansacSummary* ransac_summary) {
  RelativePoseEstimator relative_pose_estimator;
  std::unique_ptr<SampleConsensusEstimator<RelativePoseEstimator> > ransac =
      CreateAndInitializeRansacVariant(ransac_type,
                                       ransac_params,
                                       relative_pose_estimator);
  // Estimate essential matrix.
  if (!ransac->Estimate(normalized_correspondences,
                        relative_pose,
                        ransac_summary)) {
    return false;
  }

  return true;
}

}  // namespace theia
