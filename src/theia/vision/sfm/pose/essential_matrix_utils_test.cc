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
#include <algorithm>

#include "gtest/gtest.h"
#include "theia/vision/sfm/pose/essential_matrix_utils.h"
#include "theia/vision/sfm/pose/test_util.h"
#include "theia/vision/sfm/pose/util.h"

namespace theia {

using Eigen::Matrix3d;
using Eigen::Vector3d;

TEST(DecomposeEssentialMatrix, BasicTest) {
  const double kTranslationTolerance = 1e-6;
  const double kRotationTolerance = 1e-4;

  for (int i = 0; i < 100; i++) {
    const Matrix3d gt_rotation = ProjectToRotationMatrix(Matrix3d::Random());
    const Vector3d gt_translation = Vector3d::Random().normalized();
    const Matrix3d essential_matrix =
        CrossProductMatrix(gt_translation) * gt_rotation;

    Matrix3d rotation1, rotation2;
    Vector3d translation;
    DecomposeEssentialMatrix(essential_matrix,
                             &rotation1,
                             &rotation2,
                             &translation);

    const double translation_dist =
        std::min((translation - gt_translation).norm(),
                 (translation + gt_translation).norm());

    const double rotation1_dist =
        Eigen::AngleAxisd(gt_rotation.transpose() * rotation1).angle();
    const double rotation2_dist =
        Eigen::AngleAxisd(gt_rotation.transpose() * rotation2).angle();

    EXPECT_TRUE(translation_dist < kTranslationTolerance &&
                (rotation1_dist < kRotationTolerance ||
                 rotation2_dist < kRotationTolerance));
  }
}

}  // namespace theia
