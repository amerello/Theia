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

#include "theia/vision/sfm/pose/essential_matrix_utils.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace theia {

using Eigen::Matrix3d;
using Eigen::Vector3d;

// Decomposes the essential matrix into the rotation R and translation t such
// that E can be any of the four candidate solutions: [rotation1 | translation],
// [rotation1 | -translation], [rotation2 | translation], [rotation2 |
// -translation].
void DecomposeEssentialMatrix(const Matrix3d& essential_matrix,
                              Matrix3d* rotation1,
                              Matrix3d* rotation2,
                              Vector3d* translation) {
  Matrix3d d;
  d << 0, 1, 0,
      -1, 0, 0,
      0, 0, 1;

  const Vector3d& ea = essential_matrix.row(0);
  const Vector3d& eb = essential_matrix.row(1);
  const Vector3d& ec = essential_matrix.row(2);

  // Generate cross products.
  Matrix3d cross_products;
  cross_products << ea.cross(eb), ea.cross(ec), eb.cross(ec);

  // Choose the cross product with the largest norm (for numerical accuracy).
  const Vector3d cf_scales(cross_products.col(0).squaredNorm(),
                           cross_products.col(1).squaredNorm(),
                           cross_products.col(2).squaredNorm());
  int max_index;
  cf_scales.maxCoeff(&max_index);

  // For index 0, 1, we want ea and for index 2 we want eb.
  const int max_e_index = max_index / 2;

  // Construct v of the SVD.
  Matrix3d v = Matrix3d::Zero();
  v.col(2) = cross_products.col(max_index).normalized();
  v.col(0) = essential_matrix.row(max_e_index).normalized();
  v.col(1) = v.col(2).cross(v.col(0));

  // Construct U of the SVD.
  Matrix3d u = Matrix3d::Zero();
  u.col(0) = (essential_matrix * v.col(0)).normalized();
  u.col(1) = (essential_matrix * v.col(1)).normalized();
  u.col(2) = u.col(0).cross(u.col(1));

  // Possible configurations.
  *rotation1 = u * d * v.transpose();
  *rotation2 = u * d.transpose() * v.transpose();
  *translation = u.col(2).normalized();
}

}  // namespace theia
