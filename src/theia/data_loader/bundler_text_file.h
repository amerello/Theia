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

#ifndef THEIA_DATA_LOADER_BUNDLER_TEXT_FILE_H_
#define THEIA_DATA_LOADER_BUNDLER_TEXT_FILE_H_

#include <Eigen/Core>
#include <string>
#include <vector>

#include "theia/vision/sfm/camera/camera.h"
#include "theia/image/keypoint_detector/keypoint.h"

namespace theia {

// Utility struct to help contain the view list information that Bundle files
// provide.
struct BundleSiftKeyReference {
  BundleSiftKeyReference(const int ci, const int ski, const float xp,
                         const float yp)
      : camera_index(ci), sift_key_index(ski), x_pos(xp), y_pos(yp) {}
  BundleSiftKeyReference() : BundleSiftKeyReference(0, 0, 0, 0) {}
  int camera_index;
  int sift_key_index;
  float x_pos;
  float y_pos;
};

// A set of sift key reference is referred to as a BundleViewList.
typedef std::vector<BundleSiftKeyReference> BundleViewList;

// Loads the list of image names from the given bundler list file. Sets the
// focal length value to the given EXIF value if provided, and 0 otherwise.
bool ReadListsFile(const std::string& list_filename,
                   std::vector<std::string>* image_name,
                   std::vector<double>* exif_focal_length);

// Reads a SIFT key files as computed by Lowe's SIFT software:
// http://www.cs.ubc.ca/~lowe/keypoints/
//
// The vector keypoint will contain the x and y position, as well as the scale
// and orientation of each feature. This variable may be set to NULL, in which
// case the method ignores the keypoint vector.
bool ReadSiftKeyTextFile(const std::string& sift_key_file,
                         std::vector<Eigen::Vector2d>* feature_position,
                         std::vector<Eigen::VectorXf>* descriptor,
                         std::vector<Keypoint>* keypoint);

// Loads all information from a bundler file. The bundler file includes 3D
// points, camera poses, camera intrinsics, descriptors, and 2D-3D matches. This
// method will not load the descriptors from the sift key files, it will only
// store the "references" to them in the view list.
//
// Input params are as follows:
//   bundler_file: the file output by bundler containing the 3D reconstruction,
//       camera poses, feature locations, and correspondences. Usually the file
//       is named bundle.out
//   camera: A vector of theia::Camera objects that contain pose information,
//       descriptor information, and 2D-3D correspondences. The 3D point ids
//       stored correspond to the position in the world_points vector.
//   world_points: The 3D points from the reconstruction.
//   world_points_color: The RGB color of the world points.
//   view_list: The view list for each 3D point. The view list is a container
//       where each element has a camera index, sift key index, and x,y pos.
bool ReadBundleTextFile(const std::string& bundle_file,
                        std::vector<Camera>* camera,
                        std::vector<Eigen::Vector3d>* world_points,
                        std::vector<Eigen::Vector3f>* world_points_color,
                        std::vector<BundleViewList>* view_list);

}  // namespace theia

#endif  // THEIA_DATA_LOADER_BUNDLER_TEXT_FILE_H_
