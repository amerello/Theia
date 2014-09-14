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

#include "theia/image/descriptor/sift_descriptor.h"

#include <glog/logging.h>
extern "C" {
#include <vl/sift.h>
}
#include <vector>

#include "theia/image/image.h"
#include "theia/image/descriptor/descriptor_extractor.h"
#include "theia/image/keypoint_detector/keypoint.h"

namespace theia {
SiftDescriptorExtractor::~SiftDescriptorExtractor() {
  if (sift_filter_ != nullptr) vl_sift_delete(sift_filter_);
}

bool SiftDescriptorExtractor::ComputeDescriptor(
    const FloatImage& image, const Keypoint& keypoint,
    Eigen::Vector2d* feature_position, Eigen::VectorXf* descriptor) {
  CHECK(keypoint.has_scale() && keypoint.has_orientation())
      << "Keypoint must have scale and orientation to compute a SIFT "
      << "descriptor.";
  // If the filter has been set, but is not usable for the input image (i.e. the
  // width and height are different) then we must make a new filter. Adding this
  // statement will save the function from regenerating the filter for
  // successive calls with images of the same size (e.g. a video sequence).
  if (sift_filter_ != nullptr && (sift_filter_->width != image.Cols() ||
                                  sift_filter_->height != image.Rows()))
    vl_sift_delete(sift_filter_);

  // If the filter has not been set (or was deleted in the previous if), then we
  // need to create a new filter.
  if (sift_filter_ == nullptr)
    sift_filter_ = vl_sift_new(image.Cols(), image.Rows(), num_octaves_,
                               num_levels_, first_octave_);

  // Create the vl sift keypoint from the one passed in.
  VlSiftKeypoint sift_keypoint;
  vl_sift_keypoint_init(sift_filter_, &sift_keypoint, keypoint.x(),
                        keypoint.y(), keypoint.scale());

  // The VLFeat functions take in a non-const image pointer so that it can
  // calculate gaussian pyramids. Obviously, we do not want to break our const
  // input, so the best solution (for now) is to copy the image.
  FloatImage mutable_image = image.AsGrayscaleImage();

  // Calculate the first octave to process.
  int vl_status =
      vl_sift_process_first_octave(sift_filter_, mutable_image.Data());
  // Proceed through the octaves we reach the same one as the keypoint.
  while (sift_keypoint.o != sift_filter_->o_cur)
    vl_sift_process_next_octave(sift_filter_);

  if (vl_status == VL_ERR_EOF)
    LOG(FATAL) << "could not extract sift descriptors";

  // Calculate the sift feature. Note that we are passing in a direct pointer to
  // the descriptor's underlying data.
  *feature_position = Eigen::Vector2d(keypoint.x(), keypoint.y());
  *descriptor = Eigen::VectorXf(128);
  vl_sift_calc_keypoint_descriptor(sift_filter_, descriptor->data(),
                                   &sift_keypoint, keypoint.orientation());
  return true;
}

bool SiftDescriptorExtractor::ComputeDescriptors(
    const FloatImage& image, const std::vector<Keypoint>& keypoints,
    std::vector<Eigen::Vector2d>* feature_positions,
    std::vector<Eigen::VectorXf>* descriptors) {
  // If the filter has been set, but is not usable for the input image (i.e. the
  // width and height are different) then we must make a new filter. Adding this
  // statement will save the function from regenerating the filter for
  // successive calls with images of the same size (e.g. a video sequence).
  if (sift_filter_ != nullptr && (sift_filter_->width != image.Cols() ||
                                  sift_filter_->height != image.Rows()))
    vl_sift_delete(sift_filter_);

  // If the filter has not been set (or was deleted in the previous if), then we
  // need to create a new filter.
  if (sift_filter_ == nullptr)
    sift_filter_ = vl_sift_new(image.Cols(), image.Rows(), num_octaves_,
                               num_levels_, first_octave_);

  // Create the vl sift keypoint from the one passed in.
  std::vector<VlSiftKeypoint> sift_keypoints(keypoints.size());
  for (int i = 0; i < keypoints.size(); i++) {
    CHECK(keypoints[i].has_scale() && keypoints[i].has_orientation())
        << "Keypoint must have scale and orientation to compute a SIFT "
        << "descriptor.";
    vl_sift_keypoint_init(sift_filter_, &sift_keypoints[i], keypoints[i].x(),
                          keypoints[i].y(), keypoints[i].scale());
  }
  // The VLFeat functions take in a non-const image pointer so that it can
  // calculate gaussian pyramids. Obviously, we do not want to break our const
  // input, so the best solution (for now) is to copy the image.
  FloatImage mutable_image = image.AsGrayscaleImage();

  // Calculate the first octave to process.
  int vl_status =
      vl_sift_process_first_octave(sift_filter_, mutable_image.Data());

  // Proceed through the octaves we reach the same one as the keypoint.  We
  // first resize the descriptors vector so that the keypoint indicies will be
  // properly matched to the descriptors.
  descriptors->resize(keypoints.size(), Eigen::VectorXf(128));
  feature_positions->resize(keypoints.size());
  while (vl_status != VL_ERR_EOF) {
    // Go through each keypoint to see if it came from this octave.
    for (int i = 0; i < sift_keypoints.size(); i++) {
      if (sift_keypoints[i].o != sift_filter_->o_cur) continue;
      feature_positions->at(i) =
          Eigen::Vector2d(keypoints[i].x(), keypoints[i].y());
      vl_sift_calc_keypoint_descriptor(
          sift_filter_, (*descriptors)[i].data(), &sift_keypoints[i],
          keypoints[i].orientation());
    }
    vl_status = vl_sift_process_next_octave(sift_filter_);
  }
  return true;
}

bool SiftDescriptorExtractor::DetectAndExtractDescriptors(
    const FloatImage& image,
    std::vector<Eigen::Vector2d>* feature_positions,
    std::vector<Eigen::VectorXf>* descriptors) {
  // If the filter has been set, but is not usable for the input image (i.e. the
  // width and height are different) then we must make a new filter. Adding this
  // statement will save the function from regenerating the filter for
  // successive calls with images of the same size (e.g. a video sequence).
  if (sift_filter_ != nullptr && (sift_filter_->width != image.Cols() ||
                                  sift_filter_->height != image.Rows())) {
    vl_sift_delete(sift_filter_);
  }
  // If the filter has not been set (or was deleted in the previous if), then we
  // need to create a new filter.
  if (sift_filter_ == nullptr)
    sift_filter_ = vl_sift_new(image.Cols(), image.Rows(), num_octaves_,
                               num_levels_, first_octave_);

  // The VLFeat functions take in a non-const image pointer so that it can
  // calculate gaussian pyramids. Obviously, we do not want to break our const
  // input, so the best solution (for now) is to copy the image.
  FloatImage mutable_image = image.AsGrayscaleImage();

  // Calculate the first octave to process.
  int vl_status =
      vl_sift_process_first_octave(sift_filter_, mutable_image.Data());
  // Process octaves until you can't anymore.
  while (vl_status != VL_ERR_EOF) {
    // Detect the keypoints.
    vl_sift_detect(sift_filter_);

    // Get the keypoints.
    const VlSiftKeypoint* vl_keypoints = vl_sift_get_keypoints(sift_filter_);
    int num_keypoints = vl_sift_get_nkeypoints(sift_filter_);

    for (int i = 0; i < num_keypoints; i++) {
      // Calculate (up to 4) orientations of the keypoint.
      double angles[4];
      int num_angles = vl_sift_calc_keypoint_orientations(sift_filter_, angles,
                                                          &vl_keypoints[i]);
      for (int j = 0; j < num_angles; j++) {
        Eigen::VectorXf sift_descriptor(128);
        vl_sift_calc_keypoint_descriptor(
            sift_filter_, sift_descriptor.data(), &vl_keypoints[i],
            angles[j]);
        feature_positions->push_back(
            Eigen::Vector2d(vl_keypoints[i].x, vl_keypoints[i].y));
        descriptors->push_back(sift_descriptor);
      }
    }
    // Attempt to process the next octave.
    vl_status = vl_sift_process_next_octave(sift_filter_);
  }
  return true;
}

}  // namespace theia
