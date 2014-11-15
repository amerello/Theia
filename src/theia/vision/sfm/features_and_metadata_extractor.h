
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

#ifndef THEIA_VISION_SFM_FEATURES_AND_METADATA_EXTRACTOR_H_
#define THEIA_VISION_SFM_FEATURES_AND_METADATA_EXTRACTOR_H_

#include <Eigen/Core>
#include <string>
#include <vector>

#include "theia/alignment/alignment.h"
#include "theia/image/descriptor/descriptor_extractor.h"
#include "theia/image/image.h"
#include "theia/image/keypoint_detector/keypoint_detector.h"
#include "theia/util/filesystem.h"
#include "theia/util/threadpool.h"
#include "theia/vision/sfm/view_metadata.h"

namespace theia {

// The various types of keypoint detectors and feature descriptors you can
// choose.
enum class KeypointDetectorType {
  SIFT = 0,
  AGAST = 1,
  BRISK = 2,
};

enum class DescriptorExtractorType {
  SIFT = 0,
  BRIEF = 1,
  BRISK = 2,
  FREAK = 3
};

// Reads in the set of images provided then extracts descriptors and matches
// them. The view metadata (typically EXIF information such as focal length) is
// also output. We use an overloaded function to
class FeaturesAndMetadataExtractor {
 public:
  FeaturesAndMetadataExtractor(
      const KeypointDetectorType& keypoint_detector_type,
      const DescriptorExtractorType& descriptor_extractor_type)
      : keypoint_detector_type_(keypoint_detector_type),
        descriptor_extractor_type_(descriptor_extractor_type) {}
  ~FeaturesAndMetadataExtractor() {}

  // Method to extract descriptors and metadata. Descriptors must be a float
  // descriptor Eigen::VectorXf (e.g., SIFT) or a Eigen::BinaryVectorX.
  template <typename DescriptorType>
  bool Extract(const std::vector<std::string>& filenames,
               const int num_threads,
               std::vector<std::vector<Keypoint>*>* keypoints,
               std::vector<std::vector<DescriptorType>*>* descriptors,
               std::vector<ViewMetadata*>* view_metadata);

 private:
  // Extracts the features and metadata for a single image. This function is
  // called by the threadpool and is thus thread safe.
  template <typename DescriptorType>
  bool ExtractFeaturesAndMetadata(const std::string& filename,
                                  std::vector<Keypoint>* keypoints,
                                  std::vector<DescriptorType>* descriptors,
                                  ViewMetadata* view_metadata);

  // Factory methods to create the keypoint detector and descriptor extractor.
  std::unique_ptr<KeypointDetector> CreateKeypointDetector();
  std::unique_ptr<DescriptorExtractor> CreateDescriptorExtractor();

  const KeypointDetectorType keypoint_detector_type_;
  const DescriptorExtractorType descriptor_extractor_type_;

  DISALLOW_COPY_AND_ASSIGN(FeaturesAndMetadataExtractor);
};

// ---------------------- Implementation ------------------------- //
template <typename DescriptorType>
bool FeaturesAndMetadataExtractor::Extract(
    const std::vector<std::string>& filenames,
    const int num_threads,
    std::vector<std::vector<Keypoint>*>* keypoints,
    std::vector<std::vector<DescriptorType>*>* descriptors,
    std::vector<ViewMetadata*>* view_metadata) {
  CHECK_NOTNULL(keypoints)->resize(filenames.size());
  CHECK_NOTNULL(descriptors)->resize(filenames.size());
  CHECK_NOTNULL(view_metadata)->resize(filenames.size());

  // The thread pool will wait to finish all jobs when it goes out of scope.
  ThreadPool feature_extractor_pool(num_threads);
  for (int i = 0; i < filenames.size(); i++) {
    if (!FileExists(filenames[i])) {
      LOG(ERROR) << "Could not extract features for " << filenames[i]
                 << " because the file cannot be found.";
      continue;
    }

    keypoints->at(i) = new std::vector<Keypoint>();
    descriptors->at(i) = new std::vector<DescriptorType>();
    view_metadata->at(i) = new ViewMetadata;
    feature_extractor_pool.Add(
        &FeaturesAndMetadataExtractor::ExtractFeaturesAndMetadata<
                  DescriptorType>,
        this,
        filenames[i],
        keypoints->at(i),
        descriptors->at(i),
        view_metadata->at(i));
  }
  return true;
}

template <typename DescriptorType>
bool FeaturesAndMetadataExtractor::ExtractFeaturesAndMetadata(
    const std::string& filename,
    std::vector<Keypoint>* keypoints,
    std::vector<DescriptorType>* descriptors,
    ViewMetadata* view_metadata) {
  const FloatImage img(filename);

  // We create these variable here instead of upon the construction of the
  // object so that they can be thread-safe. We *should* be able to use the
  // static thread_local keywords, but apparently Mac OS-X's version of clang
  // does not actually support it!
  std::unique_ptr<KeypointDetector> keypoint_detector =
      CreateKeypointDetector();
  std::unique_ptr<DescriptorExtractor> descriptor_extractor =
      CreateDescriptorExtractor();

  // Exit if the keypoint detection fails.
  if (!keypoint_detector->DetectKeypoints(img, keypoints)) {
    LOG(ERROR) << "Could not detect keypoints in image " << filename;
    return false;
  }

  // Exit if the descriptor extraction fails.
  if (!descriptor_extractor->ComputeDescriptors(img,
                                                keypoints,
                                                descriptors)) {
    LOG(ERROR) << "Could not extract descriptors in image " << filename;
    return false;
  }

  // Only set the focal length if it is present.
  double exif_focal_length;
  if (img.FocalLengthPixels(&exif_focal_length)) {
    view_metadata->focal_length.value = exif_focal_length;
    view_metadata->focal_length.is_set = true;
  }
  view_metadata->image_width = img.Width();
  view_metadata->image_height = img.Height();

  VLOG(2) << "Successfully extracted " << descriptors->size()
          << " features from image " << filename;
  return true;
}


}  // namespace theia

#endif  // THEIA_VISION_SFM_FEATURES_AND_METADATA_EXTRACTOR_H_
