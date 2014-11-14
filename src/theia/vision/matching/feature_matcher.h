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

#ifndef THEIA_VISION_MATCHING_FEATURE_MATCHER_H_
#define THEIA_VISION_MATCHING_FEATURE_MATCHER_H_

#include <vector>

#include "theia/util/util.h"
#include "theia/vision/matching/feature_matcher_options.h"

namespace theia {

class Keypoint;

// Holds feature matches between two images.
struct FeatureMatch {
  FeatureMatch() {}
  FeatureMatch(int f1_ind, int f2_ind, float dist)
      : feature1_ind(f1_ind), feature2_ind(f2_ind), distance(dist) {}

  // Equality operator.
  bool operator==(const FeatureMatch& other) const {
    return (feature1_ind == other.feature2_ind &&
            feature2_ind == other.feature1_ind);
  }

  // Index of the feature in the first image.
  int feature1_ind;
  // Index of the feature in the second image.
  int feature2_ind;
  // Distance between the two features.
  float distance;
};

// Holds all the feature matches between a pair of images.
struct ImagePairMatch {
  int image1_ind;
  int image2_ind;
  std::vector<FeatureMatch> matches;
};

// Class for matching features between images. The matches and match quality
// depend on the options passed to the feature matching.
template <class DistanceMetric> class FeatureMatcher {
 public:
  typedef typename DistanceMetric::DistanceType DistanceType;
  typedef typename DistanceMetric::DescriptorType DescriptorType;

  FeatureMatcher() {}
  virtual ~FeatureMatcher() {}

  // Finds the nearest neighbor in desc_2 for each descriptor in desc_1.
  virtual bool Match(const FeatureMatcherOptions& options,
                     const std::vector<DescriptorType>& desc_1,
                     const std::vector<DescriptorType>& desc_2,
                     std::vector<FeatureMatch>* matches) = 0;

  // Matches all image pairs using num_threads to accelerate matching.
  virtual bool MatchAllPairs(
      const FeatureMatcherOptions& options,
      const int num_threads,
      const std::vector<std::vector<DescriptorType>* >& descriptors,
      std::vector<ImagePairMatch>* image_pair_matches)  = 0;

 private:
  DISALLOW_COPY_AND_ASSIGN(FeatureMatcher);
};

}  // namespace theia

#endif  // THEIA_VISION_MATCHING_FEATURE_MATCHER_H_
