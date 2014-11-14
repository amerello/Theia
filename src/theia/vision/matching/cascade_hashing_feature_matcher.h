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

#ifndef THEIA_VISION_MATCHING_CASCADE_HASHING_FEATURE_MATCHER_H_
#define THEIA_VISION_MATCHING_CASCADE_HASHING_FEATURE_MATCHER_H_

#include <Eigen/Core>
#include <mutex>
#include <vector>

#include "theia/vision/matching/distance.h"
#include "theia/vision/matching/feature_matcher.h"

namespace theia {

class CascadeHasher;
struct HashedImage;

// Performs features matching between two sets of features using a cascade
// hashing approach. This hashing does not require any training and is extremely
// efficient but can only be used with float features like SIFT.
class CascadeHashingFeatureMatcher : public FeatureMatcher<L2> {
 public:
  CascadeHashingFeatureMatcher() {}
  ~CascadeHashingFeatureMatcher() {}

  // Finds the nearest neighbor in desc_2 for each descriptor in desc_1.
  bool Match(const FeatureMatcherOptions& options,
             const std::vector<Eigen::VectorXf>& desc_1,
             const std::vector<Eigen::VectorXf>& desc_2,
             std::vector<FeatureMatch>* matches);

  bool MatchAllPairs(
      const FeatureMatcherOptions& options,
      const int num_threads,
      const std::vector<std::vector<Eigen::VectorXf>* >& descriptors,
      std::vector<ImagePairMatch>* image_pair_matches);

 private:
  void MatchWithMutex(
      const std::vector<HashedImage*>& descriptors,
      const FeatureMatcherOptions& options,
      const int thread_id,
      const int num_threads,
      std::mutex* matcher_mutex,
      CascadeHasher* hasher,
      std::vector<ImagePairMatch>* image_pair_matches);

  DISALLOW_COPY_AND_ASSIGN(CascadeHashingFeatureMatcher);
};

}  // namespace theia

#endif  // THEIA_VISION_MATCHING_CASCADE_HASHING_FEATURE_MATCHER_H_
