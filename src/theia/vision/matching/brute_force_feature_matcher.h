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

#ifndef THEIA_VISION_MATCHING_BRUTE_FORCE_FEATURE_MATCHER_H_
#define THEIA_VISION_MATCHING_BRUTE_FORCE_FEATURE_MATCHER_H_

#include <Eigen/Core>
#include <glog/logging.h>

#include <algorithm>
#include <mutex>
#include <thread>
#include <vector>

#include "theia/util/threadpool.h"
#include "theia/vision/matching/feature_matcher.h"
#include "theia/vision/matching/feature_matcher_utils.h"

namespace theia {

// Performs features matching between two sets of features using a brute force
// matching method.
template <class DistanceMetric>
class BruteForceFeatureMatcher : public FeatureMatcher<DistanceMetric> {
 public:
  typedef typename DistanceMetric::DistanceType DistanceType;
  typedef typename DistanceMetric::DescriptorType DescriptorType;

  BruteForceFeatureMatcher() {}
  ~BruteForceFeatureMatcher() {}

  // Finds the nearest neighbor in desc_2 for each descriptor in desc_1.
  bool Match(const FeatureMatcherOptions& options,
             const std::vector<DescriptorType>& desc_1,
             const std::vector<DescriptorType>& desc_2,
             std::vector<FeatureMatch>* matches);

   bool MatchAllPairs(
       const FeatureMatcherOptions& options,
       const int num_threads,
       const std::vector<std::vector<DescriptorType>* >& descriptors,
       std::vector<ImagePairMatch>* image_pair_matches);

 private:
  void MatchWithMutex(
    const std::vector<std::vector<DescriptorType>* >&
        descriptors,
    const FeatureMatcherOptions& options,
    const int desc1_index,
    const int desc2_index,
    std::mutex* matcher_mutex,
    std::vector<ImagePairMatch>* image_pair_matches);

  void GetFilteredMatches(const FeatureMatcherOptions& options,
                          const Eigen::MatrixXf& match_distances,
                          std::vector<FeatureMatch>* matches) const;

  DISALLOW_COPY_AND_ASSIGN(BruteForceFeatureMatcher);
};

// ---------------------- Implementation ------------------------ //

template <class DistanceMetric>
bool BruteForceFeatureMatcher<DistanceMetric>::Match(
    const FeatureMatcherOptions& options,
    const std::vector<DescriptorType>& desc_1,
    const std::vector<DescriptorType>& desc_2,
    std::vector<FeatureMatch>* matches) {
  CHECK_NOTNULL(matches)->clear();
  matches->reserve(desc_1.size());

  DistanceMetric distance;

  // Compute all pairwise distances.
  Eigen::MatrixXf match_distances(desc_1.size(), desc_2.size());
  for (int i = 0; i < desc_1.size(); i++) {
    for (int j = 0; j < desc_2.size(); j++) {
      match_distances(i, j) = distance(desc_1[i], desc_2[j]);
    }
  }

  GetFilteredMatches(options, match_distances, matches);

  // Filter by symmetric, if applicable.
  if (options.keep_only_symmetric_matches) {
    std::vector<FeatureMatch> reverse_matches;
    GetFilteredMatches(options, match_distances.transpose(), &reverse_matches);
    IntersectMatches(reverse_matches, matches);
  }
  return true;
}

template <class DistanceMetric>
void BruteForceFeatureMatcher<DistanceMetric>::MatchWithMutex(
    const std::vector<std::vector<DescriptorType>* >&
        descriptors,
    const FeatureMatcherOptions& options,
    const int desc1_index,
    const int desc2_index,
    std::mutex* matcher_mutex,
    std::vector<ImagePairMatch>* image_pair_matches) {
  ImagePairMatch image_pair_match;
  image_pair_match.image1_ind = desc1_index;
  image_pair_match.image2_ind = desc2_index;
  CHECK(Match(options,
              *descriptors[desc1_index],
              *descriptors[desc2_index],
              &image_pair_match.matches));
  // Lock mutex.
  matcher_mutex->lock();
  image_pair_matches->emplace_back(image_pair_match);
  matcher_mutex->unlock();
  VLOG(2) << "Matched images (" << desc1_index << ", " << desc2_index
          << ") in thread " << std::this_thread::get_id();
}

template <class DistanceMetric>
bool BruteForceFeatureMatcher<DistanceMetric>::MatchAllPairs(
    const FeatureMatcherOptions& options,
    const int num_threads,
    const std::vector<std::vector<DescriptorType>* >& descriptors,
    std::vector<ImagePairMatch>* image_pair_matches) {
  CHECK_NOTNULL(image_pair_matches)->clear();
  image_pair_matches->reserve(descriptors.size() * descriptors.size() / 2);

  ThreadPool pool(num_threads);
  std::mutex mutex_lock;
  for (int i = 0; i < descriptors.size(); i++) {
    for (int j = i + 1; j < descriptors.size(); j++) {
      pool.Add(&BruteForceFeatureMatcher::MatchWithMutex,
               this,
               descriptors,
               options,
               i,
               j,
               &mutex_lock,
               image_pair_matches);
    }
  }

  return true;
}

template <class DistanceMetric>
void BruteForceFeatureMatcher<DistanceMetric>::GetFilteredMatches(
    const FeatureMatcherOptions& options,
    const Eigen::MatrixXf& match_distances,
    std::vector<FeatureMatch>* matches) const {
  std::vector<std::vector<FeatureMatch> > knn_matches(match_distances.rows());
  for (int i = 0; i < match_distances.rows(); i++) {
    knn_matches[i].resize(match_distances.cols());
    for (int j = 0; j < match_distances.cols(); j++) {
      knn_matches[i][j] = FeatureMatch(i, j, match_distances(i, j));
    }
  }

  // Find the best matches.
  for (int i = 0; i < knn_matches.size(); i++) {
    std::partial_sort(knn_matches[i].begin(),
                      knn_matches[i].begin() + 2,
                      knn_matches[i].end(),
                      CompareFeaturesByDistance);
  }

  // Filter by Lowes ratio, if applicable.
  if (options.use_lowes_ratio) {
    FilterByLowesRatio(options, knn_matches, matches);
  } else {
    for (int i = 0; i < knn_matches.size(); i++) {
      matches->push_back(knn_matches[i][0]);
    }
  }

  // Filter by max distance.
  RemoveMatchesAboveMaxMatchDistance(options.max_match_distance, matches);
}

}  // namespace theia

#endif  // THEIA_VISION_MATCHING_BRUTE_FORCE_FEATURE_MATCHER_H_
