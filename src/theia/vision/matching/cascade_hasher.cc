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

#include "theia/vision/matching/cascade_hasher.h"

#include <Eigen/Core>
#include <glog/logging.h>
#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <utility>
#include <vector>

#include "theia/vision/matching/feature_matcher.h"

namespace theia {

namespace {

// based on Box-Muller transform; for more details, please refer to the
// following WIKIPEDIA website:
// http://en.wikipedia.org/wiki/Box_Muller_transform
double GetNormRand() {
  const double u1 = (rand() % 1000 + 1) / 1000.0;
  const double u2 = (rand() % 1000 + 1) / 1000.0;
  const double randval = sqrt(-2 * log(u1)) * cos(2 * acos(-1.0) * u2);

  return randval;
}

void ConvertToZeroMeanDescriptor(
    const std::vector<Eigen::VectorXf>& sift_desc,
    std::vector<HashedSiftDescriptor>* zero_mean_desc) {
  // Convert to integers.
  zero_mean_desc->resize(sift_desc.size());
  Vector128f mean = Vector128f::Zero();
  for (int i = 0; i < sift_desc.size(); i++) {
    zero_mean_desc->at(i).sift_desc = (sift_desc[i] * 255.0);
    mean += zero_mean_desc->at(i).sift_desc;
  }
  mean /= sift_desc.size();

  // Adjust the descriptors to be zero-mean.
  for (int i = 0; i < zero_mean_desc->size(); i++) {
    zero_mean_desc->at(i).sift_desc -= mean;
  }
}

}  // namespace

bool CascadeHasher::Initialize() {
  // Initialize primary hash projection.
  for (int i = 0; i < kDimHashData; i++) {
    for (int j = 0; j < 128; j++) {
      primary_hash_projection[i][j] = GetNormRand() * 1000;
    }
  }

  // Initialize secondary hash projection.
  for (int i = 0; i < kNumBucketGroups; i++) {
    for (int j = 0; j < kNumBucketBits; j++) {
      for (int k = 0; k < 128; k++) {
        secondary_hash_projection[i][j][k] = GetNormRand() * 1000;
      }
    }
  }

  return true;
}

void CascadeHasher::CreateHashedDescriptors(HashedImage* hashed_image) const {
  for (int i = 0; i < hashed_image->hashed_desc.size(); i++) {
    // Get references for cleanliness.
    const auto& sift = hashed_image->hashed_desc[i].sift_desc;
    auto& hash_code = hashed_image->hashed_desc[i].hash_code;

    // Compute hash code.
    for (int j = 0; j < kDimHashData; j++) {
      const float sum = sift.dot(primary_hash_projection[j]);
      hash_code[j] = sum > 0;
    }

    // Determine the bucket index for each group.
    for (int j = 0; j < kNumBucketGroups; j++) {
      uint16_t bucket_id = 0;
      for (int k = 0; k < kNumBucketBits; k++) {
        const float sum = sift.dot(secondary_hash_projection[j][k]);
        bucket_id = (bucket_id << 1) + (sum > 0 ? 1 : 0);
      }
      hashed_image->hashed_desc[i].bucket_ids[j] = bucket_id;
    }
  }
}

void CascadeHasher::BuildBuckets(HashedImage* hashed_image) const {
  hashed_image->buckets.resize(kNumBucketGroups);
  for (int i = 0; i < kNumBucketGroups; i++) {
    hashed_image->buckets[i].resize(kNumBucketsPerGroup);

    // Add the sift ID to the proper bucket group and id.
    for (int j = 0; j < hashed_image->hashed_desc.size(); j++) {
      const uint16_t bucket_id = hashed_image->hashed_desc[j].bucket_ids[i];
      hashed_image->buckets[i][bucket_id].push_back(j);
    }
  }
}

// Steps:
//   1) Convert SIFT to int SIFT and adjust features to be zero-mean.
//   2) Compute hash and comp hash codes.
//   3) Construct buckets.
void CascadeHasher::CreateHashedSiftDescriptors(
    const std::vector<Eigen::VectorXf>& sift_desc,
    HashedImage* hashed_image) const {
  CHECK_NOTNULL(hashed_image);

  ConvertToZeroMeanDescriptor(sift_desc, &hashed_image->hashed_desc);

  // Allocate space for hash codes and bucket ids.
  hashed_image->hashed_desc.resize(sift_desc.size());

  // Allocate space for each bucket id.
  for (int i = 0; i < sift_desc.size(); i++) {
    hashed_image->hashed_desc[i].bucket_ids.resize(kNumBucketGroups);
  }

  // Create hash codes for each feature.
  CreateHashedDescriptors(hashed_image);

  // Build the buckets.
  BuildBuckets(hashed_image);
}

// Matches images with a fast matching scheme based on the hash codes
// previously generated.
void CascadeHasher::MatchImages(const HashedImage& hashed_image1,
                                const HashedImage& hashed_image2,
                                const double lowes_ratio,
                                std::vector<FeatureMatch>* matches) const {
  const int kNumTopCandidates = 10;

  for (int i = 0; i < hashed_image1.hashed_desc.size(); i++) {
    const auto& hashed_desc = hashed_image1.hashed_desc[i];
    CHECK_GT(hashed_desc.bucket_ids.size(), 0);

    // Accumulate all descriptors in each bucket group that are in the same
    // bucket id as the query descriptor.
    std::unordered_set<int> candidate_descriptors;
    for (int j = 0; j < kNumBucketGroups; j++) {
      const int bucket_id = hashed_desc.bucket_ids[j];
      for (const auto& feature_id : hashed_image2.buckets[j][bucket_id]) {
        candidate_descriptors.insert(feature_id);
      }
    }

    // Skip matching this descriptor if there are not at least 2 candidates.
    if (candidate_descriptors.size() <= 2) {
      continue;
    }

    const int num_nearest_neighbors = std::min(
        static_cast<int>(candidate_descriptors.size()), kNumTopCandidates);

    // Compute the hamming distance of all candidates based on the comp hash
    // code.
    std::vector<std::pair<int, int> > candidate_hamming_distances;
    candidate_hamming_distances.reserve(candidate_descriptors.size());
    for (const int candidate_id : candidate_descriptors) {
      const int hamming_distance =
          (hashed_desc.hash_code ^
           hashed_image2.hashed_desc[candidate_id].hash_code).count();
      candidate_hamming_distances.emplace_back(hamming_distance, candidate_id);
    }

    // Sort the candidates based on hamming distance and choose the top k
    // candidates.
    std::partial_sort(
        candidate_hamming_distances.begin(),
        candidate_hamming_distances.begin() + num_nearest_neighbors,
        candidate_hamming_distances.end());

    // Compute the euclidean distance of each candidate.
    std::vector<std::pair<float, int> > candidate_euclidean_distances;
    candidate_euclidean_distances.reserve(num_nearest_neighbors);
    for (int j = 0; j < num_nearest_neighbors; j++) {
      const int candidate_id = candidate_hamming_distances[j].second;
      const float distance =
          (hashed_image2.hashed_desc[candidate_id].sift_desc -
           hashed_desc.sift_desc).squaredNorm();
      candidate_euclidean_distances.emplace_back(distance, candidate_id);
    }

    // Find the top 2 candidates.
    std::partial_sort(candidate_euclidean_distances.begin(),
                      candidate_euclidean_distances.begin() + 2,
                      candidate_euclidean_distances.end());

    // Only add to output matches if it passes the ratio test.
    if (candidate_euclidean_distances[0].first >
        candidate_euclidean_distances[1].first * lowes_ratio) {
      continue;
    }

    matches->emplace_back(i,
                          candidate_euclidean_distances[0].second,
                          candidate_euclidean_distances[0].first);
  }
}


}  // namespace theia
