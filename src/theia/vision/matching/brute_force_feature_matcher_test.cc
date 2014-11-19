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

#include <Eigen/Core>
#include <vector>

#include "theia/vision/matching/brute_force_feature_matcher.h"
#include "theia/vision/matching/distance.h"
#include "theia/vision/matching/feature_matcher.h"

#include "gtest/gtest.h"

namespace theia {

using Eigen::VectorXf;

static constexpr int kNumDescriptors = 10;
static constexpr int kNumDescriptorDimensions = 10;

TEST(BruteForceFeatureMatcherTest, NoOptions) {
  // Set up descriptors.
  std::vector<VectorXf> desc1(kNumDescriptors);
  std::vector<VectorXf> desc2(kNumDescriptors);
  for (int i = 0; i < kNumDescriptors; i++) {
    desc1[i] = VectorXf::Constant(kNumDescriptorDimensions, i);
    desc2[i] = VectorXf::Constant(kNumDescriptorDimensions, i);
  }

  // Set options.
  FeatureMatcherOptions options;
  options.keep_only_symmetric_matches = false;
  options.use_lowes_ratio = false;

  // Match features.
  BruteForceFeatureMatcher<L2> matcher;
  std::vector<FeatureMatch> matches;
  matcher.Match(options, desc1, desc2, &matches);

  // Check that the results are valid.
  EXPECT_EQ(matches.size(), kNumDescriptors);
  for (const FeatureMatch& match : matches) {
    EXPECT_EQ(match.feature1_ind, match.feature2_ind);
    EXPECT_EQ(match.distance, 0);
  }
}

TEST(BruteForceFeatureMatcherTest, Threshold) {
  // Set up descriptors.
  std::vector<VectorXf> desc1(kNumDescriptors);
  std::vector<VectorXf> desc2(kNumDescriptors);
  for (int i = 0; i < kNumDescriptors; i++) {
    desc1[i] = VectorXf::Constant(kNumDescriptorDimensions, 10 * i);
    desc2[i] = VectorXf::Constant(kNumDescriptorDimensions, 10 * i);
  }

  // Set one descriptor to be far away so that the max matching threshold is
  // met.
  desc2[9] = VectorXf::Constant(kNumDescriptorDimensions, 5e8);

  // Set options.
  FeatureMatcherOptions options;
  options.keep_only_symmetric_matches = false;
  options.use_lowes_ratio = false;
  options.max_match_distance = 1.0;

  // Match features.
  BruteForceFeatureMatcher<L2> matcher;
  std::vector<FeatureMatch> matches;
  matcher.Match(options, desc1, desc2, &matches);

  // Check that the results are valid.
  EXPECT_EQ(matches.size(), kNumDescriptors - 1);
  for (const FeatureMatch& match : matches) {
    EXPECT_EQ(match.feature1_ind, match.feature2_ind);
    EXPECT_EQ(match.distance, 0);
  }
}

TEST(BruteForceFeatureMatcherTest, RatioTest) {
  // Set up descriptors.
  std::vector<VectorXf> desc1(kNumDescriptors);
  std::vector<VectorXf> desc2(kNumDescriptors);
  for (int i = 0; i < kNumDescriptors; i++) {
    desc1[i] = VectorXf::Constant(kNumDescriptorDimensions, 10 * i);
    desc2[i] = VectorXf::Constant(kNumDescriptorDimensions, 10 * i + 1);
  }

  desc2.push_back(VectorXf::Constant(kNumDescriptorDimensions, 1.01));

  // Set options.
  FeatureMatcherOptions options;
  options.keep_only_symmetric_matches = false;
  options.use_lowes_ratio = true;

  // Match features.
  BruteForceFeatureMatcher<L2> matcher;
  std::vector<FeatureMatch> matches;
  matcher.Match(options, desc1, desc2, &matches);

  // Check that the results are valid.
  EXPECT_EQ(matches.size(), kNumDescriptors - 1);
  for (const FeatureMatch& match : matches) {
    EXPECT_EQ(match.feature1_ind, match.feature2_ind);
    EXPECT_EQ(match.distance, 10);
  }
}

TEST(BruteForceFeatureMatcherTest, SymmetricMatches) {
  // Set up descriptors.
  std::vector<VectorXf> desc1(kNumDescriptors);
  std::vector<VectorXf> desc2(kNumDescriptors);
  for (int i = 0; i < kNumDescriptors; i++) {
    desc1[i] = VectorXf::Constant(kNumDescriptorDimensions, 10 * i);
    desc2[i] = VectorXf::Constant(kNumDescriptorDimensions, 10 * i + 1);
  }

  desc2[0] = VectorXf::Constant(kNumDescriptorDimensions, 5.1);

  // Set options.
  FeatureMatcherOptions options;
  options.keep_only_symmetric_matches = true;
  options.use_lowes_ratio = false;

  // Match features.
  BruteForceFeatureMatcher<L2> matcher;
  std::vector<FeatureMatch> matches;
  matcher.Match(options, desc1, desc2, &matches);

  // Check that the results are valid.
  EXPECT_EQ(matches.size(), kNumDescriptors - 1);
  for (const FeatureMatch& match : matches) {
    EXPECT_EQ(match.feature1_ind, match.feature2_ind);
    EXPECT_EQ(match.distance, 10);
  }
}

TEST(BruteForceFeatureMatcherTest, RatioAndSymmetricMatches) {
  // Set up descriptors.
  std::vector<VectorXf> desc1(kNumDescriptors);
  std::vector<VectorXf> desc2(kNumDescriptors);
  for (int i = 0; i < kNumDescriptors; i++) {
    desc1[i] = VectorXf::Constant(kNumDescriptorDimensions, 10 * i);
    desc2[i] = VectorXf::Constant(kNumDescriptorDimensions, 10 * i + 1);
  }

  desc2[0] = VectorXf::Constant(kNumDescriptorDimensions, 5.1);
  desc2.push_back(VectorXf::Constant(kNumDescriptorDimensions, 11.01));

  // Set options.
  FeatureMatcherOptions options;
  options.keep_only_symmetric_matches = true;
  options.use_lowes_ratio = true;

  // Match features.
  BruteForceFeatureMatcher<L2> matcher;
  std::vector<FeatureMatch> matches;
  matcher.Match(options, desc1, desc2, &matches);

  // Check that the results are valid.
  EXPECT_EQ(matches.size(), kNumDescriptors - 2);
  for (const FeatureMatch& match : matches) {
    EXPECT_EQ(match.feature1_ind, match.feature2_ind);
    EXPECT_EQ(match.distance, 10);
  }
}

}  // namespace theia
