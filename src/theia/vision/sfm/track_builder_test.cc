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

#include <glog/logging.h>

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "theia/vision/sfm/model.h"
#include "theia/vision/sfm/track.h"
#include "theia/vision/sfm/track_builder.h"
#include "theia/vision/sfm/types.h"

namespace theia {

// Ensure that each track has been added to every view.
void VerifyTracks(const Model& model) {
  std::vector<TrackId> track_ids = model.TrackIds();
  for (const TrackId track_id : track_ids) {
    const Track* track = CHECK_NOTNULL(model.Track(track_id));
    for (const ViewId view_id : track->ViewIds()) {
      const View* view = CHECK_NOTNULL(model.View(view_id));
      EXPECT_TRUE(view->IsTrackVisible(track->Id()));
    }
  }
}

// Perfect tracks.
TEST(TrackBuilder, ConsistentTracks) {
  static const int kMaxTrackLength = 10;
  static const int kNumCorrespondences = 4;

  const std::string image_names[kNumCorrespondences][2] = {
    { "0", "1" }, { "0", "1" }, { "1", "2" }, { "1", "2" }
  };

  TrackBuilder track_builder(kMaxTrackLength);
  for (int i = 0; i < kNumCorrespondences; i++) {
    track_builder.AddFeatureCorrespondence(image_names[i][0],
                                           Feature(kInvalidTrackId, i, i),
                                           image_names[i][1],
                                           Feature(kInvalidTrackId, i, i));
  }

  Model model;
  track_builder.BuildTracks(&model);
  VerifyTracks(model);
  EXPECT_EQ(model.NumTracks(), kNumCorrespondences);
}

// Singleton tracks.
TEST(TrackBuilder, SingletonTracks) {
  // Having a small max track length will force a singleton track.
  static const int kMaxTrackLength = 2;
  static const int kNumCorrespondences = 2;

  const std::string image_names[kNumCorrespondences][2] = {
    { "0", "1" }, { "1", "2" } };

  TrackBuilder track_builder(kMaxTrackLength);
  for (int i = 0; i < kNumCorrespondences; i++) {
    track_builder.AddFeatureCorrespondence(
        image_names[i][0], Feature(kInvalidTrackId, 0, 0),
        image_names[i][1], Feature(kInvalidTrackId, 0, 0));
  }

  Model model;
  track_builder.BuildTracks(&model);
  VerifyTracks(model);
  EXPECT_EQ(model.NumTracks(), 1);
}

// Inconsistent tracks.
TEST(TrackBuilder, InconsistentTracks) {
  static const int kMaxTrackLength = 10;
  static const int kNumCorrespondences = 4;

  const std::string image_names[kNumCorrespondences][2] = {
    { "0", "1" }, { "0", "1" }, { "1", "2" }, { "1", "2" }
  };

  TrackBuilder track_builder(kMaxTrackLength);
  for (int i = 0; i < kNumCorrespondences; i++) {
    track_builder.AddFeatureCorrespondence(
        image_names[i][0], Feature(kInvalidTrackId, 0, 0),
        image_names[i][1], Feature(kInvalidTrackId, i + 1, i + 1));
  }

  Model model;
  track_builder.BuildTracks(&model);
  VerifyTracks(model);
  EXPECT_EQ(model.NumTracks(), 2);
}

// Tracks limited by size.
TEST(TrackBuilder, MaxTrackLength) {
  static const int kMaxTrackLength = 2;
  static const int kNumViews = 6;

  const std::string image_names[kNumViews] = { "0", "1", "2", "3", "4", "5" };

  TrackBuilder track_builder(kMaxTrackLength);
  for (int i = 0; i < kNumViews - 1; i++) {
    track_builder.AddFeatureCorrespondence(
        image_names[i], Feature(kInvalidTrackId, 0, 0),
        image_names[i + 1], Feature(kInvalidTrackId, 0, 0));
  }

  Model model;
  track_builder.BuildTracks(&model);
  VerifyTracks(model);
  EXPECT_EQ(model.NumTracks(), 3);
}

}  // namespace theia
