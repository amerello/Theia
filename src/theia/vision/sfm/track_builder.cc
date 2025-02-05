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

#include "theia/vision/sfm/track_builder.h"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "theia/math/graph/connected_components.h"
#include "theia/util/map_util.h"
#include "theia/vision/sfm/feature.h"
#include "theia/vision/sfm/model.h"

namespace theia {

TrackBuilder::TrackBuilder(const int max_track_length) : num_features_(0) {
  connected_components_.reset(
      new ConnectedComponents<size_t>(max_track_length));
}

TrackBuilder::~TrackBuilder() {}

void TrackBuilder::AddFeatureCorrespondence(const std::string& image_name1,
                                            const Feature& feature1,
                                            const std::string& image_name2,
                                            const Feature& feature2) {
  CHECK_NE(image_name1, image_name2)
      << "Cannot add 2 features from the same image as a correspondence for "
         "track generation.";

  const ImageNameFeaturePair image_feature1 =
      std::make_pair(image_name1, feature1);
  const ImageNameFeaturePair image_feature2 =
      std::make_pair(image_name2, feature2);

  const int feature1_id = FindOrInsert(image_feature1);
  const int feature2_id = FindOrInsert(image_feature2);

  connected_components_->AddEdge(feature1_id, feature2_id);
}

void TrackBuilder::BuildTracks(Model* model) {
  CHECK_NOTNULL(model);
  CHECK_EQ(model->NumTracks(), 0);

  // Build a reverse map mapping feature ids to ImageNameFeaturePairs.
  std::unordered_map<size_t, const ImageNameFeaturePair*> id_to_feature;
  id_to_feature.reserve(features_.size());
  for (const auto& feature : features_) {
    InsertOrDie(&id_to_feature, feature.second, &feature.first);
  }

  // Extract all connected components.
  std::unordered_map<size_t, std::unordered_set<size_t> > components;
  connected_components_->Extract(&components);

  // Each connected component is a track. Add all tracks to the model.
  int num_singleton_tracks = 0;
  int num_inconsistent_features = 0;
  for (const auto& component : components) {
    // Skip singleton tracks.
    if (component.second.size() == 1) {
      ++num_singleton_tracks;
      continue;
    }

    std::vector<std::pair<std::string, Feature> > track;
    track.reserve(component.second.size());

    // Add all features in the connected component to the track.
    std::unordered_set<std::string> image_names;
    for (const auto& feature_id : component.second) {
      const ImageNameFeaturePair& feature_to_add =
          *FindOrDie(id_to_feature, feature_id);

      // Do not add the feature if the track already contains a feature from the
      // same image.
      if (!InsertIfNotPresent(&image_names, feature_to_add.first)) {
        ++num_inconsistent_features;
        continue;
      }

      track.emplace_back(feature_to_add);
    }

    CHECK_NE(model->AddTrack(track), kInvalidTrackId)
        << "Could not build tracks.";
  }

  VLOG(2)
      << model->NumTracks() << " tracks were created. "
      << num_inconsistent_features
      << " features were dropped because they formed inconsistent tracks, and "
      << num_singleton_tracks
      << " features were dropped because they formed singleton tracks.";
}

size_t TrackBuilder::FindOrInsert(const ImageNameFeaturePair& image_feature) {
  const size_t* feature_id = FindOrNull(features_, image_feature);

  // If the feature is present, return the id.
  if (feature_id != nullptr) {
    return *feature_id;
  }

  // Otherwise, add the feature.
  const size_t new_feature_id = num_features_;
  InsertOrDieNoPrint(&features_, image_feature, new_feature_id);

  // Increment the number of features.
  ++num_features_;

  return new_feature_id;;
}

}  // namespace theia
