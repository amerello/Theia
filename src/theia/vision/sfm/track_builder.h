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

#ifndef THEIA_VISION_SFM_TRACK_BUILDER_H_
#define THEIA_VISION_SFM_TRACK_BUILDER_H_

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "theia/vision/sfm/feature.h"

namespace theia {

template <typename T> class ConnectedComponents;
class Model;

// Build tracks from feature correspondences across multiple images. Tracks are
// created with the connected components algorithm and have a maximum allowable
// size. If there are multiple features from one image in a track, we do not do
// any intelligent selection and just arbitrarily choose a feature to drop so
// that the tracks are consistent.
class TrackBuilder {
 public:
  typedef std::pair<std::string, Feature> ImageNameFeaturePair;

  explicit TrackBuilder(const int max_track_length);

  ~TrackBuilder();

  // Adds a feature correspondence between two images.
  void AddFeatureCorrespondence(const std::string& image_name1,
                                const Feature& feature1,
                                const std::string& image_name2,
                                const Feature& feature2);

  // Generates all tracks and adds them to the model.
  void BuildTracks(Model* model);

 private:
  size_t FindOrInsert(const ImageNameFeaturePair& image_feature);

  std::unordered_map<ImageNameFeaturePair, size_t> features_;
  std::unique_ptr<ConnectedComponents<size_t> > connected_components_;
  size_t num_features_;
};

}  // namespace theia

#endif  // THEIA_VISION_SFM_TRACK_BUILDER_H_
