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

#include "theia/io/model_reader.h"

#include <Eigen/Core>
#include <glog/logging.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>   // NOLINT
#include <iostream>  // NOLINT
#include <string>
#include <utility>
#include <vector>

#include "theia/util/map_util.h"
#include "theia/vision/sfm/camera/camera.h"
#include "theia/vision/sfm/model.h"
#include "theia/vision/sfm/track.h"
#include "theia/vision/sfm/types.h"
#include "theia/vision/sfm/view.h"

namespace theia {

namespace {

// Returns the old view id.
ViewId AddViewToModel(Model* model, std::ifstream* input_reader) {
  // Read name.
  int name_length;
  input_reader->read(reinterpret_cast<char*>(&name_length),
                     sizeof(name_length));

  char* buffer = new char[name_length];
  input_reader->read(buffer, name_length);
  const std::string view_name(buffer, name_length);
  delete[] buffer;

  const ViewId view_id = model->AddView(view_name);
  CHECK_NE(view_id, kInvalidViewId) << "Could not add view " << view_name;

  View* view = model->MutableView(view_id);
  view->SetEstimated(true);

  // Read id.
  ViewId old_view_id;
  input_reader->read(reinterpret_cast<char*>(&old_view_id),
                     sizeof(old_view_id));

  // Read metadata.
  input_reader->read(reinterpret_cast<char*>(view->MutableMetadata()),
                     sizeof(*view->MutableMetadata()));

  // Read camera.
  static int kCameraParametersSize = 13;
  Camera* camera = view->MutableCamera();
  input_reader->read(reinterpret_cast<char*>(camera->mutable_extrinsics()),
                     kCameraParametersSize * sizeof(double));
  int image_width, image_height;
  input_reader->read(reinterpret_cast<char*>(&image_width),
                     sizeof(image_width));
  input_reader->read(reinterpret_cast<char*>(&image_height),
                     sizeof(image_height));
  camera->SetImageSize(image_width, image_height);

  return old_view_id;
}

bool AddTrackToModel(const std::unordered_map<ViewId, ViewId>& old_id_to_new_id,
                     Model* model,
                     std::ifstream* input_reader) {
  // Read track id.
  TrackId old_track_id;
  input_reader->read(reinterpret_cast<char*>(&old_track_id),
                     sizeof(old_track_id));

  // Read number of views in the track.
  int num_views;
  input_reader->read(reinterpret_cast<char*>(&num_views), sizeof(num_views));

  // Read features that make up this track.
  std::vector<std::pair<std::string, Feature> > features;
  for (int i = 0; i < num_views; i++) {
    // Read old view id.
    ViewId old_view_id;
    input_reader->read(reinterpret_cast<char*>(&old_view_id),
                       sizeof(old_view_id));

    const ViewId new_view_id =
        FindWithDefault(old_id_to_new_id, old_view_id, kInvalidViewId);
    CHECK_NE(new_view_id, kInvalidViewId)
        << "Tried to add a track containing a view that is not present in the "
           "model. Model file is corrupted!";
    const std::string& view_name = model->View(new_view_id)->Name();

    // Read features.
    Feature feature;
    input_reader->read(reinterpret_cast<char*>(&feature.x), sizeof(feature.x));
    input_reader->read(reinterpret_cast<char*>(&feature.x), sizeof(feature.y));
    features.emplace_back(view_name, feature);
  }

  // Add track to model.
  const TrackId track_id = model->AddTrack(features);

  Track* track = model->MutableTrack(track_id);
  track->SetEstimated(true);

  // Point
  input_reader->read(reinterpret_cast<char*>(track->MutablePoint()),
                     sizeof(*track->MutablePoint()));
  return true;
}

}  // namespace

bool ReadModel(const std::string& input_file, Model* model) {
  CHECK_NOTNULL(model);
  CHECK_EQ(model->NumViews(), 0)
      << "You must provide an empty model before reading a model from disk";
  CHECK_EQ(model->NumTracks(), 0)
      << "You must provide an empty model before reading a model from disk";

  std::ifstream input_reader(input_file, std::ios::in | std::ios::binary);
  if (!input_reader.is_open()) {
    LOG(ERROR) << "Could not open the file: " << input_file << " for writing.";
    return false;
  }

  // Read views.
  int num_views;
  input_reader.read(reinterpret_cast<char*>(&num_views), sizeof(num_views));

  std::unordered_map<ViewId, ViewId> old_id_to_new_id;
  for (int i = 0; i < num_views; i++) {
    const ViewId old_id = AddViewToModel(model, &input_reader);
    old_id_to_new_id[old_id] = i;

    if ((i + 1) % 100 == 0 || i == num_views - 1) {
      std::cout << "\r Loading parameters for view " << i + 1 << " / "
                << num_views << std::flush;
    }
  }
  std::cout << std::endl;

  // Read tracks.
  int num_tracks;
  input_reader.read(reinterpret_cast<char*>(&num_tracks), sizeof(num_tracks));

  for (int i = 0; i < num_tracks; i++) {
    CHECK(AddTrackToModel(old_id_to_new_id, model, &input_reader));
     if ((i + 1) % 100 == 0 || i == num_tracks - 1) {
      std::cout << "\r Loading parameters for track " << i + 1 << " / "
                << num_tracks << std::flush;
    }
  }
  std::cout << std::endl;

  return true;
}

}  // namespace theia
