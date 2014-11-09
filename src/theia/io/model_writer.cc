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

#include "theia/io/model_writer.h"

#include <Eigen/Core>
#include <glog/logging.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>   // NOLINT
#include <iostream>  // NOLINT
#include <string>

#include "theia/vision/sfm/camera/camera.h"
#include "theia/vision/sfm/model.h"
#include "theia/vision/sfm/track.h"
#include "theia/vision/sfm/types.h"
#include "theia/vision/sfm/view.h"

namespace theia {

namespace {

int NumEstimatedViews(const Model& model) {
  int num_estimated_views = 0;
  for (const ViewId view_id : model.ViewIds()) {
    if (model.View(view_id)->IsEstimated()) {
      ++num_estimated_views;
    }
  }
  return num_estimated_views;
}

int NumEstimatedTracks(const Model& model) {
  int num_estimated_tracks = 0;
  for (const TrackId track_id : model.TrackIds()) {
    if (model.Track(track_id)->IsEstimated()) {
      ++num_estimated_tracks;
    }
  }
  return num_estimated_tracks;
}

bool WriteView(const View view, std::ofstream* output_writer) {
  // Write name.
  const int name_length = view.Name().length();
  output_writer->write(reinterpret_cast<const char*>(&name_length),
                       sizeof(name_length));
  output_writer->write(view.Name().c_str(), name_length);

  // Write id.
  const ViewId& view_id = view.Id();
  output_writer->write(reinterpret_cast<const char*>(&view_id),
                       sizeof(view_id));

  // Write metadata.
  const ViewMetadata& metadata = view.Metadata();
  output_writer->write(reinterpret_cast<const char*>(&metadata),
                       sizeof(metadata));

  // Write camera.
  static const int kCameraParametersSize = 13;
  const Camera& camera = view.Camera();
  output_writer->write(reinterpret_cast<const char*>(camera.extrinsics()),
                       kCameraParametersSize * sizeof(double));
  const int image_width = camera.ImageWidth();
  const int image_height = camera.ImageHeight();
  output_writer->write(reinterpret_cast<const char*>(&image_width),
                       sizeof(image_width));
  output_writer->write(reinterpret_cast<const char*>(&image_height),
                       sizeof(image_height));
  return true;
}

int NumEstimatedViewsInTrack(const Model& model, const Track& track) {
  int num_estimated_views = 0;
  for (const ViewId view_id : track.ViewIds()) {
    const View* view = model.View(view_id);
    if (view != nullptr && view->IsEstimated()) {
      ++num_estimated_views;
    }
  }
  return num_estimated_views;
}

bool WriteTrack(const Model& model,
                const TrackId track_id,
                std::ofstream* output_writer) {
  const Track& track = *model.Track(track_id);

  // Write track id.
  output_writer->write(reinterpret_cast<const char*>(&track_id),
                       sizeof(track_id));

  const int num_estimated_views = NumEstimatedViewsInTrack(model, track);
  output_writer->write(reinterpret_cast<const char*>(&num_estimated_views),
                       sizeof(num_estimated_views));

  // Write features that make up this track. Only include the estimated views.
  for (const ViewId view_id : track.ViewIds()) {
    const View* view = model.View(view_id);
    if (view == nullptr || !view->IsEstimated()) {
      continue;
    }

    // Write view id.
    output_writer->write(reinterpret_cast<const char*>(&view_id),
                         sizeof(view_id));

    // Write features.
    const Feature* feature = CHECK_NOTNULL(view->GetFeature(track_id));
    output_writer->write(reinterpret_cast<const char*>(&feature->x),
                         sizeof(feature->x));
    output_writer->write(reinterpret_cast<const char*>(&feature->x),
                         sizeof(feature->y));
  }

  // Point
  output_writer->write(reinterpret_cast<const char*>(&track.Point()),
                       sizeof(track.Point()));
  return true;
}

}  // namespace

bool WriteModel(const Model& model, const std::string& output_file) {
  std::ofstream output_writer(output_file, std::ios::out | std::ios::binary);
  if (!output_writer.is_open()) {
    LOG(ERROR) << "Could not open the file: " << output_file << " for writing.";
    return false;
  }

  // Write views.
  const int num_estimated_views = NumEstimatedViews(model);
  output_writer.write(reinterpret_cast<const char*>(&num_estimated_views),
                      sizeof(num_estimated_views));

  int i = 0;
  for (const ViewId view_id : model.ViewIds()) {
    const View& view = *model.View(view_id);
    if (view.IsEstimated()) {
      CHECK(WriteView(view, &output_writer)) << "Could not write view id "
                                             << view_id;
    }

    if ((i + 1) % 100 == 0 || i == num_estimated_views - 1) {
      std::cout << "\r Writing parameters for view " << i + 1 << " / "
                << num_estimated_views << std::flush;
    }
    ++i;
  }
  std::cout << std::endl;

  // Write tracks.
  int num_estimated_tracks = NumEstimatedTracks(model);
  output_writer.write(reinterpret_cast<const char*>(&num_estimated_tracks),
                      sizeof(num_estimated_tracks));

  i = 0;
  for (const TrackId track_id : model.TrackIds()) {
    const Track& track = *model.Track(track_id);
    if (track.IsEstimated()) {
      CHECK(WriteTrack(model, track_id, &output_writer))
          << "Could not write track id " << track_id;
    }

    if ((i + 1) % 100 == 0 || i == num_estimated_tracks - 1) {
      std::cout << "\r Writing parameters for track " << i + 1 << " / "
                << num_estimated_tracks << std::flush;
    }
    ++i;
  }
  std::cout << std::endl;

  return true;
}

}  // namespace theia
