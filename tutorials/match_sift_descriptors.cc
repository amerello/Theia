// Copyright (C) 2013 The Regents of the University of California (Regents).
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
#include <gflags/gflags.h>
#include <time.h>
#include <theia/theia.h>
#include <chrono>
#include <string>
#include <vector>

DEFINE_string(
    input_imgs, "",
    "Filepath of the images you want to extract features and compute matches "
    "for. The filepath should be a wildcard to match multiple images.");
DEFINE_int32(num_threads, 1,
             "Number of threads to use for feature extraction and matching.");
DEFINE_string(img_output_dir, ".", "Name of output image file.");

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  // Get image filenames.
  std::vector<std::string> img_filepaths;
  CHECK(theia::GetFilepathsFromWildcard(FLAGS_input_imgs, &img_filepaths));

  // Load images and extract features
  const int num_images = img_filepaths.size();
  std::vector<theia::FloatImage> images(num_images);
  std::vector<std::vector<theia::Keypoint> > keypoints(num_images);
  std::vector<std::vector<Eigen::VectorXf> > descriptors(num_images);
  theia::SiftDescriptorExtractor sift_extractor;

  double time_to_read_images = 0;
  for (int i = 0; i < num_images; i++) {
    auto start = std::chrono::system_clock::now();
    images[i].Read(img_filepaths[i]);
    sift_extractor.DetectAndExtractDescriptors(images[i],
                                               &keypoints[i],
                                               &descriptors[i]);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now() - start);
    time_to_read_images += duration.count();
    LOG(INFO) << "Extracted " << descriptors[i].size()
              << " features for image: " << img_filepaths[i];
  }

  // Match all image pairs.
  theia::FeatureMatcherOptions options;
  theia::CascadeHashingFeatureMatcher image_matcher;
  std::vector<theia::ImagePairMatch> image_pair_matches;
  auto start_matching = std::chrono::system_clock::now();
  image_matcher.MatchAllPairs(options,
                              FLAGS_num_threads,
                              descriptors,
                              &image_pair_matches);
  auto duration_matching =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now() - start_matching);
  const double time_for_matching = duration_matching.count();

  LOG(INFO) << "It took " << (time_to_read_images / 1000.0)
            << " seconds to extract descriptors from " << num_images
            << " images and " << (time_for_matching / 1000.0)
            << " seconds to match all image pairs ("
            << image_pair_matches.size() << " pairs).";

  for (int i = 0; i < image_pair_matches.size(); i++) {
    theia::ImageCanvas image_canvas;
    const int img1_index = image_pair_matches[i].image1_ind;
    const int img2_index = image_pair_matches[i].image2_ind;
    image_canvas.AddImage(images[img1_index]);
    image_canvas.AddImage(images[img2_index]);
    const std::string match_output = theia::StringPrintf(
        "%s/matches_%i_%i.png",
        FLAGS_img_output_dir.c_str(),
        img1_index,
        img2_index);
    image_canvas.DrawMatchedFeatures(0, keypoints[img1_index],
                                     1, keypoints[img2_index],
                                     image_pair_matches[i].matches,
                                     0.1);
    image_canvas.Write(match_output);
  }
}
