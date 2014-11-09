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

#ifndef THEIA_IMAGE_IMAGE_H_
#define THEIA_IMAGE_IMAGE_H_

#include <cimg/CImg.h>
#include <easyexif/exif.h>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <fstream>  // NOLINT
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "theia/util/util.h"

namespace theia {
template <typename T> class Image;
typedef Image<float> FloatImage;
typedef Image<uchar> UcharImage;
typedef Image<int> IntImage;

// Templated on the number of channels.
template <typename T> class Image {
 public:
  Image() {}

  // Read from file.
  explicit Image(const std::string& filename);

  // Copy function. This is a deep copy of the image.
  Image(const Image<T>& image_to_copy);

  // Copy function. This is a deep copy of the image.
  template <typename D>
  Image(const Image<D>& image_to_copy);

  ~Image() {}
  // Image information
  int Rows() const;
  int Cols() const;
  int Width() const;
  int Height() const;
  int Channels() const;

  // The focal length is set from the EXIF data. We estimate the focal length
  // using the FocalplaneXResolution and FocalplaneYResolution values that
  // describe the number of pixels per FocalPlaneResolutionUnits (e.g. pixels
  // per mm) on the focal plane. This resolution can then be used to obtain the
  // focal length value in pixels.
  bool FocalLengthPixels(double* focal_length) const;

  // Returns the pixel value at (x, y). An optional third parameter can specify
  // the color channel.
  T& operator()(const int x, const int y, const int c = 0);
  const T& operator()(const int x, const int y, const int c = 0) const;

  // Convert to other image types.
  Image<T> AsGrayscaleImage() const;
  Image<T> AsRGBImage() const;
  void ConvertToGrayscaleImage();
  void ConvertToRGBImage();

  // Write image to file.
  void Read(const std::string& filename);
  void Write(const std::string& filename) const;

  // Get a pointer to the data.
  T* Data() { return image_.data(); }
  const T* Data() const { return image_.data(); }

  // Sampling techniques.
  void HalfSample(Image<T>* dest) const;
  void TwoThirdsSample(Image<T>* dest) const;

  // Compute the integral image where pixel (x, y) is equal to the sum of all
  // values in the rectangle from (0, 0) to (x, y) non-inclusive. This means
  // that the first row and column are all zeros.
  template <typename D = T>
  Image<D> Integrate() const;

  // Computes a fast approximate gaussian blur of te image.
  void ApproximateGaussianBlur(const double sigma);

  // Resize using a Lanczos 3 filter.
  void Resize(int new_rows, int new_cols);
  void Resize(double scale);

 protected:
  template<class AnyType> friend class Image;
  friend class ImageCanvas;

  explicit Image(const cimg_library::CImg<T>& image);

  cimg_library::CImg<T> image_;
  EXIFInfo exif_parser_;
};

// ----------------- Implementation ----------------- //

// Read from file.
template <typename T> Image<T>::Image(const std::string& filename) {
  // TODO(cmsweeney): Get EXIF data here as well.
  image_.load(filename.c_str());

  // Read in the EXIF information. This forces us to re-open the file but that
  // should not be a huge performance cost.
  std::streampos file_size;
  std::ifstream file(filename, std::ios::binary);

  // get its size:
  file.seekg(0, std::ios::end);
  file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  // read the data:
  std::vector<unsigned char> jpeg_data(file_size);
  file.read(reinterpret_cast<char*>(&jpeg_data[0]), file_size);

  // TODO(cmsweeney): The exif parser returns a success code. We may want to log
  // the success code.
  exif_parser_.parseFrom(jpeg_data.data(), file_size);
}

template <typename T> Image<T>::Image(const Image<T>& image_to_copy) {
  image_ = image_to_copy.image_;
  exif_parser_ = image_to_copy.exif_parser_;
}

// Copy function. This is a deep copy of the image.
template <typename T> template <typename D>
Image<T>::Image(const Image<D>& image_to_copy) {
  image_ = image_to_copy.image_;
  exif_parser_ = image_to_copy.exif_parser_;
}

template <typename T> Image<T>::Image(const cimg_library::CImg<T>& image) {
  image_ = image;
}

template <typename T> int Image<T>::Rows() const {
  return image_.height();
}

template <typename T> int Image<T>::Cols() const {
  return image_.width();
}

template <typename T> int Image<T>::Width() const {
  return image_.width();
}

template <typename T> int Image<T>::Height() const {
  return image_.height();
}

template <typename T> int Image<T>::Channels() const {
  return image_.spectrum();
}

template <typename T>
bool Image<T>::FocalLengthPixels(double* focal_length) const {
  // All of these fields must be set.
  if (exif_parser_.FocalLength > 0 &&
      exif_parser_.FocalPlaneXResolution > 0 &&
      exif_parser_.FocalPlaneYResolution > 0 &&
      exif_parser_.FocalPlaneResolutionUnit > 1 &&
      exif_parser_.FocalPlaneResolutionUnit <= 5) {
    // Convert to mm.
    double focal_length_x = exif_parser_.FocalPlaneXResolution;
    double focal_length_y = exif_parser_.FocalPlaneYResolution;
    if (focal_length_x != focal_length_y) {
      VLOG(2) << "Can only extract EXIF focal lengths from images with square "
                 "pixels.";
      return false;
    }

    switch (exif_parser_.FocalPlaneResolutionUnit) {
      case 2:
        // Convert inches to mm.
        focal_length_x /= 25.4;
        focal_length_y /= 25.4;
        break;
      case 3:
        // Convert centimeters to mm.
        focal_length_x /= 10.0;
        focal_length_y /= 10.0;
        break;
      case 4:
        // Already in mm.
        break;
      case 5:
        // Convert micrometers to mm.
        focal_length_x *= 10.0;
        focal_length_y *= 10.0;
        break;
      default:
        break;
    }

    // Normalize for the image size in case the original size is different than
    // the current size.
    *focal_length =
        (exif_parser_.FocalLength * focal_length_x) *
        (this->Width() / static_cast<double>(exif_parser_.ImageWidth));
    ;

    return true;
  }

  // TODO(cmsweeney): We could try a CCD lookup here?
  return false;
}

template <typename T>
T& Image<T>::operator()(const int x, const int y, const int c) {
  return image_(x, y, 0, c);
}

template <typename T>
const T& Image<T>::operator()(const int x, const int y, const int c) const {
  return image_(x, y, 0, c);
}

template <typename T> void Image<T>::ConvertToGrayscaleImage() {
  if (Channels() == 1) {
    VLOG(2) << "Image is already a grayscale image. No conversion necessary.";
    return;
  }
  image_ = image_.get_RGBtoYCbCr().channel(0);
}

template <typename T> void Image<T>::ConvertToRGBImage() {
  if (Channels() == 3) {
    VLOG(2) << "Image is already an RGB image. No conversion necessary.";
    return;
  }

  // Resizing with NN interpolation by default will copy the pixel value of the
  // grayscale pixel to all RGB channels.
  image_.resize(image_.width(),
                image_.height(),
                image_.depth(),
                3);
}

template <typename T> Image<T> Image<T>::AsGrayscaleImage() const {
  if (Channels() == 1) {
    VLOG(2) << "Image is already a grayscale image. No conversion necessary.";
    return *this;
  }
  Image<T> gray_image(*this);
  gray_image.ConvertToGrayscaleImage();
  return gray_image;
}

template <typename T> Image<T> Image<T>::AsRGBImage() const {
  if (Channels() == 3) {
    VLOG(2) << "Image is already an RGB image. No conversion necessary.";
    return *this;
  }

  Image<T> rgb_image(*this);
  rgb_image.ConvertToRGBImage();
  return rgb_image;
}

template <typename T> void Image<T>::Read(const std::string& filename) {
  image_.load(filename.c_str());
}

template <typename T> void Image<T>::Write(const std::string& filename) const {
  image_.save(filename.c_str());
}

template <typename T> void Image<T>::HalfSample(Image<T>* dest) const {
  dest->image_ = image_.get_resize_halfXY();
}

template <typename T> void Image<T>::TwoThirdsSample(Image<T>* dest) const {
  const int new_width = static_cast<int>(image_.width() * 2.0 / 3.0);
  const int new_height = static_cast<int>(image_.height() * 2.0 / 3.0);
  dest->image_ = image_.get_resize(new_width, new_height, -100, -100, 6);
}

template <typename T>
template <typename D>
Image<D> Image<T>::Integrate() const {
  Image<D> integral(*this);
  cimg_library::CImg<D>& integral_image = integral.image_;

  for (int i = 0; i < image_.spectrum(); i++) {
    // Do the first row.
    for (int c = 1; c < image_.width(); c++) {
      integral_image(c, 0, 0, i) += integral_image(c - 1, 0, 0, i);
    }

    // Do the first column.
    for (int r = 1; r < image_.height(); r++) {
      integral_image(0, r, 0, i) += integral_image(0, r - 1, 0, i);
    }

    for (int r = 1; r < image_.height(); r++) {
      for (int c = 1; c < image_.width(); c++) {
        integral_image(c, r, 0, i) += integral_image(c - 1, r, 0, i);
      }
    }
  }

  return integral;
}

template <typename T>
void Image<T>::ApproximateGaussianBlur(const double sigma) {
  image_.blur(sigma);
}

template <typename T> void Image<T>::Resize(int new_rows, int new_cols) {
  image_.resize(new_cols, new_rows);
}

}  // namespace theia

#endif  // THEIA_IMAGE_IMAGE_H_
