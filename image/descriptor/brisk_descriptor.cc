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

#include "image/descriptor/brisk_descriptor.h"

#include <glog/logging.h>

#ifndef THEIA_NO_PROTOCOL_BUFFERS
#include "image/descriptor/descriptor.pb.h"
#endif
#include "image/image.h"

namespace theia {
namespace {
typedef unsigned char uchar;
// this is needed to avoid aliasing issues with the __m128i data type:
#ifdef __GNUC__
typedef unsigned char __attribute__ ((__may_alias__)) UCHAR_ALIAS;
typedef unsigned short __attribute__ ((__may_alias__)) UINT16_ALIAS;
typedef unsigned int __attribute__ ((__may_alias__)) UINT32_ALIAS;
typedef unsigned long int __attribute__ ((__may_alias__)) UINT64_ALIAS;
typedef int __attribute__ ((__may_alias__)) INT32_ALIAS;
typedef uint8_t __attribute__ ((__may_alias__)) U_INT8T_ALIAS;
#endif
#ifdef _MSC_VER
// Todo: find the equivalent to may_alias
#define UCHAR_ALIAS unsigned char //__declspec(noalias)
#define UINT32_ALIAS unsigned int //__declspec(noalias)
#define __inline__ __forceinline
#endif
}  // namespace

const float BriskDescriptorExtractor::basic_size_ = 12.0;
const unsigned int BriskDescriptorExtractor::scales_ = 64;
// 40->4 Octaves - else, this needs to be adjusted...
const float BriskDescriptorExtractor::scale_range_ = 30;
// discretization of the rotation look-up
const unsigned int BriskDescriptorExtractor::n_rot_ = 1024;

// constructors
BriskDescriptorExtractor::BriskDescriptorExtractor(bool rotation_invariant,
                                                   bool scale_invariant,
                                                   float pattern_scale) {
  std::vector<float> rList;
  std::vector<int> nList;

  // this is the standard pattern found to be suitable also
  rList.resize(5);
  nList.resize(5);
  const double f = 0.85*pattern_scale;

  rList[0] = f*0;
  rList[1] = f*2.9;
  rList[2] = f*4.9;
  rList[3] = f*7.4;
  rList[4] = f*10.8;

  nList[0] = 1;
  nList[1] = 10;
  nList[2] = 14;
  nList[3] = 15;
  nList[4] = 20;

  rotation_invariance_ = rotation_invariant;
  scale_invariance_ = scale_invariant;
  generateKernel(rList, nList, 5.85*pattern_scale, 8.2*pattern_scale);
}

void BriskDescriptorExtractor::generateKernel(std::vector<float> &radiusList,
                                              std::vector<int> &numberList,
                                              float dMax, float dMin,
                                              std::vector<int> indexChange) {
  dMax_ = dMax;
  dMin_ = dMin;

  // get the total number of points
  const int rings = radiusList.size();
  CHECK_NE(radiusList.size(), 0);
  CHECK_EQ(radiusList.size(), numberList.size());
  points_ = 0; // remember the total number of points
  for (int ring = 0; ring < rings; ring++) {
    points_+=numberList[ring];
  }
  // set up the patterns
  pattern_points_ = new BriskPatternPoint[points_*scales_*n_rot_];
  BriskPatternPoint* patternIterator = pattern_points_;

  // define the scale discretization:
  static const float lb_scale = log(scale_range_)/log(2.0);
  static const float lb_scale_step = lb_scale/(scales_);

  scale_list_ = new float[scales_];
  size_list_ = new unsigned int[scales_];

  const float sigma_scale = 1.3;

  for (unsigned int scale = 0; scale < scales_; ++scale) {
    scale_list_[scale] = pow((double)2.0, (double)(scale*lb_scale_step));
    size_list_[scale] = 0;

    // generate the pattern points look-up
    double alpha, theta;
    for (size_t rot = 0; rot < n_rot_; ++rot) {
      // this is the rotation of the feature
      theta = double(rot)*2*M_PI/double(n_rot_);
      for (int ring = 0; ring<rings; ++ring) {
        for (int num = 0; num<numberList[ring]; ++num) {
          // the actual coordinates on the circle
          alpha = (double(num))*2*M_PI/double(numberList[ring]);
          // feature rotation plus angle of the point
          patternIterator->x = scale_list_[scale]*radiusList[ring]*cos(alpha+theta);
          patternIterator->y = scale_list_[scale]*radiusList[ring]*sin(alpha+theta);
          // and the gaussian kernel sigma
          if (ring == 0) {
            patternIterator->sigma = sigma_scale*scale_list_[scale]*0.5;
          } else {
            patternIterator->sigma =
                sigma_scale*scale_list_[scale]*
                (double(radiusList[ring]))*sin(M_PI/numberList[ring]);
          }
          // adapt the sizeList if necessary
          const unsigned int size = ceil(scale_list_[scale]*radiusList[ring] +
                                         patternIterator->sigma) + 1;
          if (size_list_[scale] < size) {
            size_list_[scale] = size;
          }

          // increment the iterator
          ++patternIterator;
        }
      }
    }
  }

  // now also generate pairings
  short_pairs_ = new BriskShortPair[points_*(points_ - 1)/2];
  long_pairs_ = new BriskLongPair[points_*(points_ - 1)/2];
  no_short_pairs_ = 0;
  no_long_pairs_ = 0;

  // fill indexChange with 0..n if empty
  unsigned int indSize = indexChange.size();
  if (indSize == 0) {
    indexChange.resize(points_*(points_ - 1)/2);
    indSize = indexChange.size();
  }
  for (unsigned int i = 0; i<indSize; i++) {
    indexChange[i] = i;
  }
  const float dMin_sq = dMin_*dMin_;
  const float dMax_sq = dMax_*dMax_;
  for (unsigned int i= 1; i < points_; i++) {
    for (unsigned int j= 0; j < i; j++) { //(find all the pairs)
      // point pair distance:
      const float dx = pattern_points_[j].x - pattern_points_[i].x;
      const float dy = pattern_points_[j].y - pattern_points_[i].y;
      const float norm_sq = (dx*dx + dy*dy);
      if (norm_sq > dMin_sq) {
        // save to long pairs
        BriskLongPair& longPair = long_pairs_[no_long_pairs_];
        longPair.weighted_dx = int((dx/(norm_sq))*2048.0 + 0.5);
        longPair.weighted_dy = int((dy/(norm_sq))*2048.0 + 0.5);
        longPair.i = i;
        longPair.j = j;
        ++no_long_pairs_;
      } else if (norm_sq < dMax_sq) {
        // save to short pairs
        // make sure the user passes something sensible
        CHECK_LT(no_short_pairs_, indSize);
        BriskShortPair& shortPair = short_pairs_[indexChange[no_short_pairs_]];
        shortPair.j = j;
        shortPair.i = i;
        ++no_short_pairs_;
      }
    }
  }

  // number of bits:
  strings_ = (int)ceil((float(no_short_pairs_))/128.0)*4*4;
}

// simple alternative:
inline int BriskDescriptorExtractor::smoothedIntensity(
    const Image<uchar>& image, const Image<int>& integral,
    const float key_x, const float key_y,
    const unsigned int scale, const unsigned int rot,
    const unsigned int point) const{
  // get the float position
  const BriskPatternPoint& briskPoint = pattern_points_[scale*n_rot_*points_ +
                                                        rot*points_ + point];
  const float xf = briskPoint.x + key_x;
  const float yf = briskPoint.y + key_y;
  const int x = int(xf);
  const int y = int(yf);
  const int& imagecols = image.Cols();

  // get the sigma:
  const float sigma_half = briskPoint.sigma;
  const float area = 4.0*sigma_half*sigma_half;

  // calculate output:
  int ret_val;
  if (sigma_half<0.5) {
    //interpolation multipliers:
    const int r_x = (xf-x)*1024;
    const int r_y = (yf-y)*1024;
    const int r_x_1 = (1024-r_x);
    const int r_y_1 = (1024-r_y);
    const uchar* ptr = image.Data() + x + y*imagecols;
    // just interpolate:
    ret_val = (r_x_1*r_y_1*int(*ptr));
    ptr++;
    ret_val+=(r_x*r_y_1*int(*ptr));
    ptr+=imagecols;
    ret_val+=(r_x*r_y*int(*ptr));
    ptr--;
    ret_val+=(r_x_1*r_y*int(*ptr));
    return (ret_val+512)/1024;
  }

  // this is the standard case (simple, not speed optimized yet):
  // scaling:
  const int scaling = 4194304.0/area;
  const int scaling2 = float(scaling)*area/1024.0;

  // the integral image is larger:
  const int integralcols = imagecols+1;

  // calculate borders
  const float x_1 = xf-sigma_half;
  const float x1 = xf+sigma_half;
  const float y_1 = yf-sigma_half;
  const float y1 = yf+sigma_half;

  const int x_left = int(x_1+0.5);
  const int y_top = int(y_1+0.5);
  const int x_right = int(x1+0.5);
  const int y_bottom = int(y1+0.5);

  // overlap area - multiplication factors:
  const float r_x_1 = float(x_left)-x_1+0.5;
  const float r_y_1 = float(y_top)-y_1+0.5;
  const float r_x1 = x1-float(x_right)+0.5;
  const float r_y1 = y1-float(y_bottom)+0.5;
  const int dx = x_right-x_left-1;
  const int dy = y_bottom-y_top-1;
  const int A = (r_x_1*r_y_1)*scaling;
  const int B = (r_x1*r_y_1)*scaling;
  const int C = (r_x1*r_y1)*scaling;
  const int D = (r_x_1*r_y1)*scaling;
  const int r_x_1_i = r_x_1*scaling;
  const int r_y_1_i = r_y_1*scaling;
  const int r_x1_i = r_x1*scaling;
  const int r_y1_i = r_y1*scaling;
  if (dx+dy>2) {
    // now the calculation:
    const uchar* ptr = image.Data() + x_left + imagecols*y_top;
    // first the corners:
    ret_val = A*int(*ptr);
    ptr+=dx+1;
    ret_val+=B*int(*ptr);
    ptr+=dy*imagecols+1;
    ret_val+=C*int(*ptr);
    ptr-=dx+1;
    ret_val+=D*int(*ptr);

    // next the edges:
    const int* ptr_integral = integral.Data() + x_left +
                              integralcols*y_top+1;
    // find a simple path through the different surface corners
    const int tmp1 = (*ptr_integral);
    ptr_integral+=dx;
    const int tmp2 = (*ptr_integral);
    ptr_integral+=integralcols;
    const int tmp3 = (*ptr_integral);
    ptr_integral++;
    const int tmp4 = (*ptr_integral);
    ptr_integral += dy*integralcols;
    const int tmp5 = (*ptr_integral);
    ptr_integral--;
    const int tmp6 = (*ptr_integral);
    ptr_integral+=integralcols;
    const int tmp7 = (*ptr_integral);
    ptr_integral-=dx;
    const int tmp8 = (*ptr_integral);
    ptr_integral-=integralcols;
    const int tmp9 = (*ptr_integral);
    ptr_integral--;
    const int tmp10 = (*ptr_integral);
    ptr_integral-=dy*integralcols;
    const int tmp11 = (*ptr_integral);
    ptr_integral++;
    const int tmp12 = (*ptr_integral);

    // assign the weighted surface integrals:
    const int upper = (tmp3-tmp2+tmp1-tmp12)*r_y_1_i;
    const int middle = (tmp6-tmp3+tmp12-tmp9)*scaling;
    const int left = (tmp9-tmp12+tmp11-tmp10)*r_x_1_i;
    const int right = (tmp5-tmp4+tmp3-tmp6)*r_x1_i;
    const int bottom = (tmp7-tmp6+tmp9-tmp8)*r_y1_i;

    return (ret_val + upper + middle + left + right + bottom +
            scaling2/2)/scaling2;
  }

  // now the calculation:
  const uchar* ptr = image.Data() + x_left + imagecols*y_top;
  // first row:
  ret_val = A*int(*ptr);
  ptr++;
  const uchar* end1 = ptr+dx;
  for (; ptr<end1; ptr++) {
    ret_val+=r_y_1_i*int(*ptr);
  }
  ret_val+=B*int(*ptr);
  // middle ones:
  ptr+=imagecols-dx-1;
  const uchar* end_j = ptr+dy*imagecols;
  for (; ptr<end_j; ptr+=imagecols-dx-1) {
    ret_val+=r_x_1_i*int(*ptr);
    ptr++;
    const uchar* end2 = ptr+dx;
    for (; ptr<end2; ptr++) {
      ret_val+=int(*ptr)*scaling;
    }
    ret_val+=r_x1_i*int(*ptr);
  }
  // last row:
  ret_val+=D*int(*ptr);
  ptr++;
  const uchar* end3 = ptr+dx;
  for (; ptr<end3; ptr++) {
    ret_val+=r_y1_i*int(*ptr);
  }
  ret_val+=C*int(*ptr);

  return (ret_val+scaling2/2)/scaling2;
}

bool RoiPredicate(const float minX, const float minY,
                  const float maxX, const float maxY, const Keypoint& keyPt) {
  return (keyPt.x() < minX) || (keyPt.x() >= maxX) ||
      (keyPt.y() < minY) || (keyPt.y() >= maxY);
}


// Computes a descriptor at a single keypoint.
Descriptor* BriskDescriptorExtractor::ComputeDescriptor(
    const GrayImage& image,
    const Keypoint& keypoint) {
  Descriptor* descriptor = new BriskDescriptor;
  std::vector<Keypoint*> keypoints;
  // TODO(cmsweeney): Is there a better way to pass the address of the keypoint
  // passed in? This is sort of a lazy hack.
  Keypoint keypoint_copy = keypoint;
  keypoints.push_back(&keypoint_copy);
  std::vector<Descriptor*> descriptors;
  descriptors.push_back(descriptor);
  CHECK(ComputeDescriptors(image, keypoints, &descriptors));
  return descriptor;
}

// computes the descriptor
bool BriskDescriptorExtractor::ComputeDescriptors(
    const GrayImage& image,
    const std::vector<Keypoint*>& keypoints,
    std::vector<Descriptor*>* descriptors) {

  //Remove keypoints very close to the border
  size_t ksize = keypoints.size();
  std::vector<int> kscales; // remember the scale per keypoint
  std::vector<bool> valid_keypoint(ksize, true);
  kscales.resize(ksize);
  static const float log2 = 0.693147180559945;
  static const float lb_scalerange = log(scale_range_)/(log2);
  static const float basicSize06 = basic_size_*0.6;
  unsigned int basicscale = 0;
  if (!scale_invariance_)
    basicscale = std::max(
        (int)(scales_/lb_scalerange*(log(1.45*basic_size_/(basicSize06))/log2) +
              0.5), 0);
  for (size_t k = 0; k<ksize; k++) {
    unsigned int scale;
    if (scale_invariance_) {
      scale = std::max((int)(scales_/lb_scalerange*
                             (log(12.0*keypoints[k]->scale()/(basicSize06))/log2) +
                             0.5), 0);
      // saturate
      if (scale>=scales_)
        scale = scales_-1;
      kscales[k] = scale;
    } else {
      scale = basicscale;
      kscales[k] = scale;
    }
    const int border = size_list_[scale];
    const int border_x = image.Cols() - border - 1;
    const int border_y = image.Rows() - border - 1;
    if (RoiPredicate(border, border,border_x,border_y,*keypoints[k])) {
      valid_keypoint[k] = false;
    }
  }

  // first, calculate the integral image over the whole image:
  // current integral image
  Image<uchar> uchar_image = image.ConvertTo<uchar>();
  Image<int> _integral = uchar_image.Integrate<int>();
  int* _values = new int[points_]; // for temporary use

  // resize the descriptors:
  descriptors->resize(keypoints.size(), nullptr);

  // now do the extraction for all keypoints:

  // temporary variables containing gray values at sample points:
  int t1;
  int t2;

  // the feature orientation
  int direction0;
  int direction1;
  for (size_t k = 0; k<ksize; k++) {
    if (!valid_keypoint[k]) {
      continue;
    }
    int theta;
    Keypoint& kp = *keypoints[k];
    const int& scale = kscales[k];
    int shifter = 0;
    int* pvalues = _values;
    const float& x = kp.x();
    const float& y = kp.y();

    BriskDescriptor* brisk_descriptor = new BriskDescriptor;
    brisk_descriptor->SetKeypoint(*keypoints[k]);

    if (!rotation_invariance_) {
      // don't compute the gradient direction, just assign a rotation of 0°
      theta = 0;
    } else {
      // get the gray values in the unrotated pattern
      for (unsigned int i = 0; i < points_; i++) {
        *(pvalues++) = smoothedIntensity(uchar_image, _integral, x,
                                         y, scale, 0, i);
      }

      direction0 = 0;
      direction1 = 0;
      // now iterate through the long pairings
      const BriskLongPair* max = long_pairs_ + no_long_pairs_;
      for (BriskLongPair* iter = long_pairs_; iter<max; ++iter) {
        t1=*(_values+iter->i);
        t2=*(_values+iter->j);
        const int delta_t = (t1-t2);
        // update the direction:
        const int tmp0 = delta_t*(iter->weighted_dx)/1024;
        const int tmp1 = delta_t*(iter->weighted_dy)/1024;
        direction0+=tmp0;
        direction1+=tmp1;
      }
      brisk_descriptor->set_orientation(atan2((float)direction1,
                                              (float)direction0));
      double kp_angle = brisk_descriptor->orientation()/M_PI*180.0;
      theta = int((n_rot_*kp_angle)/(360.0)+0.5);
      if (theta<0)
        theta+=n_rot_;
      if (theta>=int(n_rot_))
        theta-=n_rot_;
    }


    // now also extract the stuff for the actual direction:
    // let us compute the smoothed values
    shifter = 0;

    //unsigned int mean=0;
    pvalues = _values;
    // get the gray values in the rotated pattern
    for (unsigned int i = 0; i<points_; i++) {
      /*
        VLOG(0) << "smoothed intensity settings: " << x << ", " << y
        << ", " << scale << ", " << theta << ", " << i;
        VLOG(0) << "border = " << size_list_[scale];
      */
      *(pvalues++) = smoothedIntensity(uchar_image, _integral, x,
                                       y, scale, theta, i);
    }

    // now iterate through all the pairings
    const BriskShortPair* max = short_pairs_ + no_short_pairs_;
    for (BriskShortPair* iter = short_pairs_; iter < max; ++iter) {
      t1 = *(_values + iter->i);
      t2 = *(_values + iter->j);
      (*brisk_descriptor)[shifter] = t1 > t2;
      ++shifter;
    }
    (*descriptors)[k] = brisk_descriptor;
  }

  // clean-up
  delete [] _values;
  return true;
}

BriskDescriptorExtractor::~BriskDescriptorExtractor() {
  delete [] pattern_points_;
  delete [] short_pairs_;
  delete [] long_pairs_;
  delete [] scale_list_;
  delete [] size_list_;
}

#ifndef THEIA_NO_PROTOCOL_BUFFERS
bool BriskDescriptorExtractor::ProtoToDescriptor(
    const DescriptorsProto& proto,
    std::vector<Descriptor*>* descriptors) const {
  descriptors->reserve(proto.feature_descriptor_size());
  for (const DescriptorProto& proto_descriptor: proto.feature_descriptor()) {
    BriskDescriptor* descriptor = new BriskDescriptor;
    CHECK_EQ(proto_descriptor.descriptor_type(), DescriptorProto::BRISK)
        << "Descriptor in proto is not a patch descriptor.";
    CHECK_EQ(proto_descriptor.binary_descriptor().size(),
             descriptor->Dimensions())
        << "Dimension mismatch in the proto and descriptors.";
    descriptor->set_x(proto_descriptor.x());
    descriptor->set_y(proto_descriptor.y());
    descriptor->set_orientation(proto_descriptor.orientation());
    descriptor->set_scale(proto_descriptor.scale());
    if (proto_descriptor.has_strength())
      descriptor->set_strength(proto_descriptor.strength());

    // Get binary array.
    *(descriptor->BinaryData())=
        std::bitset<512>(proto_descriptor.binary_descriptor());
    descriptors->push_back(descriptor);
  }
  return true;
}

bool BriskDescriptorExtractor::DescriptorToProto(
    const std::vector<Descriptor*>& descriptors,
    DescriptorsProto* proto) const {
  for (const Descriptor* generic_descriptor : descriptors) {
    const BriskDescriptor* descriptor =
        dynamic_cast<const BriskDescriptor*>(generic_descriptor);
    DescriptorProto* descriptor_proto = proto->add_feature_descriptor();
    // Add the binary array to the proto.
    descriptor_proto->set_binary_descriptor(
        descriptor->BinaryData()->to_string());

    // Set the proto type to patch.
    descriptor_proto->set_descriptor_type(DescriptorProto::BRISK);
    descriptor_proto->set_x(descriptor->x());
    descriptor_proto->set_y(descriptor->y());
    descriptor_proto->set_orientation(descriptor->orientation());
    descriptor_proto->set_scale(descriptor->scale());
    if (descriptor->has_strength())
      descriptor_proto->set_strength(descriptor->strength());
  }
  return true;
}
#endif  // THEIA_NO_PROTOCOL_BUFFERS
}  // namespace theia
