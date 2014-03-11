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

#ifndef THEIA_SOLVERS_MLESAC_H_
#define THEIA_SOLVERS_MLESAC_H_

#include <glog/logging.h>
#include <vector>

#include "theia/math/distribution.h"
#include "theia/solvers/mle_quality_measurement.h"
#include "theia/solvers/random_sampler.h"
#include "theia/solvers/sample_consensus_estimator.h"

namespace theia {

template <class Datum, class Model>
class Mlesac : public SampleConsensusEstimator<Datum, Model> {
 public:
  explicit Mlesac(const int min_sample_size)
      : SampleConsensusEstimator<Datum, Model>(min_sample_size) {}
  ~Mlesac() {}

  // Initializes the random sampler and mle support measurement.
  bool Initialize(const RansacParameters& ransac_params) {
    Sampler<Datum>* random_sampler =
        new RandomSampler<Datum>(this->min_sample_size_);
    QualityMeasurement* mle_support =
        new MLEQualityMeasurement(ransac_params.error_thresh);
    return SampleConsensusEstimator<Datum, Model>::Initialize(
        ransac_params, random_sampler, mle_support);
  }
};

}  // namespace theia

#endif  // THEIA_SOLVERS_MLESAC_H_
