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

#ifndef THEIA_VISION_SFM_VIEW_GRAPH_VIEW_GRAPH_H_
#define THEIA_VISION_SFM_VIEW_GRAPH_VIEW_GRAPH_H_

#include <unordered_map>
#include <unordered_set>

#include "theia/util/hash.h"
#include "theia/vision/sfm/twoview_info.h"
#include "theia/vision/sfm/types.h"

namespace theia {

// An undirected graph containing views in an SfM reconstruction. The graph is
// efficienctly created by only holding view ids at the vertices and
// TwoViewInfos for edge values.
class ViewGraph {
 public:
  ViewGraph() {}

  // Number of views in the graph.
  int NumViews() const;

  // Number of undirected edges in the graph.
  int NumEdges() const;

  // Returns true if the view is contained in the graph, false otherwise.
  bool HasView(const ViewId view_id) const;

  // Returns a set of the ViewIds contained in the view graph.
  std::unordered_set<ViewId> ViewIds() const;

  // Adds the view to the view graph if it is not already present.
  void AddView(const ViewId view_id);

  // Removes the view from the view graph and removes all edges connected to the
  // view. Returns true on success and false if the view did not exist in the
  // view graph.
  bool RemoveView(const ViewId view_id);

  // Adds an edge between the two views with the edge value of
  // two_view_info. New vertices are added to the graph if they did not already
  // exist. If an edge already existed between the two views then the edge value
  // is updated.
  void AddEdge(const ViewId view_id_1, const ViewId view_id_2,
               const TwoViewInfo& two_view_info);

  // Removes the edge from the view graph. Returns true if the edge is removed
  // and false if the edge did not exist.
  bool RemoveEdge(const ViewId view_id_1, const ViewId view_id_2);

  // Returns all the edges for a given
  const std::unordered_map<ViewId, TwoViewInfo>* GetEdges(
      const ViewId view_id) const;

  // Returns the edge value or NULL if it does not exist.
  const TwoViewInfo* GetEdgeValue(const ViewId view_id_1,
                                  const ViewId view_id_2) const;

  // Returns a map of all edges. Each edge is found exactly once in the map and
  // is indexed by the ViewIdPair (view id 1, view id 2) such that view id 1 <
  // view id 2.
  std::unordered_map<ViewIdPair, TwoViewInfo> GetAllEdges() const;

 private:
  // The underlying adjacency map. ViewIds are the vertices which are mapped to
  // a set of edges. The edges are stored as maps of the ViewId of the adjacent
  // node as well as the TwoViewInfo connecting the two views.
  std::unordered_map<ViewId, std::unordered_map<ViewId, TwoViewInfo> >
      vertices_;
};

}  // namespace theia

#endif  // THEIA_VISION_SFM_VIEW_GRAPH_VIEW_GRAPH_H_
