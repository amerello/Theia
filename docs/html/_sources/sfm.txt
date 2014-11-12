.. highlight:: c++

.. default-domain:: cpp

.. _documentation-sfm:

===========================
Structure from Motion (SfM)
===========================

Theia has a full Structure-from-Motion pipeline that is extremely efficient. Our
overall pipeline consists of several steps. First, we extract features (SIFT is
the default). Then, we perform two-view matching and geometric verification to
obtain relative poses between image pairs and create a :class:`ViewGraph`. Next,
we perform global pose estimation with "one-shot" SfM. One-shot SfM is different
from incremental SfM in that it considers the entire view graph at the same time
instead of successfully adding more and more images to the
:class:`Model`. One-shot SfM methods have been proven to be very fast with
comparble or better accuracy to incremental SfM approaches, and they are much
more easily parallelized. After we have obtained camera poses, we perform
triangulation and :class:`BundleAdjustment` to obtain a valid 3D reconstruction
consisting of cameras and 3D points.

Extracting and matching :ref:`documentation-features` has been covered already, so we
will now discuss how to go from feature matches to a :class:`ViewGraph`. First,
however, we must present the basic building blocks for our SfM pipeline.

Views
=====

.. class:: View

At the heart of our SfM framework is the :class:`View` class which represents
everything about an image that we want to reconstruct. It contains information
about features from the image, camera pose information, and metadata information
(including the image name and EXIF data). Views make up our basic visiblity
constraints and are a fundamental part of the SfM pipeline.

Tracks
======

.. class:: Track

A :class:`Track` represents a feature that has been matached over potentially
many images. When a feature appears in multiple images it typically means that
the features correspond to the same 3D point. These 3D points are useful
constraints in SfM model, as they represent the "structure" in
"Structure-from-Motion" and help to build a point cloud for our model.

ViewGraph
=========

.. class:: ViewGraph

A :class:`ViewGraph` is a basic SfM construct that is created from two-view
matching information. Any pair of views that have a view correlation form an
edge in the :class:`ViewGraph` such that the nodes in the graph are
:class:`View` that are connected by :class:`TwoViewInfo` objects that contain
information about the relative pose between the Views as well as matching
information.

Once you have a set of views and match information, you can add them to the view graph:

.. code:: c++

  std::vector<View> views;
  // Match all views in the set.
  std::vector<ViewIdPair, TwoViewInfo> view_pair_matches;

  ViewGraph view_graph;
  for (const auto& view_pair : view_pair_matches) {
    const ViewIdPair& view_id_pair = view_pair.first;
    const TwoViewInfo& two_view_info = view_pair.second;
    // Only add view pairs to the view graph if they have strong visual coherence.
    if (two_view_info.num_matched_features > min_num_matched_features) {
      view_graph.AddEdge(views[view_id_pair.first],
                         views[view_id_pair.second],
                         two_view_info);
    }
  }

  // Process and/or manipulate the view graph.

The edge values are especially useful for one-shot SfM where the relative poses
are heavily exploited for computing the final poses.

Camera
======

.. class:: Camera

Each :class:`View` contains a :class:`Camera` object that contains intrinsic and
extrinsic information about the camera that observed the scene. Theia has an
efficient, compact :class:`Camera` class that abstracts away common image
operations. One common example is 3D point reprojection.

.. code:: c++

   FloatImage image("my_image.jpg");
   double focal_length;
   CHECK(image.FocalLengthPixels(&focal_length));

   const double radial_distortion1 = value obtained elsewhere...
   const double radial_distortion2 = value obtained elsewhere...

   Camera camera;
   camera.SetFocalLength(focal_length);
   camera.SetPrincipalPoint(image.Width() / 2.0, image.Height() / 2.0);
   camera.SetRadialDistortion(radial_distortion1, radial_distortion2);

   const Eigen::Vector4d point = value obtained elsewhere...

   Eigen::Vector2d reprojection_pixel;
   const double depth = camera.ProjectPoint(point, &pixel);
   if (depth < 0) {
     LOG(INFO) << "Point was behind the camera!";
   }

Point projection can be a tricky function when considering the camera intrinsics
and it only becomes more complicated once nontrivial skew and aspect ratios
(which Theia also uses as camera parameters) are considered.

In addition to typical getter/setter methods for the camera parameters, the
:class:`Camera` class also defines several helper functions:.

.. function:: bool InitializeFromProjectionMatrix(const int image_width, const int image_height, const Matrix3x4d projection_matrix)

    Initializes the camera intrinsic and extrinsic parameters from the
    projection matrix by decomposing the matrix with a RQ decomposition.

    .. NOTE:: The projection matrix does not contain information about radial
        distortion, so those parameters will need to be set separately.

.. function:: void GetProjectionMatrix(Matrix3x4d* pmatrix) const

    Returns the projection matrix. Does not include radial distortion.

.. function:: void GetCalibrationMatrix(Eigen::Matrix3d* kmatrix) const

    Returns the calibration matrix in the form specified above.

.. function:: Eigen::Vector3d PixelToUnitDepthRay(const Eigen::Vector2d& pixel) const

    Converts the pixel point to a ray in 3D space such that the origin of the
    ray is at the camera center and the direction is the pixel direction rotated
    according to the camera orientation in 3D space. The returned vector is not
    unit length.


Model
=====

At the core of our SfM pipeline is an SfM :class:`Model`. A :class:`Model` is the representation of a 3D reconstuction consisting of Views and Tracks.  A :class:`View` represents an image, containing :class:`Camera` pose information, metadata (usually from EXIF) and visibility information. A :class:`Track` is feature that has been matched across multiple views which may or may not have a valid 3D point. A :class:`View` in a :class:`Model` will observe potentially many 3D points.

.. class:: Model

.. NOTE:: Docmentation coming soon...


Estimating Global Poses
=======================

.. NOTE:: Documentation coming soon..

Triangulation
=============

  Triangulation in structure from motion calculates the 3D position of an image
  coordinate that has been tracked through several, if not many, images.

  .. cpp:function:: bool Triangulate(const ProjectionMatrix& pose_left, const ProjectionMatrix& pose_right, const Eigen::Vector2d& point_left, const Eigen::Vector2d& point_right, Eigen::Vector3d* triangulated_point)

    2-view triangulation using the DLT method described in
    [HartleyZisserman]_. The poses are the (potentially calibrated) poses of the
    two cameras, and the points are the 2D image points of the matched features
    that will be used to triangulate the 3D point. If there was an error computing
    the triangulation (e.g., the point is found to be at infinity) then ``false``
    is returned. On successful triangulation, ``true`` is returned.

  .. cpp:function:: bool TriangulateNViewSVD(const std::vector<ProjectionMatrix>& poses, const std::vector<Eigen::Vector2d>& points, Eigen::Vector3d* triangulated_point)
  .. cpp:function:: bool TriangulateNView(const std::vector<ProjectionMatrix>& poses, const std::vector<Eigen::Vector2d>& points, Eigen::Vector3d* triangulated_point)

    We provide two N-view triangluation methods that minimizes an algebraic
    approximation of the geometric error. The first is the classic SVD method
    presented in [HartleyZisserman]_. The second is a custom algebraic
    minimization. Note that we can derive an algebraic constraint where we note
    that the unit ray of an image observation can be stretched by depth
    :math:`\alpha` to meet the world point :math:`X` for each of the :math:`n`
    observations:

    .. math:: \alpha_i \bar{x_i} = P_i X,

    for images :math:`i=1,\ldots,n`. This equation can be effectively rewritten as:

    .. math:: \alpha_i = \bar{x_i}^\top P_i X,

    which can be substituted into our original constraint such that:

    .. math:: \bar{x_i} \bar{x_i}^\top P_i X = P_i X
    .. math:: 0 = (P_i - \bar{x_i} \bar{x_i}^\top P_i) X

    We can then stack this constraint for each observation, leading to the linear
    least squares problem:

    .. math:: \begin{bmatrix} (P_1 - \bar{x_1} \bar{x_1}^\top P_1) \\ \vdots \\ (P_n - \bar{x_n} \bar{x_n}^\top P_n) \end{bmatrix} X = \textbf{0}

    This system of equations is of the form :math:`AX=0` which can be solved by
    extracting the right nullspace of :math:`A`. The right nullspace of :math:`A`
    can be extracted efficiently by noting that it is equivalent to the nullspace
    of :math:`A^\top A`, which is a 4x4 matrix.

Bundle Adjustment
=================

.. NOTE:: Docmentation coming soon...

Similarity Transformation
=========================

  .. cpp:function:: void AlignPointCloudsICP(const int num_points, const double left[], const double right[], double rotation[3 * 3], double translation[3])

    We implement ICP for point clouds. We use Besl-McKay registration to align
    point clouds. We use SVD decomposition to find the rotation, as this is much
    more likely to find the global minimum as compared to traditional ICP, which
    is only guaranteed to find a local minimum. Our goal is to find the
    transformation from the left to the right coordinate system. We assume that
    the left and right models have the same number of points, and that the
    points are aligned by correspondence (i.e. left[i] corresponds to right[i]).

  .. cpp:function:: void AlignPointCloudsUmeyama(const int num_points, const double left[], const double right[], double rotation[3 * 3], double translation[3], double* scale)

    This function estimates the 3D similiarty transformation using the least
    squares method of [Umeyama]_. The returned rotation, translation, and scale
    align the left points to the right such that :math:`Right = s * R * Left +
    t`.

  .. cpp:function:: void GdlsSimilarityTransform(const std::vector<Eigen::Vector3d>& ray_origin, const std::vector<Eigen::Vector3d>& ray_direction, const std::vector<Eigen::Vector3d>& world_point, std::vector<Eigen::Quaterniond>* solution_rotation, std::vector<Eigen::Vector3d>* solution_translation, std::vector<double>* solution_scale)

    Computes the solution to the generalized pose and scale problem based on the
    paper "gDLS: A Scalable Solution to the Generalized Pose and Scale Problem"
    by Sweeney et. al. [SweeneyGDLS]_. Given image rays from one coordinate
    system that correspond to 3D points in another coordinate system, this
    function computes the rotation, translation, and scale that will align the
    rays with the 3D points. This is used for applications such as loop closure
    in SLAM and SfM. This method is extremely scalable and highly accurate
    because the cost function that is minimized is independent of the number of
    points. Theoretically, up to 27 solutions may be returned, but in practice
    only 4 real solutions arise and in almost all cases where n >= 6 there is
    only one solution which places the observed points in front of the
    camera. The rotation, translation, and scale are defined such that:
    :math:`sp_i + \alpha_i d_i = RX_i + t` where the observed image ray has an
    origin at :math:`p_i` in the unit direction :math:`d_i` corresponding to 3D
    point :math:`X_i`.

    ``ray_origin``: the origin (i.e., camera center) of the image ray used in
    the 2D-3D correspondence.

    ``ray_direction``: Normalized image rays corresponding to model points. Must
    contain at least 4 points.

    ``world_point``: 3D location of features. Must correspond to the image_ray
    of the same index. Must contain the same number of points as image_ray, and
    at least 4.

    ``solution_rotation``: the rotation quaternion of the candidate solutions

    ``solution_translation``: the translation of the candidate solutions

    ``solution_scale``: the scale of the candidate solutions
