.. highlight:: c++

.. default-domain:: cpp

.. _documentation-pose:

=====================
Pose and Resectioning
=====================

Theia contains efficient and robust implementations of the following pose and
resectioning algorithms. We attempted to make each method as general as possible so that users were not tied to Theia data structures to use the methods. The interface for all pose methods uses Eigen types for feature positions, 3D positions, and pose rotations and translations.

* :ref:`section-p3p`

* :ref:`section-five_point_essential_matrix`

* :ref:`section-four_point_homography`

* :ref:`section-eight_point`

* :ref:`section-dls_pnp`

* :ref:`section-four_point_focal_length`

* :ref:`section-five_point_focal_length_radial_distortion`

You can include the Pose module in your code with the following line:

.. code-block:: c++

  #include <theia/pose.h>

.. _section-p3p:

Perspective Three Point (P3P)
=============================

  .. function:: bool PoseFromThreePoints(const Eigen::Vector2d feature_position[3], const Eigen::Vector3d world_point[3], std::vector<Eigen::Matrix3d>* solution_rotations, std::vector<Eigen::Vector3d>* solution_translations)

    Computes camera pose using the three point algorithm and returns all
    possible solutions (up to 4). Follows steps from the paper "A Novel
    Parameterization of the Perspective-Three-Point Problem for a direct
    computation of Absolute Camera position and Orientation" by [Kneip]_\. This
    algorithm has been proven to be up to an order of magnitude faster than
    other methods. The output rotation and translation define world-to-camera
    transformation.

    ``feature_position``: Image points corresponding to model points. These should be
    calibrated image points as opposed to pixel values.

    ``world_point``: 3D location of features.

    ``solution_rotations``: the rotation matrix of the candidate solutions

    ``solution_translation``: the translation of the candidate solutions

    ``returns``: Whether the pose was computed successfully, along with the
    output parameters ``rotation`` and ``translation`` filled with the valid
    poses.

.. _section-five_point_essential_matrix:

Five Point Relative Pose
========================

  .. function:: bool FivePointRelativePose(const Eigen::Vector2d image1_points[5], const Eigen::Vector2d image2_points[5], std::vector<Eigen::Matrix3d>* rotation, std::vector<Eigen::Vector3d>* translation)

    Computes the relative pose between two cameras using 5 corresponding
    points. Algorithm is implemented based on "An Efficient Solution to the
    Five-Point Relative Pose Problem" by [Nister]_. The rotation and translation
    returned are defined such that :math:`E=t_x * R` and :math:`y^\top * E * x =
    0` where :math:`y` are points from image2 and :math:`x` are points from image1.

    ``image1_points``: Location of features on the image plane of image 1.

    ``image2_points``: Location of features on the image plane of image 2.

    ``returns``: Output the number of poses computed as well as the relative
    rotation and translation.


.. _section-four_point_homography:

Four Point Algorithm for Homography
===================================

  .. function:: bool FourPointHomography(const std::vector<Eigen::Vector2d>& image_1_points, const std::vector<Eigen::Vector2d>& image_2_points, Eigen::Matrix3d* homography)

    Computes the 2D `homography
    <http://en.wikipedia.org/wiki/Homography_(computer_vision)>`_ mapping points
    in image 1 to image 2 such that: :math:`x' = Hx` where :math:`x` is a point in
    image 1 and :math:`x'` is a point in image 2. The algorithm implemented is
    the DLT algorithm based on algorithm 4.2 in [HartleyZisserman]_.

    ``image_1_points``: Image points from image 1. At least 4 points must be
    passed in.

    ``image_2_points``: Image points from image 2. At least 4 points must be
    passed in.

    ``homography``: The computed 3x3 homography matrix.

.. _section-eight_point:

Eight Point Algorithm for Fundamental Matrix
============================================

  .. function:: bool NormalizedEightPoint(const std::vector<Eigen::Vector2d>& image_1_points, const std::vector<Eigen::Vector2d>& image_2_points, Eigen::Matrix3d* fundamental_matrix)

    Computes the `fundamental matrix
    <http://en.wikipedia.org/wiki/Fundamental_matrix_(computer_vision)>`_ relating
    image points between two images such that :math:`x' F x = 0` for all
    correspondences :math:`x` and :math:`x'` in images 1 and 2 respectively. The
    normalized eight point algorithm is a speedy estimation of the fundamental
    matrix (Alg 11.1 in [HartleyZisserman]_) that minimizes an algebraic error.

    ``image_1_points``: Image points from image 1. At least 8 points must be
    passed in.

    ``image_2_points``: Image points from image 2. At least 8 points must be
    passed in.

    ``fundamental_matrix``: The computed fundamental matrix.

    ``returns:`` true on success, false on failure.


  .. function:: bool GoldStandardEightPoint(const std::vector<Eigen::Vector2d>& image_1_points, const std::vector<Eigen::Vector2d>& image_2_points, Eigen::Matrix3d* fundamental_matrix)

    Computes the `fundamental matrix
    <http://en.wikipedia.org/wiki/Fundamental_matrix_(computer_vision)>`_
    relating image points between two images such that :math:`x' F x = 0` for
    all correspondences :math:`x` and :math:`x'` in images 1 and 2
    respectively. The gold standard algorithm computes an initial estimation of
    the fundmental matrix from the :func:`NormalizedEightPoint` then uses
    Levenberg-Marquardt to minimize the geometric error (i.e., reprojection
    error) according to algorithm 11.3 in [HartleyZisserman]_.

    ``image_1_points``: Image points from image 1. At least 8 points must be
    passed in.

    ``image_2_points``: Image points from image 2. At least 8 points must be
    passed in.

    ``fundamental_matrix``: The computed fundamental matrix.

    ``returns:`` true on success, false on failure.


.. _section-dls_pnp:

Perspective N-Point
===================

  .. function:: void DlsPnp(const std::vector<Eigen::Vector2d>& feature_position, const std::vector<Eigen::Vector3d>& world_point, std::vector<Eigen::Quaterniond>* solution_rotation, std::vector<Eigen::Vector3d>* solution_translation)

    Computes the camera pose using the Perspective N-point method from "A Direct
    Least-Squares (DLS) Method for PnP" by [Hesch]_ and Stergios Roumeliotis. This
    method is extremely scalable and highly accurate for the PnP problem. A
    minimum of 4 points are required, but there is no maximum number of points
    allowed as this is a least-squared approach. Theoretically, up to 27 solutions
    may be returned, but in practice only 4 real solutions arise and in almost all
    cases where n >= 6 there is only one solution which places the observed points
    in front of the camera. The returned rotation and translations are
    world-to-camera transformations.

    ``feature_position``: Normalized image rays corresponding to model points. Must
    contain at least 4 points.

    ``points_3d``: 3D location of features. Must correspond to the image_ray of
    the same index. Must contain the same number of points as image_ray, and at
    least 4.

    ``solution_rotation``: the rotation quaternion of the candidate solutions

    ``solution_translation``: the translation of the candidate solutions


.. _section-four_point_focal_length:

Four Point Focal Length
=======================

  .. function:: int FourPointPoseAndFocalLength(const std::vector<Eigen::Vector2d>& feature_positions, const std::vector<Eigen::Vector3d>& world_points, std::vector<Eigen::Matrix<double, 3, 4> >* projection_matrices)

    Computes the camera pose and unknown focal length of an image given four 2D-3D
    correspondences, following the method of [Bujnak]_. This method involves
    computing a grobner basis from a modified constraint of the focal length and
    pose projection.

    ``feature_position``: Normalized image rays corresponding to model points. Must
    contain at least 4 points.

    ``points_3d``: 3D location of features. Must correspond to the image_ray of
    the same index. Must contain the same number of points as image_ray, and at
    least 4.

    ``projection_matrices``: The solution world-to-camera projection matrices,
    inclusive of the unknown focal length. For a focal length f and a camera
    calibration matrix :math:`K=diag(f, f, 1)`, the projection matrices returned
    are of the form :math:`P = K * [R | t]`.


.. _section-five_point_focal_length_radial_distortion:

Five Point Focal Length and Radial Distortion
=============================================

  .. function:: bool FivePointFocalLengthRadialDistortion(const std::vector<Eigen::Vector2d>& feature_positions, const std::vector<Eigen::Vector3d>& world_points, const int num_radial_distortion_params, std::vector<Eigen::Matrix<double, 3, 4> >* projection_matrices, std::vector<std::vector<double> >* radial_distortions)

    Compute the absolute pose, focal length, and radial distortion of a camera
    using five 3D-to-2D correspondences [Kukelova]_. The method solves for the
    projection matrix (up to scale) by using a cross product constraint on the
    standard projection equation. This allows for simple solution to the first two
    rows of the projection matrix, and the third row (which contains the focal
    length and distortion parameters) can then be solved with SVD on the remaining
    constraint equations from the first row of the projection matrix. See the
    paper for more details.

    ``feature_positions``: the 2D location of image features. Exactly five
    features must be passed in.

    ``world_points``: 3D world points corresponding to the features
    observed. Exactly five points must be passed in.

    ``num_radial_distortion_params``: The number of radial distortion paramters to
	solve for. Must be 1, 2, or 3.

    ``projection_matrices``: Camera projection matrices (that encapsulate focal
	length). These solutions are only valid up to scale.

    ``radial_distortions``: Each entry of this vector contains a vector with the
    radial distortion parameters (up to 3, but however many were specified in
    ``num_radial_distortion_params``).

    ``return``: true if successful, false if not.
