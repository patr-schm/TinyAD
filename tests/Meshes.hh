/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#include <Eigen/Core>

/**
 * Triangle mesh in the plane with 6 vertices and 4 faces.
 * Init stretched version.
 * Position constraints at two vertices, that require 90° ccw rotation.
 */
template <typename PassiveT>
void planar_test_mesh(
        Eigen::MatrixX<PassiveT>& _V_rest,
        Eigen::MatrixX<PassiveT>& _V_init,
        Eigen::MatrixXi& _F,
        std::vector<Eigen::Index>& _b,
        std::vector<Eigen::Vector2<PassiveT>>& _bc)
{
    _V_rest = Eigen::MatrixX<PassiveT> (6, 2);
    _V_rest << 0.0, 0.0,
          1.0, 0.0,
          0.0, 1.0,
          1.0, 1.0,
          0.0, 2.0,
          1.0, 2.0;

    // Init stretched version
    _V_init = _V_rest;
    _V_init.col(0) *= 0.5;
    _V_init.col(1) *= 0.25;

    _F = Eigen::MatrixXi(4, 3);
    _F << 0, 1, 2,
          1, 3, 2,
          2, 3, 5,
          2, 5, 4;

    // Add position constraints at two vertices, that require 90° ccw rotation.
    _b = { 0, 4 };
    _bc = { { 0.0, 0.0 }, { -2.0, 0.0 } };
}
