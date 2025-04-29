/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#include <gtest/gtest.h>
#include <TinyAD/ScalarFunction.hh>
#include "Meshes.hh"

TEST(DynamicElementsTest, DynamicElementsTest)
{
    auto func = TinyAD::scalar_function<2>(TinyAD::range(4));
    
    // Instantiate two element groups with valences 3 and 1
    func.add_elements_dynamic<3, 1>(TinyAD::range(4), [&] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        using T = TINYAD_SCALAR_TYPE(element);
        int e = (int)element.handle;

        // Element e accesses e many variables
        Eigen::Vector2<T> sum = Eigen::Vector2<T>::Zero();
        for (int v = 0; v < e; ++v)
            sum += element.variables(v);

        return sum.squaredNorm();
    });

    ASSERT_EQ(func.objective_terms.size(), 2);
    ASSERT_EQ(func.objective_terms[0]->n_elements(), 2);
    ASSERT_EQ(func.objective_terms[1]->n_elements(), 2);

    Eigen::VectorXd x = Eigen::VectorXd::Ones(4 * 2);
    auto [f, g, H_proj] = func.eval_with_hessian_proj(x);
}

namespace
{

Eigen::SparseMatrix<double> reference_laplace(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F)
{
    Eigen::SparseMatrix<double> L(V.rows(), V.rows());
    for (int f = 0; f < F.rows(); ++f)
    {
        for (int i = 0; i < 3; ++i)
        {
            int v1 = F(f, i);
            int v2 = F(f, (i + 1) % 3);
            L.coeffRef(v1, v1) += 1.0;
            L.coeffRef(v1, v2) -= 1.0;
        }
    }

    L.makeCompressed();
    return L;
}

}

TEST(DynamicElementsTest, LaplaceTest)
{
    // Load bunny mesh (closed surface)
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    bunny_closed_mesh(V, F);

    // Compute reference Laplacian
    Eigen::SparseMatrix<double> L_ref = reference_laplace(V, F);
    L_ref.makeCompressed();

    // Compute Laplacian as Hessian of Dirichlet energy using static triangle elements
    {
        auto func = TinyAD::scalar_function<1>(TinyAD::range(V.rows()));
        func.add_elements<3>(TinyAD::range(F.rows()), [&] (auto& element)
        {
            using T = TINYAD_SCALAR_TYPE(element);
            int f = (int)element.handle;
            T a = element.variable(F(f, 0));
            T b = element.variable(F(f, 1));
            T c = element.variable(F(f, 2));
            return 0.25 * (sqr(b - a) + sqr(c - b) + sqr(a - c));
        });

        Eigen::SparseMatrix<double> L_tri = func.eval_hessian_of_quadratic();
        L_tri.makeCompressed();

        ASSERT_EQ(L_tri.rows(), L_ref.rows());
        ASSERT_EQ(L_tri.cols(), L_ref.cols());
        ASSERT_EQ(L_tri.nonZeros(), L_ref.nonZeros());
        ASSERT_NEAR((L_tri - L_ref).norm(), 0.0, 1e-12) << "Laplacian mismatch!";
    }

    // Compute Laplacian as Hessian of Dirichlet energy using dynamic vertex elements.
    {
        // Compute vertex to neighbor mapping
        std::vector<std::vector<int>> vertex_to_neighbors(V.rows());
        for (int f = 0; f < F.rows(); ++f)
        {
            for (int i = 0; i < 3; ++i)
            {
                int v1 = F(f, i);
                int v2 = F(f, (i + 1) % 3);
                vertex_to_neighbors[v1].push_back(v2);
            }
        }

        // Output distribution of vertex valences
        std::map<int, int> valence_histogram;
        for (const auto& neighbors : vertex_to_neighbors)
            ++valence_histogram[(int)neighbors.size()];
        for (const auto& [valence, count] : valence_histogram)
            TINYAD_INFO("Valence " << valence << ": " << count << " vertices");

        // We compile code for one-ring elements of sizes 6, 7, 8, and 10.
        // The maxiumum vertex valence in this mesh is 9.
        // Including the vertex itself, this gives a maximum element valence of 10.
        auto func = TinyAD::scalar_function<1>(TinyAD::range(V.rows()));
        func.add_elements_dynamic<6, 7, 8, 10>(TinyAD::range(V.rows()), [&] (auto& element)
        {
            using T = TINYAD_SCALAR_TYPE(element);
            int v = (int)element.handle;
            
            T v_val = element.variable(v);
            std::vector<T> neigh_vals(vertex_to_neighbors[v].size());
            for (int i = 0; i < vertex_to_neighbors[v].size(); ++i)
                neigh_vals[i] = element.variable(vertex_to_neighbors[v][i]);
            
            T dirichlet = 0.0;
            for (int i = 0; i < neigh_vals.size(); ++i)
                dirichlet += 0.25 * sqr(v_val - neigh_vals[i]);
            
            return dirichlet;
        });

        Eigen::SparseMatrix<double> L_vert = func.eval_hessian_of_quadratic();
        L_vert.makeCompressed();

        ASSERT_EQ(L_vert.rows(), L_ref.rows());
        ASSERT_EQ(L_vert.cols(), L_ref.cols());
        // ASSERT_EQ(L_vert.nonZeros(), L_ref.nonZeros()); // NNZ are different, we add a bunch of extra "zeros"! Consider pruning the resulting Hessian.
        ASSERT_NEAR((L_vert - L_ref).norm(), 0.0, 1e-12) << "Laplacian mismatch!";
    }
}