#ifndef __EIGEN_HPP__
#define __EIGEN_HPP__

#include "_matrix.hpp"
#include "_operation.hpp"
#include <limits>

std::pair<Matrix, Matrix> QR_decomposition_GS(Matrix const& mat);
std::pair<Matrix, Matrix> QR_decomposition_HS(Matrix const& mat);
std::pair<Matrix, Matrix> QR_decomposition_wrapper(Matrix const& mat, int type = 1);
std::vector<double> find_eigenvalue(Matrix const& mat);
std::vector<double> find_eigenvector(Matrix const& mat, double eigenvalue);
bool is_orthogonal_matrix(Matrix const& mat);

#endif