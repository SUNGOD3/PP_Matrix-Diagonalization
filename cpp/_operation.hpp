#ifndef __OPERATION_HPP__
#define __OPERATION_HPP__

#include "_matrix.hpp"
#include "cmath"

Matrix multiply_tile(Matrix const& mat1, Matrix const& mat2, size_t tsize);
Matrix multiply_naive(Matrix const& mat1, Matrix const& mat2);

double dot_product(const std::vector<double>& a, const std::vector<double>& b);
std::vector<double> subtract(const std::vector<double> &a, const std::vector<double> &b);

std::vector<double> subtract_projection(const std::vector<double>& a, const std::vector<double>& b);
std::vector<double> scalar_vector(const std::vector<double> &a, double scalar);
std::vector<double> subtract_vector(const std::vector<double> &a, const std::vector<double> &b);
std::vector<double> normalize(const std::vector<double>& a);
Matrix gram_schmidt(Matrix const& mat);
Matrix householder(std::vector<double>& x, size_t n, size_t e);
std::vector<std::vector<double>> null_space(Matrix const& mat, size_t valid_row);

#endif