#include "_eigen.hpp"
#define thread_num 8
std::pair<Matrix, Matrix> QR_decomposition_wrapper(Matrix const& mat, int type)
{
    if (type == 1) {
        return QR_decomposition_GS(mat);
    } else if (type == 2) {
        return QR_decomposition_HS(mat);
    } else {
        throw std::invalid_argument("QR_decomposition_wrapper: invalid type");
    }
}

std::pair<Matrix, Matrix> QR_decomposition_GS(Matrix const& mat)
{
    Matrix Q = gram_schmidt(mat);
    //Matrix R = Q.transpose() * mat;
    Matrix R = multiply_tile(Q.transpose() , mat, 32);
    return {Q, R};
}

std::pair<Matrix, Matrix> QR_decomposition_HS(Matrix const& mat)
{
    Matrix Q = Matrix::Identity(mat.nrow(), mat.ncol()), R = mat, H;
    //use 2D matrix to store the vectors, which is more cache friendly for parallel computing
    std::vector<std::vector<double>> vv(mat.ncol(), std::vector<double>(mat.nrow()));
    for (size_t i = 0; i<mat.ncol()-1; ++i) 
    {
        size_t j;
        #pragma omp parallel for num_threads(thread_num)
        for (j = i; j<mat.nrow(); ++j) 
        {
            vv[i][j] = R(j, i);
        }
        H = householder(vv[i], mat.nrow(), i);
        // use multiply_tile to improve cache hit rate & palallel computing
        R = multiply_tile(H, R, 32);
        //R = H * R;
        Q = multiply_tile(Q, H, 32);
        //Q = Q * H;
    }
    /*
    std::vector<std::vector<double>> vv(mat.nrow(), std::vector<double>(mat.ncol()-1));
    #pragma omp parallel for num_threads(thread_num)
    for (size_t i = 0; i<mat.ncol()-1; ++i) 
    {
        for (size_t j = 0; j<mat.nrow(); ++j) 
        {
            if (j >= i)
                vv[i][j] = R(j, i);
            else
                vv[i][j] = 0;
        }
        H = householder(vv[i], mat.nrow(), i);
        R = H * R;
        Q = Q * H;
    }
    */
    return {Q, R};    
}

std::vector<double> find_eigenvalue(Matrix const& mat)
{
    size_t iteration = 5000000;
    
    Matrix A_old = mat;
    Matrix A_new = mat;
    double diff = std::numeric_limits<double>::max();
    double tolerance = 1e-10;
    size_t i = 0;
    while (diff > tolerance && i < iteration) {
        A_old = A_new;
        std::pair<Matrix, Matrix> QR = QR_decomposition_GS(A_old);
        A_new = QR.second * QR.first;
        diff = 0;
        for (size_t i = 0; i < A_old.nrow(); i++) {
            diff += std::abs(A_old(i, i) - A_new(i, i));
        }
        i++;
    }
    std::vector<double> eigenvalues;
    for (size_t i = 0; i < A_new.nrow(); i++) {
        eigenvalues.push_back(A_new(i, i)); // The eigenvalues are the diagonal elements
    }
    return eigenvalues;
}

std::vector<double> find_eigenvector(Matrix const& mat, double eigenvalue)
{
    return {};
}

bool is_orthogonal_matrix(Matrix const& mat)
{
    Matrix T = mat.transpose();
    Matrix I = mat * T;
    for (size_t i = 0; i < I.nrow(); i++) {
        for (size_t j = 0; j < I.ncol(); j++) {
            if (i == j) {
                if (std::abs(I(i, j) - 1) > EPSILON) {
                    return false;
                }
            } else {
                if (std::abs(I(i, j)) > EPSILON) {
                    return false;
                }
            }
        }
    }
    return true;
}
