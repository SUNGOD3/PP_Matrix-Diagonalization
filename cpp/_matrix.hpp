#ifndef __MATRIX_HPP__
#define __MATRIX_HPP__
#define thread_num 16
constexpr double EPSILON = 0.000001;

#include <iostream>
#include <vector>
#include <cmath>
#include <utility>
#include <iomanip>
#include <cstring>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class Matrix {

public:
    Matrix(): m_nrow(0), m_ncol(0), m_buffer(nullptr) {}

    Matrix(size_t nrow, size_t ncol): m_nrow(nrow), m_ncol(ncol) 
    {
        reset_buffer(nrow, ncol);
    }

    Matrix(const Matrix &mat): m_nrow(mat.m_nrow), m_ncol(mat.m_ncol)
    {
        reset_buffer(m_nrow, m_ncol);
        for (size_t i = 0; i < m_nrow; ++i)
        {
            for (size_t j = 0; j < m_ncol; ++j)
            {
                m_buffer[i*m_nrow+j] = mat(i, j);
            }
        }
    }

    Matrix(size_t nrow, size_t ncol, std::vector<double> const & vec): m_nrow(nrow), m_ncol(ncol)
    {
        if (vec.size() != nrow * ncol)
        {
            throw std::out_of_range("Matrix::Matrix(): vector size differs from matrix size");
        }

        reset_buffer(nrow, ncol);
        (*this) = vec;
    }

    Matrix(Matrix &&mat): m_nrow(mat.m_nrow), m_ncol(mat.m_ncol)
    {
        reset_buffer(0, 0);
        std::swap(m_nrow, mat.m_nrow);
        std::swap(m_ncol, mat.m_ncol);
        std::swap(m_buffer, mat.m_buffer);
    }

    ~Matrix() { delete[] m_buffer; }

    Matrix transpose() const;    

    double* data() const { return m_buffer; }
    size_t nrow() const { return m_nrow; }
    size_t ncol() const { return m_ncol; }

    double   operator() (size_t row, size_t col) const; //getter
    double & operator() (size_t row, size_t col); //setter
    std::vector<double> operator() (size_t row) const; //getter
    
    Matrix & operator=(Matrix const & mat); // copy assignment operator
    Matrix & operator=(Matrix && mat); // move assignment operator
    Matrix & operator=(std::vector<double> const & vec);
    Matrix & operator*(Matrix const & mat) const;
    Matrix & operator*(std::vector<double> const & vec);
    Matrix & operator*(double scalar) const;
    Matrix & operator+(Matrix const & mat) const;
    Matrix & operator-(Matrix const & mat) const;

    bool operator==(Matrix const & mat) const;

    static Matrix Identity(size_t nrow, size_t ncol) {
        Matrix ret(nrow, ncol);
        #pragma omp parallel for num_threads(thread_num)
        for (size_t i = 0; i < nrow; i++) {
            ret(i, i) = 1;
        }
        return ret;
    }
    
    py::buffer_info buffer_info() const {
        return py::buffer_info(
            m_buffer,
            sizeof(double),
            py::format_descriptor<double>::format(),
            2,
            { m_nrow, m_ncol },
            { m_ncol * sizeof(double), sizeof(double) }
        );
    }

    void reset_buffer(size_t nrow, size_t ncol) 
    {
        if (m_buffer != nullptr) delete[] m_buffer;
        size_t buff_size = nrow * ncol;
        m_buffer = buff_size > 0 ? new double[buff_size] : nullptr;
        m_nrow = nrow;
        m_ncol = ncol;
        for (size_t i = 0; i < buff_size; ++i)
        {
            m_buffer[i] = 0;
        }
    }

    friend std::ostream & operator<<(std::ostream & os, Matrix const & mat);
private:
    size_t m_nrow = 0;
    size_t m_ncol = 0;
    double * m_buffer = nullptr;
    
};

std::ostream & operator<<(std::ostream & os, std::vector<double> const & vec);

#endif
