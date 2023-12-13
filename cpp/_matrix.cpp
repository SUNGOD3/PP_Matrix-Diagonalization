#include "_matrix.hpp"
Matrix & Matrix::operator=(Matrix const & mat)
{
    if (this == &mat) return *this;
    if (this->m_ncol != mat.m_ncol || this->m_nrow != mat.m_nrow)
    {
        reset_buffer(mat.m_nrow, mat.m_ncol);
    }

    for (size_t i = 0; i < mat.m_nrow; ++i)
    {
        for (size_t j = 0; j < mat.m_ncol; ++j)
        {
            this->m_buffer[i*m_nrow+j] = mat(i, j);
        }
    }
    
    return *this;
}

Matrix & Matrix::operator=(Matrix && mat)
{
    if (this == &mat) return *this;
    reset_buffer(0, 0);
    std::swap(this->m_ncol, mat.m_ncol);
    std::swap(this->m_nrow, mat.m_nrow);
    std::swap(this->m_buffer, mat.m_buffer);
    return *this;
}

Matrix & Matrix::operator=(std::vector<double> const & vec)
{
    if (vec.size() != m_nrow * m_ncol)
    {
        throw std::out_of_range("Matrix::operator=: vector size differs from matrix size");
    }

    size_t k = 0;
    for (size_t i = 0; i < m_nrow; ++i)
    {
        for (size_t j = 0; j < m_ncol; ++j)
        {
            m_buffer[i*m_nrow+j] = vec[k++];
        }
    }
    return *this;
}

bool Matrix::operator==(Matrix const & mat) const
{
    if (mat.m_ncol != m_ncol || mat.m_nrow != m_nrow) return false;
    for (size_t i = 0; i < mat.m_nrow; ++i)
    {
        for (size_t j = 0; j < mat.m_ncol; ++j)
        {
            if(mat(i, j) != m_buffer[i*m_nrow+j])return false;
        }
    }
    return true;
}

std::ostream & operator<<(std::ostream & os, Matrix const &mat)
{
    for (size_t i = 0; i < mat.m_nrow; ++i)
    {
        for (size_t j = 0; j < mat.m_ncol; ++j)
        {
            os << std::setw(10) << std::fixed << std::setprecision(8) << std::right << mat(i, j) << " ";
        }
        os << std::endl;
    }
    return os;
}


double Matrix::operator() (size_t row, size_t col) const
{
    //add the boundary check
    if (row >= m_nrow || col >= m_ncol)
    {
        throw std::out_of_range("Matrix::operator(): index out of range");
    }

    return m_buffer[row*m_ncol + col];
}

double & Matrix::operator() (size_t row, size_t col)
{
    //add the boundary check
    if (row >= m_nrow || col >= m_ncol)
    {
        throw std::out_of_range("Matrix::operator(): index out of range");
    }

    return m_buffer[row*m_ncol + col];
}

// return a row vector
std::vector<double> Matrix::operator() (size_t row) const
{
    if (row >= m_nrow)
    {
        throw std::out_of_range("Matrix::operator(): index out of range");
    }

    std::vector<double> vec(m_ncol);
    for (size_t i=0; i<m_ncol; ++i)
    {
        vec[i] = m_buffer[row*m_ncol + i];
    }
    return vec;
}

Matrix & Matrix::operator*(Matrix const & mat) const
{
    if (m_ncol != mat.m_nrow)
    {
        throw std::out_of_range("Matrix::operator*: invalid matrix dimensions for multiplication");
    }

    Matrix *result = new Matrix(m_nrow, mat.m_ncol);

    for (size_t i=0; i<m_nrow; ++i)
    {
        for (size_t j=0; j<mat.m_ncol; ++j)
        {
            double v = 0;
            for (size_t k=0; k<m_ncol; ++k)
            {
                v += (*this)(i,k) * mat(k,j);
            }
            (*result)(i, j) = v;
        }
    }
    return *result;
}

Matrix & Matrix::operator*(std::vector<double> const & vec)
{
    if (m_ncol != vec.size())
    {
        throw std::out_of_range("Matrix::operator*: matrix column differs from vector size");
    }

    Matrix *result = new Matrix(m_nrow, 1);

    for (size_t i=0; i<m_nrow; ++i)
    {
        double v = 0;
        for (size_t j=0; j<m_ncol; ++j)
        {
            v += (*this)(i,j) * vec[j];
        }
        (*result)(i, 0) = v;
    }

    return *result;
}

Matrix & Matrix::operator*(double scalar) const
{
    Matrix *result = new Matrix(m_nrow, m_ncol);

    for (size_t i=0; i<m_nrow; ++i)
    {
        for (size_t j=0; j<m_ncol; ++j)
        {
            (*result)(i, j) = (*this)(i, j) * scalar;
        }
    }

    return *result;
}

Matrix & Matrix::operator+(Matrix const & mat) const
{
    if (m_ncol != mat.m_ncol || m_nrow != mat.m_nrow)
    {
        throw std::out_of_range("Matrix::operator+: matrix dimensions differ");
    }

    Matrix *result = new Matrix(m_nrow, m_ncol);

    for (size_t i=0; i<m_nrow; ++i)
    {
        for (size_t j=0; j<m_ncol; ++j)
        {
            (*result)(i, j) = (*this)(i, j) + mat(i, j);
        }
    }

    return *result;
}

Matrix & Matrix::operator-(Matrix const & mat) const
{
    if (m_ncol != mat.m_ncol || m_nrow != mat.m_nrow)
    {
        throw std::out_of_range("Matrix::operator-: matrix dimensions differ");
    }

    Matrix *result = new Matrix(m_nrow, m_ncol);

    for (size_t i=0; i<m_nrow; ++i)
    {
        for (size_t j=0; j<m_ncol; ++j)
        {
            (*result)(i, j) = (*this)(i, j) - mat(i, j);
        }
    }

    return *result;
}

Matrix Matrix::transpose() const
{
    Matrix result(m_ncol, m_nrow);
    size_t i;
    #pragma omp parallel for num_threads(thread_num)
    for (i=0; i<m_nrow; ++i)
    {
        size_t j;
        for (j=0; j<m_ncol; ++j)
        {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}
//operator overloading for vector
std::ostream & operator<<(std::ostream & os, std::vector<double> const & vec)
{
    for (size_t i=0; i<vec.size(); ++i)
    {
        os << vec[i] << " ";
    }
    os << std::endl;
    return os;
}
