#include "_operation.hpp"
//#include <execution> 
#define thread_num 16


double dot_product(const std::vector<double>& a, const std::vector<double>& b) 
{
    //Avoid duplication of parallelized versions
    double result = std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
    return result;
}

double dot_product_p(const std::vector<double>& a) 
{
    double result = 0;
    size_t i;
    #pragma omp parallel for num_threads(thread_num) reduction(+:result)
    for (i = 0; i < a.size(); i++) {
        result += a[i] * a[i];
    }
    return result;
}

std::vector<double> normalize(std::vector<double>& a) 
{
    //Avoid duplication of parallelized versions
    double norm = sqrt(dot_product(a, a));
    transform(a.begin(), a.end(), a.begin(), [norm](double &c){ return c/norm; });
    return a;
}

std::vector<double> normalize_p(std::vector<double>& a) 
{
    double norm = sqrt(dot_product_p(a));
    #pragma omp parallel for num_threads(thread_num)
    for (size_t i = 0; i < a.size(); i++) {
        a[i] /= norm;
    }
    return a;
}

bool is_zero_vector(const std::vector<double>& a) 
{
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] > EPSILON) {
            return false;
        }
    }
    return true;
    //return std::all_of(std::execution::par, a.begin(), a.end(), [](double x) { return fabs(x) <= EPSILON; });
}

std::vector<double> ssdd(const std::vector<double> &a, const std::vector<double> &b, const std::vector<double> &c, const double &cc)
{
    const size_t n = a.size();
    std::vector<double> result(n);
    double scalar0 = 0.0;
    //#pragma omp parallel for num_threads(thread_num) reduction(+:scalar0, scalar1)
    for(size_t i = 0; i < n; i++){
        scalar0 += b[i] * c[i];
    }
    double scalar = scalar0 / cc;
    //#pragma omp parallel for num_threads(thread_num)
    for (size_t i = 0; i < n; i++) {
        result[i] = a[i] - scalar * c[i];
    }
    return result;
}

Matrix gram_schmidt(Matrix const& mat) 
{
    const size_t n = mat.ncol();
    Matrix Q(n, n), TansMat = mat.transpose();
    std:: vector <double> Qc(n);
    size_t num_non_zero_vec = 0;
    for (size_t i = 0; i < n; i++) {
        std::vector<double> vi = TansMat(i);
        //critical section
        //use formula simplification to reduce the number of operations 
        for(size_t j = 0; j < num_non_zero_vec; j++)
            vi = ssdd(vi, TansMat(i), Q(j), Qc[j]);
        if(!is_zero_vector(vi)) {
            vi = normalize_p(vi);
            double sum = 0;
            //size_t j;
            #pragma omp parallel for num_threads(thread_num) reduction(+:sum)
            for(size_t j = 0; j < n; j++) {
                Q(num_non_zero_vec, j) = vi[j];
                //Move some operations that can be parallelized here
                sum += vi[j] * vi[j];
            }
            Qc[num_non_zero_vec++]=sum;
        }
    }

    std::vector<std::vector<double>> null_vecs = null_space(Q, num_non_zero_vec);
    #pragma omp parallel for num_threads(thread_num)
    for (size_t i=num_non_zero_vec; i<n; i++) {
        std::vector<double> vi = normalize(null_vecs[i-num_non_zero_vec]);
        for (size_t j=0; j<n; j++) 
            Q(i, j) = vi[j];
    }
    return Q.transpose();
}

std::vector<std::vector<double>> null_space(Matrix const& mat, size_t const valid_row)
{
    // use const instead of function call (e.g. mat.nrow())
    const size_t n = mat.ncol();
    // Step 1: Create an augmented matrix [A|0]
    // pallarelize this part at _matrix.cpp
    Matrix augmented_mat = mat;
    // std::cout<<"augmented_mat: \n"<<augmented_mat<<std::endl;
    // std::cout<<"pass1"<<std::endl;
    // Step 2: Perform Gaussian elimination
    for (size_t r = 0, lead = 0; r < valid_row; r++, lead++) {
        if (lead >= n)
            return {};
        size_t i = r;
        //critical section
        for(;augmented_mat(i, lead) == 0;) {
            i++;
            if (i == valid_row) {
                i = r;
                lead++;
                if (n == lead)
                    return {};
            }
        }
        // Swap rows i and r
        // if i==r then do nothing
        if(i!=r){
            #pragma omp parallel for num_threads(thread_num)
            for (size_t j = 0; j < n; j++) {
                std::swap(augmented_mat(i, j), augmented_mat(r, j));
            }
        }
        auto lv = augmented_mat(r, lead);
        #pragma omp parallel for num_threads(thread_num)
        for (size_t j = 0; j < n; j++) {
            augmented_mat(r, j) /= lv;
        }
        #pragma omp parallel for num_threads(thread_num) schedule(dynamic)
        for (i = 0; i < valid_row; i++) {
            if (i != r) {
                auto sub = augmented_mat(i, lead);
                for (size_t j = 0; j < n; j++) {
                    augmented_mat(i, j) -= (augmented_mat(r, j) * sub);
                }
            }
        }
    }
    // std::cout<<"pass2"<<std::endl;
    // Step 3: Identify pivot columns
    std::vector<bool> isPivot(n, false);
    // 
    int nsv = 0;
    size_t i, j;
    #pragma omp parallel for num_threads(thread_num) schedule(dynamic) reduction(+:nsv)
    for (i = 0; i < valid_row; i++) {
        for (j = 0; j < n; j++) {
            if (augmented_mat(i, j)) { // not balanced -> dynamic
                isPivot[j] = true;
                ++nsv; // count the size of null space vector at the step 3
                break;
            }
        }
    }

    // Step 4: Solve the system for each free variable
    int sz = 0;
    std::vector<std::vector<double>> null_space_vectors(nsv); // initialize the null space vector size
    #pragma omp parallel for num_threads(thread_num) reduction(+:sz) schedule(dynamic)
    for (i = 0; i < isPivot.size(); i++) {
        if (!isPivot[i]) {
            std::vector<double> special_solution(mat.ncol(), 0);
            special_solution[i] = 1;
            size_t r;
            for (r = 0; r < valid_row; r++)
                special_solution[r] = -augmented_mat(r, i);
            #pragma omp critical 
            {
                null_space_vectors[sz++] = special_solution;
            }
        }
    }

    // Return the null space
    return null_space_vectors;
}

Matrix householder(std::vector<double>& x, size_t n, size_t e)
{
    // x = x - ||x||e
    x[e] = x[e] - sqrt(dot_product(x, x));

    // x/||x||
    x = normalize(x);

    Matrix H = Matrix::Identity(n, n);
    size_t i;
    //#pragma omp parallel for num_threads(thread_num) private(j) schedule(dynamic)
    //To ensure QR is correct, parallelization is not used
    for (i = 0; i < n; ++i)
    {
        H(i, i) -= 2 * x[i] * x[i];
        x[i] += x[i];
        size_t j;
        for (j = i+1; j < n; j++)
        {
            H(i, j) = (H(j, i) = -x[i] * x[j]);
        }
    }
    // h = householder matrix = I - 2 * vvt
    

    return H;
}

//function to multiply two matrix m*n & n*p with tile size and m, n, p are all multiples of tile size
Matrix multiply_tile(Matrix const& mat1, Matrix const& mat2, size_t tsize)
{
    //Check whether the multiplication is possible
    if (mat1.ncol() != mat2.nrow())
    {
        throw std::out_of_range("invalid matrix dimensions for multiplication");
    }

    // Create a new matrix to store the result of the multiplication.
    Matrix result(mat1.nrow(), mat2.ncol());
    const size_t m = mat1.nrow();
    const size_t n = mat2.ncol();
    const size_t p = mat1.ncol();
    // Divide the matrices into tiles of size tile_m x tile_n and tile_n x tile_p.
    size_t i, j, k, ii, jj, kk;
    double sum;
    #pragma omp parallel for private(i, j, k, ii, jj ,kk ,sum) collapse(2) num_threads(thread_num)
    for (i = 0; i < m; i += tsize)
    {
        for (j = 0; j < n; j += tsize)
        {
            for (k = 0; k < p; k += tsize)
            {
                size_t upper_i = std::min(i + tsize, m);
                size_t upper_j = std::min(j + tsize, n);
                size_t upper_k = std::min(k + tsize, p);
                for (ii = i; ii < upper_i ; ++ii)
                {
                    for (jj = j; jj < upper_j ; ++jj) 
                    {
                        sum = .0;
                        for (kk = k; kk < upper_k; ++kk)
                        {
                            sum += mat1(ii, kk) * mat2(kk, jj);
                        }
                        result(ii, jj) += sum;
                    }
                }
            }
        }
    }

    return result;
}

Matrix multiply_naive(Matrix const& mat1, Matrix const& mat2)
{
    // Check if the dimensions of the matrices are valid for multiplication.
    if (mat1.ncol() != mat2.nrow())
    {
        throw std::out_of_range("invalid matrix dimensions for multiplication");
    }

    // Create a new matrix to store the result of the multiplication.
    Matrix result(mat1.nrow(), mat2.ncol());

    // Multiply each element of mat1 with the corresponding element of mat2 and add the result to the corresponding element of the result matrix.
    for (size_t i = 0; i < mat1.nrow(); ++i)
    {
        for (size_t j = 0; j < mat2.ncol(); ++j)
        {
	        double sum = .0;
            for (size_t k = 0; k < mat1.ncol(); ++k)
            {
                sum += mat1(i, k) * mat2(k, j);
            }
	        result(i, j) = sum;
        }
    }

    return result;
}