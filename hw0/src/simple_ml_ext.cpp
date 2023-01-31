#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

using namespace std;

namespace py = pybind11;

void sub(float *a, float *b, size_t rows, size_t cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            a[i * cols + j] -= b[i * cols + j];
        }
    }
}

void matmul(const float *a, const float *b, float *r, size_t a_rows, size_t a_cols, size_t b_cols)
{
    for (int i = 0; i < a_rows; i++) {
        for (int k = 0; k < b_cols; k++) {
            for (int j = 0; j < a_cols; j++) {
                r[i*b_cols+k] += a[a_cols*i+j] * b[b_cols*j+k];
            }
        }
    }
}

void compute_softmax_grad(float *z, const unsigned char *by, size_t batch, size_t k)
{
    for (int i = 0; i < batch; i++) {
        float sum = 0;
        for (int j = 0; j < k; j++) {
            z[i*k+j] = std::exp(z[i * k + j]);
            sum += z[i*k+j];
        }
        for (int j = 0; j < k; j++) {
            z[i*k+j] /= sum;
        }
        z[i * k + (by[i] - '0')] -= 1;
    }
}

void mul_scalar(float *x, size_t rows, size_t cols, float multiplier)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            x[i*cols + j] *= multiplier;
        }
    }
}

void divide_scalar(float *x, size_t rows, size_t cols, float divisor)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            x[i*cols + j] /= divisor;
        }
    }
}


void transpose(const float *x, float *r, size_t rows, size_t cols)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            r[j*cols+i] = x[i*cols + j];
        }
    }
}


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    int iter_num = m / batch;
    float *z = (float*)malloc(batch*k*sizeof(float));
    float *grad = (float*)malloc(n*k*sizeof(float));
    float *transpose_x = (float*)malloc(n*batch*sizeof(float));
    for (int iter = 0; iter < iter_num; iter++)
    {
        const float *bX = X + iter * batch * n;
        const unsigned char *by = y + iter * batch;
        matmul(bX, theta, z, batch, n, k);
        compute_softmax_grad(z, by, batch, k);
        transpose(bX, transpose_x, batch, n);
        matmul(transpose_x, z, grad, n, batch, k);
        divide_scalar(grad, n, k, batch);
        mul_scalar(grad, n, k, lr);
        sub(theta, grad, n, k);
    }
    free(z);
    free(grad);
    free(transpose_x);
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
