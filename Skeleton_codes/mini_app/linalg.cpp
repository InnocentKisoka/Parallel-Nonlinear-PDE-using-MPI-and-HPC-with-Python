// linear algebra subroutines
// Ben Cumming @ CSCS

#include <iostream>

#include <cmath>
#include <cstdio>

#include <mpi.h>

#include <immintrin.h>

#include "linalg.h"
#include "operators.h"
#include "stats.h"
#include "data.h"

namespace linalg {

bool cg_initialized = false;
Field r;
Field Ap;
Field p;
Field Fx;
Field Fxold;
Field v;
Field xold;

using namespace operators;
using namespace stats;
using data::Field;

// initialize temporary storage fields used by the cg solver
// I do this here so that the fields are persistent between calls
// to the CG solver. This is useful if we want to avoid malloc/free calls
// on the device for the OpenACC implementation (feel free to suggest a better
// method for doing this)
void cg_init(int nx, int ny)
{
    Ap.init(nx,ny);
    r.init(nx,ny);
    p.init(nx,ny);
    Fx.init(nx,ny);
    Fxold.init(nx,ny);
    v.init(nx,ny);
    xold.init(nx,ny);

    cg_initialized = true;
}

////////////////////////////////////////////////////////////////////////////////
//  blas level 1 reductions
////////////////////////////////////////////////////////////////////////////////

// computes the inner product of x and y
// x and y are vectors on length N
double hpc_dot(Field const& x, Field const& y)
{
    double result = 0;
    double result_global = 0;
    int N = y.length();
	
	
	
	double tmp1,tmp2,tmp3,tmp4;
	tmp1 = tmp2 = tmp3 = tmp4 = 0.f;
	int i;	
	for(i=0;i<N-(N%4);i+=4)
	{
		tmp1 += x[i+0] * y[i+0];
		tmp2 += x[i+1] * y[i+1];
		tmp3 += x[i+2] * y[i+2];
		tmp4 += x[i+3] * y[i+3];
	}
    for (; i < N; i++)
    	result += x[i] * y[i];	
		result += (tmp1 + tmp2 + tmp3 + tmp4);
	

	/*
	 *
	 * 1)
	 *
	for(i=0;i<N;++i)
		result += x[i]*y[i];
	*/
	MPI_Allreduce(&result, &result_global, 1, MPI_DOUBLE, MPI_SUM, data::domain.comm_cart);
    return result_global;
}

// computes the 2-norm of x
// x is a vector on length N
double hpc_norm2(Field const& x)
{
    double result = 0;
    double result_global = 0;
    int N = x.length();
	double tmp1,tmp2,tmp3,tmp4;
	tmp1 = tmp2 = tmp3 = tmp4 = 0.f;
	int i;
	// loop unrolling for some extra performance
	for(i=0;i<N-N%4;i+=4)
	{
		tmp1 += x[i+0] * x[i+0];
		tmp2 += x[i+1] * x[i+1];
		tmp3 += x[i+2] * x[i+2];
		tmp4 += x[i+3] * x[i+3];
	}
    for (; i < N; i++)
        result += x[i] * x[i];	
		result += (tmp1 + tmp2 + tmp3 + tmp4);
	/*
	for(i=0;i<N;++i)
		result += x[i]*x[i];
	*/
	MPI_Allreduce(&result, &result_global, 1, MPI_DOUBLE, MPI_SUM, data::domain.comm_cart);

    return sqrt(result_global);
}

// sets entries in a vector to value
// x is a vector on length N
// value is a scalar
void hpc_fill(Field& x, const double value)
{
    int N = x.length();

    for (int i = 0; i < N; i++)
        x[i] = value;
}

////////////////////////////////////////////////////////////////////////////////
//  blas level 1 vector-vector operations
////////////////////////////////////////////////////////////////////////////////

// computes y := alpha*x + y
// x and y are vectors on length N
// alpha is a scalar
void hpc_axpy(Field& y, const double alpha, Field const& x)
{
    int N = y.length();

    for (int i = 0; i < N; i++)
        y[i] += alpha * x[i];
}

// computes y = x + alpha*(l-r)
// y, x, l and r are vectors of length N
// alpha is a scalar
void hpc_add_scaled_diff(Field& y, Field const& x, const double alpha,
    Field const& l, Field const& r)
{
    int N = y.length();

    for (int i = 0; i < N; i++)
        y[i] = x[i] + alpha * (l[i] - r[i]);
}

// computes y = alpha*(l-r)
// y, l and r are vectors of length N
// alpha is a scalar
void hpc_scaled_diff(Field& y, const double alpha,
    Field const& l, Field const& r)
{
    int N = y.length();

    for (int i = 0; i < N; i++)
        y[i] = alpha * (l[i] - r[i]);
}

// computes y := alpha*x
// alpha is scalar
// y and x are vectors on length n
void hpc_scale(Field& y, const double alpha, Field& x)
{
    int N = y.length();

    for (int i = 0; i < N; i++)
        y[i] = alpha * x[i];
}

// computes linear combination of two vectors y := alpha*x + beta*z
// alpha and beta are scalar
// y, x and z are vectors on length n
void hpc_lcomb(Field& y, const double alpha, Field& x, const double beta,
    Field const& z)
{
    int N = y.length();

    for (int i = 0; i < N; i++)
        y[i] = alpha * x[i] + beta * z[i];
}

// copy one vector into another y := x
// x and y are vectors of length N
void hpc_copy(Field& y, Field const& x)
{
    int N = y.length();

    for (int i = 0; i < N; i++)
        y[i] = x[i];
}

// conjugate gradient solver
// solve the linear system A*x = b for x
// the matrix A is implicit in the objective function for the diffusion equation
// the value in x constitute the "first guess" at the solution
// x(N)
// ON ENTRY contains the initial guess for the solution
// ON EXIT  contains the solution
void hpc_cg(Field& x, Field const& b, const int maxiters, const double tol, bool& success)
{
    // this is the dimension of the linear system that we are to solve
    int nx = data::domain.nx;
    int ny = data::domain.ny;

    if(!cg_initialized) {
        cg_init(nx,ny);
    }

    // epslion value use for matrix-vector approximation
    double eps     = 1.e-4;
    double eps_inv = 1. / eps;

    // allocate memory for temporary storage
    hpc_fill(Fx,    0.0);
    hpc_fill(Fxold, 0.0);
    hpc_copy(xold, x);

    
    double time = -MPI_Wtime();

    diffusion(x, Fxold);
    time += MPI_Wtime();
    // v = x + epsilon*x
    hpc_scale(v, 1.0 + eps, x);

    // Fx = F(v)
    diffusion(v, Fx);


    

  //  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //  std::cout << rank << ">> " << time << std::endl;


    // r = b - A*x
    // where A*x = (Fx-Fxold)/eps
    hpc_add_scaled_diff(r, b, -eps_inv, Fx, Fxold);

    // p = r
    hpc_copy(p, r);

    // r_old_inner = <r,r>
    double r_old_inner = hpc_dot(r, r), r_new_inner = r_old_inner;

    // check for convergence
    success = false;
    if (sqrt(r_old_inner) < tol)
    {
        success = true;
        return;
    }

    int iter;
    for(iter=0; iter<maxiters; iter++) {
        // Ap = A*p
        hpc_lcomb(v, 1.0, xold, eps, p);
        diffusion(v, Fx);
        hpc_scaled_diff(Ap, eps_inv, Fx, Fxold);

        // alpha = r_old_inner / p'*Ap
        double alpha = r_old_inner / hpc_dot(p, Ap);

        // x += alpha*p
        hpc_axpy(x, alpha, p);

        // r -= alpha*Ap
        hpc_axpy(r, -alpha, Ap);

        // find new norm
        r_new_inner = hpc_dot(r, r);

        // test for convergence
        if (sqrt(r_new_inner) < tol) {
            success = true;
            break;
        }

        // p = r + r_new_inner.r_old_inner * p
        hpc_lcomb(p, 1.0, r, r_new_inner / r_old_inner, p);

        r_old_inner = r_new_inner;
    }
    stats::iters_cg += iter + 1;

    if (!success)
        std::cerr << "ERROR: CG failed to converge" << std::endl;
}

} // namespace linalg