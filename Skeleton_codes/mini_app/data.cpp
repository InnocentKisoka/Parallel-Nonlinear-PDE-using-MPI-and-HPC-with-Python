#include <iostream>

#include <cmath>

#include <mpi.h>
#include <omp.h>

#include "data.h"

namespace data{

// fields that hold the solution
Field y_new;
Field y_old;

// fields that hold the boundary points
Field bndN;
Field bndE;
Field bndS;
Field bndW;

// buffers used during boundary halo communication
Field buffN;
Field buffE;
Field buffS;
Field buffW;

Discretization options;
SubDomain      domain;

void SubDomain::init(int mpi_rank, int mpi_size, Discretization& discretization)
{
	// here just following the comments
	//    // TODO: determine the number of sub-domains in the x and y dimensions
    //       using MPI_Dims_create
    int dims[2] = { 0, 0 };
    // determine the number of subdomains in the x and y dimensions using "MPI_Dims_create"  
	MPI_Dims_create(mpi_size, 2, dims);
 
    ndomy = dims[0];
    ndomx = dims[1];
// TODO: create a 2D non-periodic Cartesian topology using MPI_Cart_create
    int periods[2] = { 0, 0 };
    // create a 2D non-periodic cartesian topology using "MPI_Cart_create"
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &this->comm_cart);


    // TODO: retrieve coordinates of the rank in the topology using
    // MPI_Cart_coords

    int coords[2];
    // retrieve coordinates of the rank in the topology using "MPI_Cart_coords"
	MPI_Cart_coords(comm_cart, mpi_rank, 2, coords);
    domy = coords[0]+1;
    domx = coords[1]+1;

    // TODO: set neighbours for all directions using MPI_Cart_shift
    
	MPI_Cart_shift(comm_cart, 0, 1, &neighbour_south, &neighbour_north);
	MPI_Cart_shift(comm_cart, 1, 1, &neighbour_west,  &neighbour_east);

    // get bounding box
    nx = discretization.nx / ndomx;
    ny = discretization.ny / ndomy;
    startx = (domx-1)*nx+1;
    starty = (domy-1)*ny+1;

    // Adjusting for grid dimensions 
    
    if( domx == ndomx )
        nx = discretization.nx - startx + 1;
    if( domy == ndomy )
        ny = discretization.ny - starty + 1;

    endx = startx + nx -1;
    endy = starty + ny -1;

    // get total number of grid points in this sub-domain
    N = nx*ny;

    rank = mpi_rank;
    size = mpi_size;
}

// print domain decomposition information to stdout
void SubDomain::print() {
    for(int irank=0; irank<size; irank++) {
        if(irank == rank) {
            std::cout << "rank " << rank << "/" << size
                      << " : (" << domx << "," << domy << ")"
                      << " neigh N:S " << neighbour_north 
                      <<":"             << neighbour_south
                      << " neigh E:W " << neighbour_east 
                      << ":          " << neighbour_west
                      << " local dims " << nx << " x " << ny
                      << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double time = omp_get_wtime();
   
    while(omp_get_wtime()-time < 1e-1);
}

}