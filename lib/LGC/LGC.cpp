#pragma float_control(except, on)
#include <iostream>

#include "nanoflann.h"
#include "Eigen/Dense"

#include "LGC.hpp"

#define EPSIL 0.00001 // regularization parameter for LGC

void sharpening_for_dr(double *data, int obs, int dim, double alpha, unsigned int T, unsigned int k)
{
	typedef nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd> KDTree; // alias for NanoFLANN's KDTReeEigenMatrixAdaptor

	// Memory allocation
	Eigen::MatrixXd gradient_rho(1, dim); // Eigen matrix for storing the density gradient
	Eigen::MatrixXd ptCloud(obs, dim);	  // Eigen matrix for storing the point cloud (i.e. dataset)
	Eigen::MatrixXd shift(1, dim);		  // Eigen matrix for storing coordinate shifts
	Eigen::MatrixXd tmp_location(1, dim); // Eigen matrix for storing a temporary location

	double *location = new double[dim];						   // query point location
	double *out_dist_sqr_tmp = new double[k + 1];		   // vector storing k+1 squared distances (includes distance to self)
	Eigen::Index *ret_index_tmp = new Eigen::Index[k + 1]; // Eigen return index of nearest neighbours

	size_t nMatches = 0; // number of matches found during the nearest neighbour search

	// Assign and fill
	std::fill((Eigen::Index *)ret_index_tmp, (Eigen::Index *)(ret_index_tmp + k + 1), 0); // fill ret_index_tmp with zeroes
	std::fill((double *)out_dist_sqr_tmp, (double *)(out_dist_sqr_tmp + k + 1), 0);		  // fill out_dist_sqr_tmp with zeroes

	// Put data into point cloud
	int idx1 = 0, idx2 = 0;
	for (idx1 = 0; idx1 < obs; idx1++)
	{
		for (idx2 = 0; idx2 < dim; idx2++)
			ptCloud(idx1, idx2) = data[dim * idx1 + idx2];
	}

	/// FLMDS
	std::cout << "    LGC in progress:";
	unsigned int flag = 0;  // variable storing the # of the current iteration
	double h1 = 0; // variable storing nearest distance squared NN from a given point
	while (flag < T)
	{
		std::cout << "    " << flag << " ";

		// Build a KDTree index
		KDTree index(dim, ptCloud, 10); // maximum number of leaf points = 10 (this is also the default set in the KDTree constructor)
		index.index->buildIndex();

		for (idx1 = 0; idx1 < obs; idx1++) // for all observations
		{
			for (idx2 = 0; idx2 < dim; idx2++) // each observation is the query point
				location[idx2] = ptCloud(idx1, idx2);

			nMatches = index.index->knnSearch(&location[0], k + 1, &ret_index_tmp[0], &out_dist_sqr_tmp[0]);

			h1 = out_dist_sqr_tmp[nMatches - 1]; // h1 is square the distance to the kth nearest neighbor of current point
			if (h1 != 0.0) // run only when h1 is non-zero
			{
				Eigen::MatrixXd neighbors_of_pt(nMatches, dim); // Eigen matrix storing the nearest neighbours to the query point

				for (idx2 = 0; idx2 < nMatches; idx2++) // store nearest neighbour coords
					neighbors_of_pt.row(idx2) = ptCloud.row(ret_index_tmp[idx2]);

				for (idx2 = 0; idx2 < dim; idx2++) // save coords of query point to Eigen matrix
					tmp_location(0, idx2) = location[idx2];

				// Initialize density `gradient` and `shift` to zero
				for (idx2 = 0; idx2 < dim; idx2++)
				{
					gradient_rho(0, idx2) = 0;
					shift(0, idx2) = 0;
				}

				// compute total gradient
				for (idx2 = 0; idx2 < nMatches; idx2++) // for all the neighbors do summation
				{
					gradient_rho = gradient_rho - (((tmp_location - neighbors_of_pt.row(idx2)).array() * 2) / h1).matrix();
				}
				shift = gradient_rho / std::max(gradient_rho.norm(), EPSIL);
				ptCloud.row(idx1) = ptCloud.row(idx1) + (shift * alpha);
			}
			else
			{
				std::cout << "-";
			}
		}

		flag++;
	}

	std::cout << std::endl;

	// copy ptCloud into data
	for (idx1 = 0; idx1 < obs; idx1++)
	{
		for (idx2 = 0; idx2 < dim; idx2++)
			data[dim * idx1 + idx2] = ptCloud(idx1, idx2);
	}

	// free memory
	delete[] out_dist_sqr_tmp;
	delete[] ret_index_tmp;
	delete[] location;
}