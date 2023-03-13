#ifndef _MAIN_HPP_
#define _MAIN_HPP_

#include <iostream>
#include "tapkee/tapkee.hpp"

typedef unsigned long long uint64;
typedef unsigned long uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;

// Tapkee default parameters
unsigned int TAPKEE_NUM_DR; // no. of DR defined in tapkee (LandmarkMultidimensionalScaling 11, RandomProjection 15, tDistributedStochasticNeighborEmbedding 17)

// SDR parameters
unsigned int NUM_ITR = 0;
double LEARNING_RATE = 0.1;

// Run modes
#define DR_MODE 1
#define SDR_MODE 2

/**
 *	@brief Function printing usage information for the sdr command.
 */
void print_usage()
{
	std::cerr << "Usage: sdr <mode> <TAPKEE_NUM_DR> <NUM_ITR> <LEARNING_RATE> <path>\n";
	std::cerr << "mode=1 --> DR, mode=2 --> SDR, mode=3 --> DR & SDR\n";
	std::cerr << "TAPKEE_NUM_DR:\n";
	std::cerr << "\tLandmarkMultidimensionalScaling 11\n";
	std::cerr << "\tMultidimensionalScaling 10\n";
	std::cerr << "\tPCA 14\n";
	std::cerr << "\tRandomProjection 15\n";
	std::cerr << "\ttDistributedStochasticNeighborEmbedding 17\n";
	std::cerr << "NUM_ITR: # of iterations (typically from 0 to 10)\n";
	std::cerr << "LEARNING_RATE: like in GD (typically a small float, like 0.1)\n";
	std::cerr << "path: base directory containing data directories\n";
}

/**
 * @brief Function reading header file of given dataset.
 * @param obs Number of observations.
 * @param dim Number of data dimensions.
 * @param file_str Path to the header file.
 */
void read_obs_dim(int *obs, int *dim, const char *file_str); // later: throw / catch --> readfile fail process instead of exit() or return

/**
 * @brief Function reading data from file and storing it in allocated memory (rows/lines: observations, columns: dimensions/attributes).
 * @param dst_arr Destination array in which data should be stored.
 * @param file_str Path to data file.
 */
void read_file(double *dst_arr, const char *file_str);

/**
 * @brief Function storing the reduced data to a file (rows/lines: observations, columns: dimensions/attributes).
 * @param out_arr Array that is to be written to a file.
 * @param file_str Filepath.
 * @param obs Number of observations in dataset.
 * @param aim_dim Number of data attributes (dimensions) after dimensionality reduction.
 */
void write_file(float *out_arr, const char *file_str, int obs, int aim_dim);

/**
 * @brief Function storing the reduced data to a file (rows/lines: observations, columns: dimensions/attributes).
 * @param out_arr Array that is to be written to a file.
 * @param file_str Filepath.
 * @param obs Number of observations in dataset.
 * @param aim_dim Number of data attributes (dimensions) after dimensionality reduction.
 */
void write_file(double *out_arr, const char *file_str, int obs, int aim_dim);

#endif