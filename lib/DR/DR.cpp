#pragma float_control(except, on)
#include <iostream>
#include <fstream>
#include <random>

#include "DR.hpp"

DR::DR()
{
	// std::cout << "DR instance constructed." << std::endl;

    // Tapkee default parameters
    this->TAPKEE_NUM_DR = 19; // Tapkee default is PassThru
    this->TAPKEE_NUM_NEIGHBORS = 10; // Tapkee default is 5
    this->TAPKEE_TARGET_DIMENSION = 2; // Tapkee default is 2
	this->TAPKEE_GAUSSIAN_KERNEL_WIDTH = 1.0; // Tapkee default is 1.0
	this->TAPKEE_MAX_ITERATION = 100; // Tapkee default is 100
    this->TAPKEE_LANDMARK_RATIO = 0.1; // Tapkee default is 0.5
    this->TAPKEE_SNE_PERPLEXITY = 50.0; // Tapkee default is 30.0
	this->TAPKEE_SNE_THETA = 0.5; // Tapkee default is 0.5
	this->TAPKEE_SQUISHING_RATE = 0.99; // Tapkee default is 0.99
}

DR::~DR()
{
	// std::cout << "DR instance destructed." << std::endl;
}

double* DR::apply_DR(double* data, int obs, int dim, char* filepath)
{
    tapkee::DenseMatrix data_mat(obs, dim); // create Tapkee DenseMatrix instance
    this->copy_data(data, data_mat, obs, dim); // copy data to DenseMatrix

	// apply dimensionality reduction
    tapkee::TapkeeOutput output = tapkee::initialize()
									.withParameters((
													tapkee::method = static_cast<tapkee::DimensionReductionMethod>(this->TAPKEE_NUM_DR),
													tapkee::num_neighbors = static_cast<tapkee::IndexType>(this->TAPKEE_NUM_NEIGHBORS),
													tapkee::target_dimension = static_cast<tapkee::IndexType>(this->TAPKEE_TARGET_DIMENSION),
													tapkee::gaussian_kernel_width = static_cast<tapkee::ScalarType>(this->TAPKEE_GAUSSIAN_KERNEL_WIDTH),
													tapkee::max_iteration = static_cast<tapkee::IndexType>(this->TAPKEE_MAX_ITERATION),
													tapkee::landmark_ratio = static_cast<tapkee::ScalarType>(this->TAPKEE_LANDMARK_RATIO),
													tapkee::sne_perplexity = static_cast<tapkee::ScalarType>(this->TAPKEE_SNE_PERPLEXITY),
													tapkee::sne_theta = static_cast<tapkee::ScalarType>(this->TAPKEE_SNE_THETA),
													tapkee::squishing_rate = static_cast<tapkee::ScalarType>(this->TAPKEE_SQUISHING_RATE)
													))
									.embedUsing(data_mat.transpose());

	// write output to file
	this->write_file(output.embedding, filepath, obs, this->TAPKEE_TARGET_DIMENSION);

	// copy data to c array
	// NB: simply returning output.embedding.data() is not appropriate since Eigen::Matrix stores data in column major order by default
	double *data_out;
	data_out = new double[obs * this->TAPKEE_TARGET_DIMENSION];
	int idx1 = 0, idx2 = 0;
	for (idx1 = 0; idx1 < obs; idx1++)
	{
		for (idx2 = 0; idx2 < this->TAPKEE_TARGET_DIMENSION; idx2++)
			data_out[this->TAPKEE_TARGET_DIMENSION * idx1 + idx2] = output.embedding(idx1, idx2);
	}

	return data_out;
}

void DR::set_seed(unsigned int seed)
{
	std::srand(seed);
}

void DR::copy_data(double *arr_src, tapkee::DenseMatrix &arr_dst, int &obs, int &dim)
{
	int idx1 = 0, idx2 = 0;
	for (idx1 = 0; idx1 < obs; idx1++)
	{
		for (idx2 = 0; idx2 < dim; idx2++)
			arr_dst(idx1, idx2) = (double)arr_src[idx1 * dim + idx2];
	}
}

void DR::write_file(tapkee::DenseMatrix &out_arr, const char *file_str, int obs, int aim_dim)
{
	std::ofstream out_file;
	int idx1 = 0;
	int idx2 = 0;
	out_file.open(file_str);
	if (out_file.fail())
	{
		std::cerr << "(ERROR) write_input_data(): failed to open file" << std::endl;
		return;
	}
	for (int idx1 = 0; idx1 < obs; idx1++)
	{
		for (int idx2 = 0; idx2 < aim_dim; idx2++)
			out_file << out_arr(idx1, idx2) << " ";
		out_file << "\n";
	}
	out_file.close();
}