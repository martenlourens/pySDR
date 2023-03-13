#pragma float_control(except, on)
#include <chrono>
#include <fstream>
#include <cstring>
#include <vector>
#include <map>
#include <cmath>
// #include <filesystem>
#include <random>
#include <string>
#include <sstream>

#include "main.hpp"
#include "Eigen/Dense"
#include "nanoflann.h"
#include "LGC.hpp"
#include "DR.hpp"

int main(int argc, char *argv[])
{
	std::string path;
	unsigned int mode = 0; // mode=1 (DR), mode=2 (SDR), mode=3 (DR & SDR)

	// if not enough args are given print usage information and exit with code 1
	if (argc != 6)
	{
		print_usage();
		std::exit(1);
	}

	// store configuration parameters
	size_t sz; // temporary variable storing number of characters processed
	mode = std::stoi(std::string(argv[1]), &sz);
	TAPKEE_NUM_DR = std::stoi(std::string(argv[2]), &sz);
	NUM_ITR = std::stoi(std::string(argv[3]), &sz);
	LEARNING_RATE = std::stod(std::string(argv[4]), &sz);

	// raise invalid mode error & print usage information
	if (mode == 0 || mode > 3)
	{
		std::cerr << "Invalid mode\n";
		print_usage();
		std::exit(1);
	}

	path = argv[5]; // store path to directory containing data directories

	// print configuration
	std::cout << "path: " << path << std::endl;
	std::cout << "mode: " << mode << std::endl;
	std::cout << "TAPKEE_NUM_DR: " << TAPKEE_NUM_DR << std::endl;
	std::cout << "NUM_ITR: " << NUM_ITR << std::endl;
	std::cout << "LEARNING_RATE: " << LEARNING_RATE << std::endl;

	// standard file names
	const char *file_str_header = "data__header.txt"; // input data header
	const char *file_str_original_data = "data_.txt"; // input data file
	const char *file_str_dr = "data__result_dr.txt";  // output data for dr
	const char *file_str_s_dr = "data__s_dr.txt";	  // output data for sdr
	const char *file_str_info = "info.txt";			  // output data or etc. info
	const char *file_str_lgc = "data__lgc.txt";		  // output data for lgc

	std::string fullpath;
	std::string cur_path;

	std::ofstream in_file;
	cur_path = path; // p_par.path().string().c_str(); //Read header
	std::cout << cur_path << std::endl;
	cur_path += "/";

	// store configuration in log file
	fullpath = cur_path + file_str_info;
	std::cout << fullpath << std::endl;
	in_file.open(fullpath);
	in_file << "Number of iterations: " << NUM_ITR << std::endl;
	in_file << "Learning rate: " << LEARNING_RATE << std::endl;
	in_file << "Tapkee DR used: " << TAPKEE_NUM_DR << std::endl;
	if (in_file.fail())
	{
		std::cerr << "    (ERROR) main(): Failed to open log file" << std::endl;
		return 1;
	}
	in_file << std::endl;

	// std::cout << "- DIR: " << path.c_str() << endl;//p_par.path().string().c_str() << endl;
	fullpath = cur_path + file_str_header;
	int obs = 0, dim = 0;						//{obs, dim} (number of observations and dimensions)
	double *data;							// pointer to input data
	read_obs_dim(&obs, &dim, fullpath.c_str()); // read header (file containing dimensionality of the data)
	if (obs > 0 && dim > 1)
		data = new double[obs * dim]; // heap memory allocation for dataset
	fullpath = cur_path + file_str_original_data;
	read_file(data, fullpath.c_str()); // read data and store it in allocated memory NB: C++ is row major

	double max1 = 0, min1 = 0;
	int idx1 = 0, idx2 = 0;
	// iterate over columns (dimensions)
	for (idx1 = 0; idx1 < dim; idx1++) // Normalize data between 0 and 1 (this is not what happens!!! (x-min)/(max-min) can be negative if any x is negative)
	{
		// find minimum and maximum for current data attribute
		max1 = data[idx1];
		min1 = data[idx1];
		for (idx2 = 1; idx2 < obs; idx2++)
		{
			max1 = std::max(data[idx2 * dim + idx1], max1);
			min1 = std::min(data[idx2 * dim + idx1], min1);
		}

		// normalize data for current data attribute
		for (idx2 = 0; idx2 < obs; idx2++)
			data[idx2 * dim + idx1] = ((data[idx2 * dim + idx1] - min1) / (max1 - min1));
	}

	std::chrono::duration<double, std::milli> elapsed;
	std::chrono::high_resolution_clock::time_point t;

	DR* reducer = new DR();
	reducer->set_seed(42);
	reducer->TAPKEE_NUM_DR = TAPKEE_NUM_DR;

	if (mode & DR_MODE)
	{
		std::cout << "=========================================\n";
		std::cout << "DR Mode\n";

		t = std::chrono::high_resolution_clock::now();

		fullpath = cur_path + file_str_dr;
		reducer->apply_DR(data, obs, dim, (char*)fullpath.c_str());

		elapsed = std::chrono::high_resolution_clock::now() - t;
		in_file << "Wall-clock time of vanilla DR: " << elapsed.count() << " ms" << std::endl;
		std::cout << "    "
					<< "DR: " << elapsed.count() << " ms" << std::endl;
	}

	if (mode & SDR_MODE)
	{
		std::cout << "=========================================\n";
		std::cout << "SDR Mode\n";

		// LGC step
		fullpath = cur_path + file_str_lgc;

		t = std::chrono::high_resolution_clock::now();

		// check whether the learning rate is positive
		if (LEARNING_RATE > 0)
			sharpening_for_dr(data, obs, dim, LEARNING_RATE);
		else
			std::cout << "    Check learning rate parameter for LGC." << std::endl;

		elapsed = std::chrono::high_resolution_clock::now() - t;
		in_file << "Wall-clock time of LGC step of SDR: " << elapsed.count() << " ms" << std::endl;
		std::cout << "    LGC step"
				  << ": " << elapsed.count() << " ms" << std::endl;

		write_file(data, fullpath.c_str(), obs, dim);

		// DR step
		t = std::chrono::high_resolution_clock::now();

		fullpath = cur_path + file_str_s_dr;
		reducer->apply_DR(data, obs, dim, (char*)fullpath.c_str());

		elapsed = std::chrono::high_resolution_clock::now() - t;
		in_file << "Wall-clock time of DR step of SDR: " << elapsed.count() << " ms" << std::endl;
		std::cout << "    DR step: " << elapsed.count() << " ms" << std::endl;
	}

	in_file.close();
	if (obs > 0 && dim > 1)
		delete[] data;

	delete reducer;

	return 0;
}

void read_obs_dim(int *obs, int *dim, const char *file_str)
{
	std::ifstream in_file;
	int idx = 0;
	in_file.open(file_str);
	if (in_file.fail())
	{
		std::cerr << "(ERROR) read_input_data(): failed to open file" << std::endl;
		return;
	}
	while (!in_file.eof())
		in_file >> *obs >> *dim;
	in_file.close();
}

void read_file(double *dst_arr, const char *file_str)
{
	std::ifstream in_file;
	int idx = 0;
	in_file.open(file_str);
	if (in_file.fail())
	{
		std::cerr << "(ERROR) read_input_data(): failed to open file" << std::endl;
		return;
	}
	while (!in_file.eof())
	{
		in_file >> dst_arr[idx]; // Copy each entry of matrix
		idx++;
	}
	in_file.close();
}

void write_file(float *out_arr, const char *file_str, int obs, int aim_dim)
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
			out_file << out_arr[aim_dim * idx1 + idx2] << ", ";
		out_file << "\n";
	}
	out_file.close();
}

void write_file(double *out_arr, const char *file_str, int obs, int aim_dim)
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
			out_file << out_arr[aim_dim * idx1 + idx2] << " ";
		out_file << "\n";
	}
	out_file.close();
}