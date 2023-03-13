#ifndef _DR_HPP_
#define _DR_HPP_

#include "tapkee/tapkee.hpp"

class DR {
    public:
        unsigned int TAPKEE_NUM_DR;
        unsigned int TAPKEE_NUM_NEIGHBORS;
        unsigned int TAPKEE_TARGET_DIMENSION;
        float TAPKEE_GAUSSIAN_KERNEL_WIDTH;
        unsigned int TAPKEE_MAX_ITERATION;
        float TAPKEE_LANDMARK_RATIO;
        float TAPKEE_SNE_PERPLEXITY;
        float TAPKEE_SNE_THETA;
        float TAPKEE_SQUISHING_RATE;

        DR();
        ~DR();

        /**
         * @brief Function that applies dimensionality reduction using the Tapkee API.
         * @param data Data array that is to be projected to a lower dimension.
         * @param obs Number of observations in dataset.
         * @param dim Number of data attributes (dimensions) after dimensionality reduction.
         */
        double* apply_DR(double* data, int obs, int dim, char* filepath);

        /**
         * @brief Sets the seed of the random number generator.
         * @param seed
         */
        void set_seed(unsigned int seed);

    protected:
        /**
         * @brief Function used to copy data from data array to Tapkee data matrix.
         * @param arr_src Source array.
         * @param arr_dst Destination array.
         * @param obs Number of observations in data array.
         * @param dim Number of data attributes (dimensions) in data array.
         */
        void copy_data(double *arr_src, tapkee::DenseMatrix &arr_dst, int &obs, int &dim);

        /**
         * @brief Function storing the reduced data to a file (rows/lines: observations, columns: dimensions/attributes).
         * @param out_arr Array that is to be written to a file.
         * @param file_str Filepath.
         * @param obs Number of observations in dataset.
         * @param aim_dim Number of data attributes (dimensions) after dimensionality reduction.
         */
        void write_file(tapkee::DenseMatrix &out_arr, const char* file_str, int obs, int aim_dim);
};

#endif