#ifndef _LGC_HPP_
#define _LGC_HPP_

extern "C"
{
    /**
     * @brief Function that applies LGC (Local Gradient Clustering) on a given dataset.
     * @param data Dataset to be sharpened.
     * @param obs Number of observations in given dataset.
     * @param dim Number of data attributes (dimensions) in given dataset.
     * @param lambda1 Learning rate to be used by the LGC algorithm.
     */
    void sharpening_for_dr(double *data, int obs, int dim, double alpha, unsigned int T = 10, unsigned int k = 100);
};
#endif