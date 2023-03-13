#include "DR.hpp"
#include <cstdlib>

extern "C"
{
    DR* createDR()
    {
        return new DR();
    }

    void destroyDR(DR* dr)
    {
        delete dr;
    }
  
    double* apply_DRDR(DR* dr, double* data, int obs, int dim, char* filepath)
    {
        return dr->apply_DR(data, obs, dim, filepath);
    }

    void set_seedDR(DR* dr, unsigned int seed)
    {
        dr->set_seed(seed);
    }

    unsigned int get_methodDR(DR* dr)
    {
        return dr->TAPKEE_NUM_DR;
    }

    void set_methodDR(DR* dr, unsigned int method)
    {
        dr->TAPKEE_NUM_DR = method;
    }

    unsigned int get_num_neighborsDR(DR* dr)
    {
        return dr->TAPKEE_NUM_NEIGHBORS;
    }

    void set_num_neighborsDR(DR* dr, unsigned int num_neighbors)
    {
        dr->TAPKEE_NUM_NEIGHBORS = num_neighbors;
    }

    unsigned int get_target_dimensionDR(DR* dr)
    {
        return dr->TAPKEE_TARGET_DIMENSION;
    }

    void set_target_dimensionDR(DR* dr, unsigned int target_dimension)
    {
        dr->TAPKEE_TARGET_DIMENSION = target_dimension;
    }

    float get_gaussian_kernel_widthDR(DR* dr)
    {
        return dr->TAPKEE_GAUSSIAN_KERNEL_WIDTH;
    }

    void set_gaussian_kernel_widthDR(DR* dr, float gaussian_kernel_width)
    {
        dr->TAPKEE_GAUSSIAN_KERNEL_WIDTH = gaussian_kernel_width;
    }

    int get_max_iterationDR(DR* dr)
    {
        return dr->TAPKEE_MAX_ITERATION;
    }

    void set_max_iterationDR(DR* dr, unsigned int max_iteration)
    {
        dr->TAPKEE_MAX_ITERATION = max_iteration;
    }

    float get_landmark_ratioDR(DR* dr)
    {
        return dr->TAPKEE_LANDMARK_RATIO;
    }

    void set_landmark_ratioDR(DR* dr, float landmark_ratio)
    {
        dr->TAPKEE_LANDMARK_RATIO = landmark_ratio;
    }

    float get_sne_perplexityDR(DR* dr)
    {
        return dr->TAPKEE_SNE_PERPLEXITY;
    }

    void set_sne_perplexityDR(DR* dr, float sne_perplexity)
    {
        dr->TAPKEE_SNE_PERPLEXITY = sne_perplexity;
    }

    float get_sne_thetaDR(DR* dr)
    {
        return dr->TAPKEE_SNE_THETA; 
    }

    void set_sne_thetaDR(DR* dr, float sne_theta)
    {
        dr->TAPKEE_SNE_THETA = sne_theta;
    }

    float get_squishing_rateDR(DR* dr)
    {
        return dr->TAPKEE_SQUISHING_RATE;
    }

    void set_squishing_rateDR(DR* dr, float squishing_rate)
    {
        dr->TAPKEE_SQUISHING_RATE = squishing_rate;
    }

    void free_memory(void *ptr)
    {
        free(ptr);
    }
};