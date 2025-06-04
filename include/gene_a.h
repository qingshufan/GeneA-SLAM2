#ifndef GENE_A_H
#define GENE_A_H

#include "autoencoder_dbscan.h"


using Chromosome = std::vector<bool>;
using Population = std::vector<Chromosome>;

class Autoencoder;

class GeneA {
public:
    GeneA();
    void testGA(const std::vector<cv::KeyPoint>& keypoints, std::vector<cv::KeyPoint>& selected_keypoints, 
                std::vector<cv::KeyPoint>& red_keypoints, std::vector<bool>& best_x);
    std::vector<std::vector<cv::KeyPoint>> ae_dbscan(const std::vector<cv::KeyPoint>& keypoints, 
                                                     std::vector<cv::KeyPoint>& all_selected_keypoints);
    void runGeneA(const std::vector<cv::KeyPoint>& keypoints, 
        std::vector<cv::KeyPoint>& all_selected_keypoints, std::vector<cv::KeyPoint>& all_red_keypoints);

private:
    double fitness(const Chromosome& chromosome, const std::vector<cv::KeyPoint>& keypoints, double sum_x, double sum_y);
    Chromosome selection(const Population& population, const std::vector<cv::KeyPoint>& keypoints, double sum_x, double sum_y);
    Chromosome crossover(const Chromosome& parent1, const Chromosome& parent2);
    void mutation(Chromosome& chromosome);
};

#endif // GENE_A_H