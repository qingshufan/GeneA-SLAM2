#ifndef DEPTH_OFFSET_H
#define DEPTH_OFFSET_H

#include <vector>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <opencv2/opencv.hpp>

class DepthOffset {
public:
    DepthOffset();
    ~DepthOffset();
    std::vector<int> dbscan(const std::vector<Eigen::VectorXd>& data, double eps, int minPts);
    std::vector<int> binaryToGray(const std::vector<int>& binary);
    std::vector<int> grayToBinary(const std::vector<int>& gray);
    double fitness(const std::vector<int>& gray_chromosome, const std::vector<Eigen::VectorXd>& points, std::vector<Eigen::Vector2d>& selected_points);
    std::vector<std::vector<int>> selection(const std::vector<std::vector<int>>& population, const std::vector<double>& fitness_values);
    std::pair<std::vector<int>, std::vector<int>> crossover(const std::vector<int>& parent1, const std::vector<int>& parent2);
    void mutation(std::vector<int>& gray_chromosome, double mutation_rate);
    void adjust_parameters(double& crossover_rate, double& mutation_rate, double avg_fitness, double best_fitness);
    std::vector<std::vector<int>> elite_selection(const std::vector<std::vector<int>>& population, const std::vector<double>& fitness_values, int elite_size);
    int adjust_population_size(int current_size, double convergence_rate);
    std::vector<int> genetic_algorithm(const std::vector<Eigen::VectorXd>& points, int initial_population_size = 50, int generations = 50, int elite_size = 5);
    cv::Mat offsetMask(const cv::Mat& image, const cv::Mat& depth);
    cv::Mat mapMask(const cv::Mat& image, const cv::Mat& depth);
};

#endif    