#ifndef AUTOENCODER_DBSCAN_H
#define AUTOENCODER_DBSCAN_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>

class Autoencoder {
public:
    Autoencoder(int input_size, int hidden_size);
    ~Autoencoder();
    void train(const std::vector<Eigen::VectorXd>& data, int epochs, double learning_rate);
    void train(const std::vector<Eigen::VectorXd>& data, int epochs, double learning_rate, int batch_size);
    Eigen::VectorXd get_hidden_representation(const Eigen::VectorXd& input);
    double euclidean_distance(const Eigen::VectorXd& a, const Eigen::VectorXd& b);
    std::vector<int> dbscan(const std::vector<Eigen::VectorXd>& data, double eps, int minPts);
    std::vector<std::vector<cv::KeyPoint>> getClusters(const std::vector<cv::KeyPoint>& keypoints, const std::vector<int>& labels);
    double determine_eps(const std::vector<Eigen::VectorXd>& data, double quantile);
    std::vector<Eigen::VectorXd> keypointsToEigen(const std::vector<cv::KeyPoint>& keypoints);
    std::vector<cv::KeyPoint> eigenToKeypoints(const std::vector<Eigen::VectorXd>& data);

private:
    Eigen::MatrixXd encoder_weights;
    Eigen::MatrixXd decoder_weights;
    Eigen::VectorXd encoder_bias;
    Eigen::VectorXd decoder_bias;
    int input_size;
    int hidden_size;

    Eigen::VectorXd sigmoid(const Eigen::VectorXd& x);
    Eigen::VectorXd sigmoid_derivative(const Eigen::VectorXd& x);

    Eigen::MatrixXd sigmoids(const Eigen::MatrixXd& x);
    Eigen::MatrixXd sigmoid_derivatives(const Eigen::MatrixXd& x);

    Eigen::VectorXd relu(const Eigen::VectorXd& x);
    Eigen::VectorXd relu_derivative(const Eigen::VectorXd& x);

    Eigen::MatrixXd relus(const Eigen::MatrixXd& x);
    Eigen::MatrixXd relu_derivatives(const Eigen::MatrixXd& x);
};

#endif // AUTOENCODER_DBSCAN_H