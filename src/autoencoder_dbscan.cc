#include "autoencoder_dbscan.h"
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>
#include <pcl/kdtree/kdtree_flann.h>


Autoencoder::Autoencoder(int input_size, int hidden_size) : input_size(input_size), hidden_size(hidden_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 0.1);

    encoder_weights = Eigen::MatrixXd::NullaryExpr(hidden_size, input_size, [&](std::ptrdiff_t) { return d(gen); });
    decoder_weights = Eigen::MatrixXd::NullaryExpr(input_size, hidden_size, [&](std::ptrdiff_t) { return d(gen); });
    encoder_bias = Eigen::VectorXd::NullaryExpr(hidden_size, [&](std::ptrdiff_t) { return d(gen); });
    decoder_bias = Eigen::VectorXd::NullaryExpr(input_size, [&](std::ptrdiff_t) { return d(gen); });
}

Autoencoder::~Autoencoder() {}

Eigen::VectorXd Autoencoder::sigmoid(const Eigen::VectorXd& x) {
    return 1.0 / (1.0 + (-x).array().exp());
}

Eigen::VectorXd Autoencoder::sigmoid_derivative(const Eigen::VectorXd& x) {
    return x.array() * (1 - x.array());
}

Eigen::MatrixXd Autoencoder::sigmoids(const Eigen::MatrixXd& x) {
    return 1.0 / (1.0 + (-x).array().exp());
}

Eigen::MatrixXd Autoencoder::sigmoid_derivatives(const Eigen::MatrixXd& x) {
    return x.array() * (1 - x.array());
}

Eigen::VectorXd Autoencoder::relu(const Eigen::VectorXd& x) {
    return x.cwiseMax(0);
}
Eigen::VectorXd Autoencoder::relu_derivative(const Eigen::VectorXd& x) {
    Eigen::VectorXd result = x;
    for (int i = 0; i < result.size(); ++i) {
        result(i) = (result(i) > 0) ? 1 : 0;
    }
    return result;
}

Eigen::MatrixXd Autoencoder::relus(const Eigen::MatrixXd& x) {
    return x.cwiseMax(0);
}
Eigen::MatrixXd Autoencoder::relu_derivatives(const Eigen::MatrixXd& x) {
    return (x.array() > 0).cast<double>();
}

void Autoencoder::train(const std::vector<Eigen::VectorXd>& data, int epochs, double learning_rate, int batch_size) {
    int n = data.size();
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int start = 0; start < n; start += batch_size) {
            int end = std::min(start + batch_size, n);
            int current_batch_size = end - start;

            Eigen::MatrixXd batch_input(input_size, current_batch_size);
            for (int i = 0; i < current_batch_size; ++i) {
                batch_input.col(i) = data[start + i];
            }

            Eigen::MatrixXd hidden = relus(encoder_weights * batch_input + encoder_bias.replicate(1, current_batch_size));
            Eigen::MatrixXd output = relus(decoder_weights * hidden + decoder_bias.replicate(1, current_batch_size));

            Eigen::MatrixXd output_error = output - batch_input;
            Eigen::MatrixXd output_delta = output_error.array() * relu_derivatives(output).array();

            Eigen::MatrixXd hidden_error = decoder_weights.transpose() * output_delta;
            Eigen::MatrixXd hidden_delta = hidden_error.array() * relu_derivatives(hidden).array();

            decoder_weights -= learning_rate * output_delta * hidden.transpose() / current_batch_size;
            decoder_bias -= learning_rate * output_delta.rowwise().mean();
            encoder_weights -= learning_rate * hidden_delta * batch_input.transpose() / current_batch_size;
            encoder_bias -= learning_rate * hidden_delta.rowwise().mean();
        }
    }
}

void Autoencoder::train(const std::vector<Eigen::VectorXd>& data, int epochs, double learning_rate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (const auto& input : data) {
            Eigen::VectorXd hidden = sigmoid(encoder_weights * input + encoder_bias);
            Eigen::VectorXd output = sigmoid(decoder_weights * hidden + decoder_bias);

            Eigen::VectorXd output_error = output - input;
            Eigen::VectorXd output_delta = output_error.array() * sigmoid_derivative(output).array();

            Eigen::VectorXd hidden_error = decoder_weights.transpose() * output_delta;
            Eigen::VectorXd hidden_delta = hidden_error.array() * sigmoid_derivative(hidden).array();

            decoder_weights -= learning_rate * output_delta * hidden.transpose();
            decoder_bias -= learning_rate * output_delta;
            encoder_weights -= learning_rate * hidden_delta * input.transpose();
            encoder_bias -= learning_rate * hidden_delta;
        }
    }
}



Eigen::VectorXd Autoencoder::get_hidden_representation(const Eigen::VectorXd& input) {
    return sigmoid(encoder_weights * input + encoder_bias);
}

double Autoencoder::euclidean_distance(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    return (a - b).norm();
}

std::vector<int> Autoencoder::dbscan(const std::vector<Eigen::VectorXd>& data, double eps, int minPts) {
    int n = data.size();
    std::vector<int> labels(n, -1);
    int cluster_id = 0;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& point : data) {
        pcl::PointXYZ p;
        p.x = point[0];
        p.y = point[1];
        p.z = point.size() > 2 ? point[2] : 0.0;
        cloud->points.push_back(p);
    }
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    for (int i = 0; i < n; ++i) {
        if (labels[i] != -1) continue;

        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;

        if (kdtree.radiusSearch(cloud->points[i], eps, pointIdxRadiusSearch, pointRadiusSquaredDistance) < minPts) {
            labels[i] = 0; 
            continue;
        }

        cluster_id++;
        labels[i] = cluster_id;

        for (size_t k = 0; k < pointIdxRadiusSearch.size(); ++k) {
            int neighbor = pointIdxRadiusSearch[k];
            if (labels[neighbor] == 0) {
                labels[neighbor] = cluster_id;
            }
            if (labels[neighbor] != -1) continue;

            labels[neighbor] = cluster_id;

            std::vector<int> new_pointIdxRadiusSearch;
            std::vector<float> new_pointRadiusSquaredDistance;
            if (kdtree.radiusSearch(cloud->points[neighbor], eps, new_pointIdxRadiusSearch, new_pointRadiusSquaredDistance) >= minPts) {
                pointIdxRadiusSearch.insert(pointIdxRadiusSearch.end(), new_pointIdxRadiusSearch.begin(), new_pointIdxRadiusSearch.end());
            }
        }
    }

    return labels;
}    
std::vector<std::vector<cv::KeyPoint>> Autoencoder::getClusters(const std::vector<cv::KeyPoint>& keypoints, const std::vector<int>& labels) {
    std::unordered_map<int, std::vector<cv::KeyPoint>> cluster_map;
    for (size_t i = 0; i < keypoints.size(); ++i) {
        int label = labels[i];
        if (label > 0) {
            cluster_map[label].push_back(keypoints[i]);
        }
    }
    std::vector<std::vector<cv::KeyPoint>> clusters;
    for (const auto& pair : cluster_map) {
        clusters.push_back(pair.second);
    }
    return clusters;
}    


double Autoencoder::determine_eps(const std::vector<Eigen::VectorXd>& data, double quantile) {
    std::vector<double> distances;
    int n = data.size();
    if (n < 2) {
        return 0.0;
    }

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            distances.push_back(euclidean_distance(data[i], data[j]));
        }
    }
    std::sort(distances.begin(), distances.end());
    double eps = 0;
    int index = 0;
    while (eps == 0 && quantile <= 0.9) {
        index = static_cast<int>(distances.size() * quantile);
        if (index >= static_cast<int>(distances.size())) {
            index = distances.size() - 1;
        }
        eps = distances[index];
        quantile += 0.05;
    }

    return distances[index];
}

std::vector<Eigen::VectorXd> Autoencoder::keypointsToEigen(const std::vector<cv::KeyPoint>& keypoints) {
    std::vector<Eigen::VectorXd> data;
    for (const auto& kp : keypoints) {
        Eigen::VectorXd vec(6);
        vec << kp.pt.x, kp.pt.y, kp.size, kp.angle, kp.response, kp.octave;
        data.push_back(vec);
    }
    return data;
}
std::vector<cv::KeyPoint> Autoencoder::eigenToKeypoints(const std::vector<Eigen::VectorXd>& data) {
    std::vector<cv::KeyPoint> keypoints;
    for (const auto& vec : data) {
        cv::KeyPoint kp(vec(0), vec(1), vec(2), vec(3), vec(4), static_cast<int>(vec(5)));
        keypoints.push_back(kp);
    }
    return keypoints;
}