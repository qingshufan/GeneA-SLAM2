#include "depth_offset.h"

#include <numeric>
#include <random>
#include <algorithm>
#include <limits>

DepthOffset::DepthOffset() {}

DepthOffset::~DepthOffset() {}

std::vector<int> DepthOffset::dbscan(const std::vector<Eigen::VectorXd>& data, double eps, int minPts) {
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

std::vector<int> DepthOffset::binaryToGray(const std::vector<int>& binary) {
    std::vector<int> gray(binary.size());
    gray[0] = binary[0];
    for (size_t i = 1; i < binary.size(); ++i) {
        gray[i] = binary[i - 1] ^ binary[i];
    }
    return gray;
}

std::vector<int> DepthOffset::grayToBinary(const std::vector<int>& gray) {
    std::vector<int> binary(gray.size());
    binary[0] = gray[0];
    for (size_t i = 1; i < gray.size(); ++i) {
        binary[i] = binary[i - 1] ^ gray[i];
    }
    return binary;
}

double DepthOffset::fitness(const std::vector<int>& gray_chromosome, const std::vector<Eigen::VectorXd>& points, std::vector<Eigen::Vector2d>& selected_points) {
    std::vector<int> chromosome = grayToBinary(gray_chromosome);
    selected_points.clear();
    double sum_depth = 0.0;
    double variance = 0.0;
    int n_depth = 0;
    int min_x = std::numeric_limits<int>::max();
    int max_x = std::numeric_limits<int>::min();
    int min_y = std::numeric_limits<int>::max();
    int max_y = std::numeric_limits<int>::min();

    for (size_t i = 0; i < chromosome.size(); ++i) {
        if (chromosome[i] == 1) {
            selected_points.emplace_back(points[i](0), points[i](1));
            double depth = points[i](2);
            int x = static_cast<int>(points[i](0));
            int y = static_cast<int>(points[i](1));
            min_x = std::min(min_x, x);
            max_x = std::max(max_x, x);
            min_y = std::min(min_y, y);
            max_y = std::max(max_y, y);

            n_depth++;
            double delta = depth - sum_depth;
            sum_depth += delta / n_depth;
            variance += delta * (depth - sum_depth);
        }
    }

    if (n_depth == 0) {
        return std::numeric_limits<double>::max();
    }

    variance /= n_depth;

    int area = (max_x - min_x) * (max_y - min_y);

    double smoothness = 0.0;
    if (selected_points.size() > 1) {
        for (size_t i = 0; i < selected_points.size() - 1; ++i) {
            double dx = selected_points[i + 1](0) - selected_points[i](0);
            double dy = selected_points[i + 1](1) - selected_points[i](1);
            smoothness += std::sqrt(dx * dx + dy * dy);
        }
        smoothness /= (selected_points.size() - 1);
    }

    double weight_variance = 0.5;
    double weight_area = 0.3;
    double weight_smoothness = 0.2;

    return weight_variance * variance + weight_area * area + weight_smoothness * smoothness;
}

std::vector<std::vector<int>> DepthOffset::selection(const std::vector<std::vector<int>>& population, const std::vector<double>& fitness_values) {
    std::vector<std::vector<int>> new_population;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    double total_fitness = 0.0;
    for (double fit : fitness_values) {
        total_fitness += fit;
    }

    for (size_t i = 0; i < population.size(); ++i) {
        double r = dis(gen) * total_fitness;
        double sum = 0.0;
        for (size_t j = 0; j < population.size(); ++j) {
            sum += fitness_values[j];
            if (sum >= r) {
                new_population.push_back(population[j]);
                break;
            }
        }
    }

    return new_population;
}

std::pair<std::vector<int>, std::vector<int>> DepthOffset::crossover(const std::vector<int>& parent1, const std::vector<int>& parent2) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, parent1.size() - 1);
    int crossover_point = dis(gen);

    std::vector<int> child1(parent1.begin(), parent1.begin() + crossover_point);
    child1.insert(child1.end(), parent2.begin() + crossover_point, parent2.end());

    std::vector<int> child2(parent2.begin(), parent2.begin() + crossover_point);
    child2.insert(child2.end(), parent1.begin() + crossover_point, parent1.end());

    return std::make_pair(child1, child2);
}

void DepthOffset::mutation(std::vector<int>& gray_chromosome, double mutation_rate) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int& gene : gray_chromosome) {
        if (dis(gen) < mutation_rate) {
            gene = 1 - gene;
        }
    }
}

void DepthOffset::adjust_parameters(double& crossover_rate, double& mutation_rate, double avg_fitness, double best_fitness) {
    if (avg_fitness - best_fitness < 0.1) {
        crossover_rate = 0.8;
        mutation_rate = 0.05;
    } else {
        crossover_rate = 0.6;
        mutation_rate = 0.01;
    }
}

std::vector<std::vector<int>> DepthOffset::elite_selection(const std::vector<std::vector<int>>& population, const std::vector<double>& fitness_values, int elite_size) {
    std::vector<std::pair<double, int>> fitness_index;
    for (size_t i = 0; i < fitness_values.size(); ++i) {
        fitness_index.push_back(std::make_pair(fitness_values[i], static_cast<int>(i)));
    }
    std::sort(fitness_index.begin(), fitness_index.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
        return a.first > b.first;
    });

    std::vector<std::vector<int>> elite_population;
    for (int i = 0; i < elite_size; ++i) {
        elite_population.push_back(population[fitness_index[i].second]);
    }
    return elite_population;
}

int DepthOffset::adjust_population_size(int current_size, double convergence_rate) {
    if (convergence_rate < 0.01) {
        return std::min(current_size + 10, 50); 
    } else if (convergence_rate > 0.1) {
        return std::max(current_size - 10, 10); 
    }
    return current_size;
}

std::vector<int> DepthOffset::genetic_algorithm(const std::vector<Eigen::VectorXd>& points, int initial_population_size, int generations, int elite_size) {
    std::vector<std::vector<int>> population;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    for (int i = 0; i < initial_population_size; ++i) {
        std::vector<int> binary_chromosome(points.size());
        for (int& gene : binary_chromosome) {
            gene = dis(gen);
        }
        std::vector<int> gray_chromosome = binaryToGray(binary_chromosome);
        population.push_back(gray_chromosome);
    }

    double crossover_rate = 0.6;
    double mutation_rate = 0.01;
    int population_size = initial_population_size;
    std::vector<Eigen::Vector2d> selected_points;

    for (int gen = 0; gen < generations; ++gen) {
        std::vector<double> fitness_values;
        double total_fitness = 0.0;
        double best_fitness = std::numeric_limits<double>::max();
        double worst_fitness = std::numeric_limits<double>::min();

        for (const auto& gray_chromosome : population) {
            double fit = fitness(gray_chromosome, points, selected_points);
            fitness_values.push_back(1.0 / fit); 
            total_fitness += fit;
            best_fitness = std::min(best_fitness, fit);
            worst_fitness = std::max(worst_fitness, fit);
        }

        double avg_fitness = total_fitness / population_size;
        double convergence_rate = (worst_fitness - best_fitness) / worst_fitness;

        adjust_parameters(crossover_rate, mutation_rate, avg_fitness, best_fitness);
        population_size = adjust_population_size(population_size, convergence_rate);

        std::vector<std::vector<int>> elite_population = elite_selection(population, fitness_values, elite_size);

        std::vector<std::vector<int>> new_population = selection(population, fitness_values);

        while (new_population.size() + elite_size > population_size) {
            new_population.pop_back();
        }

        new_population.insert(new_population.end(), elite_population.begin(), elite_population.end());

        for (size_t i = 0; i < new_population.size(); i += 2) {
            if (i + 1 < new_population.size()) {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> dis(0.0, 1.0);
                if (dis(gen) < crossover_rate) {
                    std::pair<std::vector<int>, std::vector<int>> children = crossover(new_population[i], new_population[i + 1]);
                    new_population[i] = children.first;
                    new_population[i + 1] = children.second;
                }
            }
            mutation(new_population[i], mutation_rate);
        }

        population = new_population;
    }

    double best_fitness = std::numeric_limits<double>::max();
    std::vector<int> best_gray_chromosome;
    for (const auto& gray_chromosome : population) {
        double fit = fitness(gray_chromosome, points, selected_points);
        if (fit < best_fitness) {
            best_fitness = fit;
            best_gray_chromosome = gray_chromosome;
        }
    }

    std::vector<int> best_chromosome = grayToBinary(best_gray_chromosome);

    return best_chromosome;
}

static float quickSelect(std::vector<float>& arr, int k) {
    if (arr.size() == 1) {
        return arr[0];
    }

    float pivot = arr[arr.size() / 2];

    std::vector<float> left, middle, right;

    for (float x : arr) {
        if (x < pivot) {
            left.push_back(x);
        } else if (x == pivot) {
            middle.push_back(x);
        } else {
            right.push_back(x);
        }
    }

    if (k < left.size()) {
        return quickSelect(left, k);
    } else if (k < left.size() + middle.size()) {
        return middle[0];
    } else {
        return quickSelect(right, k - left.size() - middle.size());
    }
}

static float calculateMedian(std::vector<float>& depthValues) {
    size_t size = depthValues.size();
    if (size % 2 == 0) {
        return (quickSelect(depthValues, size / 2 - 1) + quickSelect(depthValues, size / 2)) / 2.0;
    } else {
        return quickSelect(depthValues, size / 2);
    }
}

//**** the modified code from NGD-SLAM ****
cv::Mat CreateMaskFromClusters(const std::map<int, std::vector<cv::Point3f>>& clusters, const cv::Mat& mImDepth2)
{
    cv::Mat hullMask = cv::Mat::zeros(mImDepth2.rows, mImDepth2.cols, CV_8UC1);
    cv::Mat mask = cv::Mat::zeros(mImDepth2.rows, mImDepth2.cols, CV_8UC1);

    for (const auto& cluster : clusters)
    {
        cv::Mat localMask = cv::Mat::zeros(mImDepth2.rows, mImDepth2.cols, CV_8UC1);
        std::vector<float> pointsDepth;

        if (cluster.second.size() > 1)
        {
            float minX = std::numeric_limits<float>::max(), minY = std::numeric_limits<float>::max();
            float maxX = std::numeric_limits<float>::lowest(), maxY = std::numeric_limits<float>::lowest();

            for (const cv::Point3f& pt : cluster.second)
            {
                minX = std::min(minX, pt.x);
                maxX = std::max(maxX, pt.x);
                minY = std::min(minY, pt.y);
                maxY = std::max(maxY, pt.y);

                float depth = mImDepth2.at<float>(pt.y, pt.x);
                if(depth >= 0.05) pointsDepth.push_back(depth);
            }

            cv::Point topLeft(static_cast<int>(minX), static_cast<int>(minY));
            cv::Point bottomRight(static_cast<int>(maxX), static_cast<int>(maxY));

            float width = bottomRight.x - topLeft.x;
            float height = bottomRight.y - topLeft.y;
            if(width < 50 || height < 50) continue;

            cv::Size maskSize = localMask.size();
            int increaseWidth = 15;
            int increaseHeight = 15;
            cv::Point adjustedTopLeft = cv::Point(
                std::max(0, topLeft.x - increaseWidth),
                std::max(0, topLeft.y - increaseHeight));
            cv::Point adjustedBottomRight = cv::Point(
                std::min(maskSize.width, bottomRight.x + increaseWidth),
                std::min(maskSize.height, bottomRight.y + increaseHeight));

            cv::rectangle(localMask, adjustedTopLeft, adjustedBottomRight, cv::Scalar(1), cv::FILLED);
            cv::rectangle(hullMask, adjustedTopLeft, adjustedBottomRight, cv::Scalar(1), cv::FILLED);
        }
        else continue;

        std::sort(pointsDepth.begin(), pointsDepth.end());
        float value = pointsDepth[pointsDepth.size() / 2];

        for (int y = 0; y < localMask.rows; y++)
        {
            for (int x = 0; x < localMask.cols; x++)
            {
                int val = (int)localMask.at<uchar>(y, x);
                if (val == 1 && std::abs(mImDepth2.at<float>(y, x) - value) > 0.3)
                {
                    localMask.at<uchar>(y, x) = 0;
                }
            }
        }

        cv::Mat labels, stats, centroids;
        int nLabels = cv::connectedComponentsWithStats(localMask, labels, stats, centroids, 4, CV_32S);

        int largestLabel = 0;
        int largestArea = 0;
        for (int label = 1; label < nLabels; ++label)
        {
            int area = stats.at<int>(label, cv::CC_STAT_AREA);
            if (area > largestArea)
            {
                largestArea = area;
                largestLabel = label;
            }
        }

        for (int y = 0; y < labels.rows; ++y)
        {
            for (int x = 0; x < labels.cols; ++x)
            {
                if (labels.at<int>(y, x) == largestLabel) mask.at<uchar>(y, x) = 1;
            }
        }
    }

    // int dilationSize = 3;
    // cv::Mat dilationElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * dilationSize + 1, 2 * dilationSize + 1), cv::Point(dilationSize, dilationSize));
    // cv::dilate(mask, mask, dilationElement);

    return mask;
}
//**** modified code from NGD-SLAM ****
static int mask_index1=0;

cv::Mat DepthOffset::mapMask(const cv::Mat& image, const cv::Mat& depth) {
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8U);
    if (depth.empty()) {
        return mask;
    }
    int rows = depth.rows;
    int cols = depth.cols;
    float t4 = 8e-6; //8e-6 broader 2e-5 smaller 
    float t5 = 1e-6;//9e-13 useless
    // float t4 = 5e-5;
    // float t5 = 5e-6;
    
    float minDepth = std::numeric_limits<float>::max();
    float maxDepth = std::numeric_limits<float>::lowest();
    for (int v = 1; v < rows - 1; v += 3) {
        for (int u = 1; u < cols - 1; u += 3) {
            float sum = 0;
            int count = 0;
            for (int i = v - 1; i <= v + 1; ++i) {
                for (int j = u - 1; j <= u + 1; ++j) {
                    sum += depth.at<float>(i, j);
                    count++;
                }
            }
            float mean = sum / count;
            float variance = 0;
            for (int i = v - 1; i <= v + 1; ++i) {
                for (int j = u - 1; j <= u + 1; ++j) {
                    variance += std::pow(depth.at<float>(i, j) - mean, 2);
                }
            }
            variance /= count;
            if (variance > 0 && variance < t4 && variance > t5) {
                for (int i = v - 1; i <= v + 1; ++i) {
                    for (int j = u - 1; j <= u + 1; ++j) {
                        mask.at<uchar>(i, j) = 1;
                        float currentDepth = depth.at<float>(i, j);
                        if(currentDepth<=0) continue;
                        if (currentDepth < minDepth) {
                            minDepth = currentDepth;
                        }
                        if (currentDepth > maxDepth) {
                            maxDepth = currentDepth;
                        }
                    }
                }
            }
        }
    }
    for (int v = 1; v < rows - 1; v += 3) {
        for (int u = 1; u < cols - 1; u += 3) {
                for (int i = v - 1; i <= v + 1; ++i) {
                    for (int j = u - 1; j <= u + 1; ++j) {
                        float currentDepth = depth.at<float>(i, j);
                        if(currentDepth<=0) continue;
                        if (currentDepth >= minDepth && currentDepth <= maxDepth) {
                            mask.at<uchar>(i, j) = 1;
                        }
                    }
                }
        }
    }
    cv::Mat& merged_mask = mask;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(merged_mask,merged_mask, kernel);
    return merged_mask;
}

static cv::Mat depthToColor(const cv::Mat& depthImage) {
    cv::Mat colorImage(depthImage.size(), CV_8UC3);
    double minVal, maxVal;
    cv::minMaxLoc(depthImage, &minVal, &maxVal);

    for (int y = 0; y < depthImage.rows; ++y) {
        for (int x = 0; x < depthImage.cols; ++x) {
            float depth = depthImage.at<float>(y, x);
            float normalizedDepth = (depth - minVal) / (maxVal - minVal);
            cv::Vec3b color;

            if (normalizedDepth < 0.25) {
                color[0] = 255;
                color[1] = static_cast<uchar>(normalizedDepth * 4 * 255);
                color[2] = 0;
            } else if (normalizedDepth < 0.5) {
                color[0] = static_cast<uchar>((0.5 - normalizedDepth) * 4 * 255);
                color[1] = 255;
                color[2] = 0;
            } else if (normalizedDepth < 0.75) {
                color[0] = 0;
                color[1] = 255;
                color[2] = static_cast<uchar>((normalizedDepth - 0.5) * 4 * 255);
            } else {
                color[0] = 0;
                color[1] = static_cast<uchar>((1 - normalizedDepth) * 4 * 255);
                color[2] = 255;
            }
            colorImage.at<cv::Vec3b>(y, x) = color;
        }
    }
    return colorImage;
}

static void drawStar(cv::Mat& image, cv::Point center, int radius, cv::Scalar color, int thickness = 1) {
    const double PI = 3.14159265358979323846;
    const int numPoints = 5;
    cv::Point points[1][10];
    for (int i = 0; i < numPoints * 2; ++i) {
        double angle = 2 * PI * i / (numPoints * 2) - PI / 2;
        int r = (i % 2 == 0) ? radius : radius / 2;
        points[0][i] = cv::Point(center.x + r * cos(angle), center.y + r * sin(angle));
    }
    const cv::Point* ppt[1] = { points[0] };
    int npt[] = { 10 };
    cv::polylines(image, ppt, npt, 1, true, color, thickness);
}
static int mask_index=0;
cv::Mat DepthOffset::offsetMask(const cv::Mat& image, const cv::Mat& depth) {
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8U);
    if (depth.empty()) {
        return mask;
    }
    int rows = depth.rows;
    int cols = depth.cols;
    float t4 = 5e-5;
    float t5 = 5e-6;

    // float t4 = 1e-4;
    // float t5 = 1e-6;
    float minDepth = std::numeric_limits<float>::max();
    float maxDepth = std::numeric_limits<float>::lowest();

    std::vector<Eigen::VectorXd> data;
    data.reserve(rows * cols + 1);
    for (int v = 1; v < rows - 1; v += 3) {
        for (int u = 1; u < cols - 1; u += 3) {
            float sum = 0;
            int count = 0;
            for (int i = v - 1; i <= v + 1; ++i) {
                for (int j = u - 1; j <= u + 1; ++j) {
                    sum += depth.at<float>(i, j);
                    count++;
                }
            }
            float mean = sum / count;
            float variance = 0;
            for (int i = v - 1; i <= v + 1; ++i) {
                for (int j = u - 1; j <= u + 1; ++j) {
                    variance += std::pow(depth.at<float>(i, j) - mean, 2);
                }
            }
            variance /= count;
            if (variance > 0 && variance < t4 && variance > t5) {
                for (int i = v - 1; i <= v + 1; ++i) {
                    for (int j = u - 1; j <= u + 1; ++j) {
                        float currentDepth = depth.at<float>(i, j);
                        if (currentDepth > 0) {
                            mask.at<uchar>(i, j) = 1;
                            break;
                        }
                    }
                }
            }
        }
    }
    int t6 = 1;
    int gridSize = 2;
    for (int gridY = 0; gridY < rows; gridY += gridSize) {
        for (int gridX = 0; gridX < cols; gridX += gridSize) {
            int startV = gridY;
            int startU = gridX;
            int endV = std::min(gridY + gridSize, rows);
            int endU = std::min(gridX + gridSize, cols);

            bool foundPoint = false;
            for (int v = startV; v < endV; v += t6) {
                for (int u = startU; u < endU; u += t6) {
                    if (mask.at<uchar>(v, u) == 0) {
                        continue;
                    }
                    float currentDepth = depth.at<float>(v, u);
                    if (currentDepth > 0) {
                        Eigen::VectorXd vec(3);
                        vec << v, u, currentDepth;
                        data.push_back(vec);
                        foundPoint = true;
                        break;
                    }
                }
                if (foundPoint) {
                    break;
                }
            }
        }
    }

    mask.setTo(0);
    float eps = 5; 
    int minPts = 4; 
    std::vector<int> labels = dbscan(data, eps, minPts);
    std::map<int, std::vector<Eigen::VectorXd>> cluster_map;
    for (size_t i = 0; i < data.size(); ++i) {
        int label = labels[i];
        if (label > 0) {
            cluster_map[label].push_back(data[i]);
        }
    }
    std::vector<std::pair<int, std::vector<Eigen::VectorXd>>> sorted_clusters(cluster_map.begin(), cluster_map.end());
    std::sort(sorted_clusters.begin(), sorted_clusters.end(), [](const std::pair<int, std::vector<Eigen::VectorXd>>& a, const std::pair<int, std::vector<Eigen::VectorXd>>& b) {
        return a.second.size() > b.second.size();
    });
    std::map<int, std::vector<cv::Point3f>> clusters_for_mask;
    cv::Mat cmask = cv::Mat::zeros(image.size(), CV_8U);
    for (int i = 0; i < std::min(2, static_cast<int>(sorted_clusters.size())); i++) {
        int cluster_id = sorted_clusters[i].first;
        const std::vector<Eigen::VectorXd>& cluster_points = sorted_clusters[i].second;
        std::vector<cv::Point3f> points;
        for (const auto& point : cluster_points) {
            cmask.at<uchar>(point[0], point[1])=1;
            points.emplace_back(point[1], point[0], point[2]);
        }
        clusters_for_mask[cluster_id] = points;
    }
    // cv::Mat coloredImage = depthToColor(depth);
    // for (int y = 0; y < image.rows; ++y) {
    //     for (int x = 0; x < image.cols; ++x) {
    //         ushort depthValue = depth.at<ushort>(y, x);
    //         if (cmask.at<uchar>(y, x) == 1) {
    //             drawStar(coloredImage, cv::Point(x, y), 2, cv::Scalar(0, 0, 125), 2);
    //         } 
    //     }
    // }
    // std::stringstream stt;
    // stt << "feature/mask" << mask_index << ".png";
    // cv::imwrite(stt.str(), coloredImage);
    // for (int i = 0; i < std::min(2, static_cast<int>(sorted_clusters.size())); i++) {
    //     auto& ts = sorted_clusters[i].second;
    //     std::vector<float> depthValues;
    //     depthValues.reserve(ts.size());
    //     int minU = INT_MAX, maxU = INT_MIN, minV = INT_MAX, maxV = INT_MIN;
    //     for (auto& t : ts) {
    //         int u = static_cast<int>(t[1]);
    //         int v = static_cast<int>(t[0]);
    //         minU = std::min(minU, u);
    //         maxU = std::max(maxU, u);
    //         minV = std::min(minV, v);
    //         maxV = std::max(maxV, v);

    //         float currentDepth = depth.at<float>(u, v);
    //         // mask.at<uchar>(u, v) = 1;
    //         depthValues.push_back(currentDepth);
    //     }

    //     std::string filename = "mask_" + std::to_string(i) + ".png";
    //     cv::imwrite(filename, mask * 255);

    //     if (!depthValues.empty()) {
    //         float median = calculateMedian(depthValues);
    //         float threshold = 0.33;
    //         for (int v = std::max(1, minV); v < std::min(rows - 1, maxV + 1); v += 3) {
    //             for (int u = std::max(1, minU); u < std::min(cols - 1, maxU + 1); u += 3) {
    //                 for (int i = v - 1; i <= v + 1; ++i) {
    //                     for (int j = u - 1; j <= u + 1; ++j) {
    //                         float currentDepth = depth.at<float>(i, j);
    //                         if (currentDepth <= 0) continue;
    //                         if (std::abs(currentDepth - median) <= threshold) {
    //                             mask.at<uchar>(i, j) = 1;
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

    cv::Mat merged_mask = CreateMaskFromClusters(clusters_for_mask,depth);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(merged_mask, merged_mask, kernel);

    // for (int y = 0; y < image.rows; ++y) {
    //     for (int x = 0; x < image.cols; ++x) {
    //         ushort depthValue = depth.at<ushort>(y, x);
    //         if (merged_mask.at<uchar>(y, x) == 1) {
    //             drawStar(coloredImage, cv::Point(x, y), 2, cv::Scalar(0, 0, 125), 2);
    //         } 
    //     }
    // }
    return merged_mask;
}    