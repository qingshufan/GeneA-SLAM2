#include "gene_a.h"

#include <cstdlib>
#include <ctime>
#include <algorithm>

GeneA::GeneA() {
    srand(time(0));
}
void GeneA::runGeneA(const std::vector<cv::KeyPoint>& keypoints, 
    std::vector<cv::KeyPoint>& all_selected_keypoints, std::vector<cv::KeyPoint>& all_red_keypoints){
    std::vector<std::vector<cv::KeyPoint>> clusters = ae_dbscan(keypoints, all_selected_keypoints);

    for (const auto &cluster : clusters) {
        std::vector<cv::KeyPoint> selected_keypoints;
        std::vector<cv::KeyPoint> red_keypoints;
        std::vector<bool> best_x;

        testGA(cluster, selected_keypoints, red_keypoints, best_x);

        all_selected_keypoints.insert(all_selected_keypoints.end(), std::make_move_iterator(selected_keypoints.begin()), std::make_move_iterator(selected_keypoints.end()));
        all_red_keypoints.insert(all_red_keypoints.end(), std::make_move_iterator(red_keypoints.begin()), std::make_move_iterator(red_keypoints.end()));
    }
}
double GeneA::fitness(const Chromosome& chromosome, const std::vector<cv::KeyPoint>& keypoints, double sum_x, double sum_y) {
    double value = 0;
    for (size_t i = 0; i < chromosome.size(); ++i) {
        if (chromosome[i]) {
            value += 1;
            sum_x -= keypoints[i].pt.x;
            sum_y -= keypoints[i].pt.y;
        }
    }
    double mean_x, mean_y;
    if (keypoints.size() - value != 0) {
        mean_x = sum_x / (keypoints.size() - value);
        mean_y = sum_y / (keypoints.size() - value);
    } else {
        return 0;
    }
    double variance = 0;
    for (size_t i = 0; i < chromosome.size(); ++i) {
        if (!chromosome[i]) {
            const auto& kp = keypoints[i];
            variance += (kp.pt.x - mean_x) * (kp.pt.x - mean_x) + (kp.pt.y - mean_y) * (kp.pt.y - mean_y);
        }
    }
    variance /= (keypoints.size() - value);
    double rate = static_cast<double>(value) / chromosome.size();
    const double mr = 0.25;
    if (rate > mr) {
        variance += (rate - mr / 2) * variance;
    }
    return variance;
}

Chromosome GeneA::selection(const Population& population, const std::vector<cv::KeyPoint>& keypoints, double sum_x, double sum_y) {
    int index1 = rand() % population.size();
    int index2 = rand() % population.size();
    if (fitness(population[index1], keypoints, sum_x, sum_y) < fitness(population[index2], keypoints, sum_x, sum_y)) {
        return population[index1];
    } else {
        return population[index2];
    }
}

Chromosome GeneA::crossover(const Chromosome& parent1, const Chromosome& parent2) {
    Chromosome offspring(parent1.size());
    size_t crossover_point = rand() % parent1.size();
    for (size_t i = 0; i < parent1.size(); ++i) {
        if (i < crossover_point) {
            offspring[i] = parent1[i];
        } else {
            offspring[i] = parent2[i];
        }
    }
    return offspring;
}

void GeneA::mutation(Chromosome& chromosome) {
    for (auto gene : chromosome) {
        double random_number = static_cast<double>(rand()) / RAND_MAX;
        if (random_number < 0.001) {
            gene = !gene;
        }
    }
}

void GeneA::testGA(const std::vector<cv::KeyPoint>& keypoints, std::vector<cv::KeyPoint>& selected_keypoints, 
                   std::vector<cv::KeyPoint>& red_keypoints, std::vector<bool>& best_x) {
    double sum_x = 0, sum_y = 0;
    for (const auto& kp : keypoints) {
        sum_x += kp.pt.x;
        sum_y += kp.pt.y;
    }
    double init_fitness = fitness(Chromosome(keypoints.size()), keypoints, sum_x, sum_y);
    size_t population_size = 10;
    int generations = 50;
    Population population(population_size, Chromosome(keypoints.size()));
    for (auto& chromosome : population) {
        for (auto gene : chromosome) {
            double random_number = static_cast<double>(rand()) / RAND_MAX;
            if (random_number < 0.05) {
                gene = true;
            } else {
                gene = false;
            }
        }
    }
    for (int generation = 0; generation < generations; ++generation) {
        Population new_population;
        while (new_population.size() < population_size) {
            Chromosome parent1 = selection(population, keypoints, sum_x, sum_y);
            Chromosome parent2 = selection(population, keypoints, sum_x, sum_y);
            double random_number = static_cast<double>(rand()) / RAND_MAX;
            if (random_number < 0.8) {
                Chromosome offspring = crossover(parent1, parent2);
                mutation(offspring);
                new_population.push_back(offspring);
            }
        }
        population = new_population;
    }
    auto best_chromosome = *std::min_element(population.begin(), population.end(), [&](const Chromosome& a, const Chromosome& b) {
        return fitness(a, keypoints, sum_x, sum_y) < fitness(b, keypoints, sum_x, sum_y);
    });
    double best_fitness = fitness(best_chromosome, keypoints, sum_x, sum_y);
    for (size_t i = 0; i < best_chromosome.size(); ++i) {
        if (!best_chromosome[i]) {
            selected_keypoints.push_back(keypoints[i]);
        } else {
            red_keypoints.push_back(keypoints[i]);
        }
    }
    best_x = best_chromosome;
}

std::vector<std::vector<cv::KeyPoint>> GeneA::ae_dbscan(const std::vector<cv::KeyPoint>& keypoints, std::vector<cv::KeyPoint>& all_selected_keypoints) {
    if (keypoints.empty()) {
        return std::vector<std::vector<cv::KeyPoint>>();
    }
    int input_size = 6;
    int hidden_size = 2;
    int epochs = 100;
    double learning_rate = 0.05;
    Autoencoder autoencoder(input_size, hidden_size);
    std::vector<Eigen::VectorXd> data = autoencoder.keypointsToEigen(keypoints);
    autoencoder.train(data, epochs, learning_rate, data.size());
    std::vector<Eigen::VectorXd> hidden_representations;
    hidden_representations.reserve(data.size());
    for (const auto& point : data) {
        hidden_representations.push_back(autoencoder.get_hidden_representation(point));
    }
    if (hidden_representations.empty()) {
        return std::vector<std::vector<cv::KeyPoint>>();
    }
    double eps = autoencoder.determine_eps(hidden_representations, 0.05);
    int minPts = hidden_representations[0].size() + 1;
    std::vector<int> labels = autoencoder.dbscan(hidden_representations, eps, minPts);
    for (size_t j = 0; j < keypoints.size(); ++j) {
        if (labels[j] == 0) {
            all_selected_keypoints.push_back(keypoints[j]);
        }
    }
    return autoencoder.getClusters(keypoints, labels);
}