#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <algorithm>
#include <random>
#include <Eigen/Dense>
#include <cmath>

class Neural_Network
{
public:
    Neural_Network(int input_size, int hidden_size, int output_size);
    ~Neural_Network();
    
    void train(Eigen::MatrixXd& images, Eigen::MatrixXd& labels, int epochs, double learning_rate, int batch_size);
    void test(Eigen::MatrixXd& images, Eigen::MatrixXd& labels);
    std::vector<int> getting_label_values() const;
    std::vector<int> getting_predicted_label() const;

private:
    const int input_size;
    const int hidden_size;
    const int output_size;
    
    // ADAM optimizer hyperparameters
    const double beta1 = 0.9;    // Exponential decay rate for first moment
    const double beta2 = 0.999;  // Exponential decay rate for second moment
    const double epsilon = 1e-8; // Small constant for numerical stability
    
    // Network weights
    Eigen::MatrixXd weightsLayer1;
    Eigen::MatrixXd weightsLayer2;
    
    // ADAM optimizer variables
    Eigen::MatrixXd m1;  // First moment for layer 1
    Eigen::MatrixXd v1;  // Second moment for layer 1
    Eigen::MatrixXd m2;  // First moment for layer 2
    Eigen::MatrixXd v2;  // Second moment for layer 2
    int t = 0;    // Time step counter
    
    // Intermediate results - resized based on batch size
    Eigen::MatrixXd hidden_layer_output;
    Eigen::MatrixXd output_layer_output;
    Eigen::MatrixXd output_layer_soft_max;
    Eigen::MatrixXd output_layer_loss;
    Eigen::MatrixXd relu_input;
    Eigen::MatrixXd hidden_layer_with_bias;
    Eigen::MatrixXd input_with_bias;
    
    // Results storage
    std::vector<int> predicted_label;
    std::vector<int> label_values;
    
    // Forward and backward propagation
    Eigen::MatrixXd forward(Eigen::MatrixXd& input, Eigen::MatrixXd& label);
    void backward(Eigen::MatrixXd& label, double learning_rate);
    
    // Layer operations
    Eigen::MatrixXd relu_forward(const Eigen::MatrixXd& x);
    Eigen::MatrixXd relu_backward(const Eigen::MatrixXd& x);
    Eigen::MatrixXd soft_max_forward(const Eigen::MatrixXd& x);
    Eigen::MatrixXd soft_max_backward(const Eigen::MatrixXd& d_output, const Eigen::MatrixXd& soft_max_output);
    Eigen::MatrixXd cross_loss_forward(const Eigen::MatrixXd& pred, const Eigen::MatrixXd& label);
    Eigen::MatrixXd cross_loss_backward(const Eigen::MatrixXd& pred, const Eigen::MatrixXd& label);
    
    // Helper functions
    void adam_update(Eigen::MatrixXd& weights, Eigen::MatrixXd& m, Eigen::MatrixXd& v, 
                    const Eigen::MatrixXd& gradient, double learning_rate);
    
    // Resize intermediate matrices for new batch size
    void resize_matrices(int batch_size);
};

#endif