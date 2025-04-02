#include "Neural_Network.hpp"
#include "../eigen/3.4.0_1/include/eigen3/Eigen/Dense"
#include <iostream>
#include <cmath>
#include <algorithm>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {
    // Helper function for Xavier initialization
    inline double xavier_limit(const int fan_in, const int fan_out) {
        return std::sqrt(6.0 / (fan_in + fan_out));
    }
}

Neural_Network::Neural_Network(int input_size, int hidden_size, int output_size)
    : input_size(input_size), hidden_size(hidden_size), output_size(output_size) {
    
    // Create layers
    layers.push_back(std::make_unique<DenseLayer>(input_size, hidden_size));
    layers.push_back(std::make_unique<ReLULayer>(hidden_size));
    layers.push_back(std::make_unique<DenseLayer>(hidden_size, output_size));
    layers.push_back(std::make_unique<SoftmaxLayer>(output_size));
}

Neural_Network::~Neural_Network() {
    layers.clear();
}

void Neural_Network::resize_matrices(int batch_size)
{
    // Resize all intermediate matrices to match the new batch size
    hidden_layer_output.resize(hidden_size, batch_size);
    output_layer_output.resize(output_size, batch_size);
    output_layer_soft_max.resize(output_size, batch_size);
    output_layer_loss.resize(output_size, batch_size);
    relu_input.resize(hidden_size, batch_size);
    hidden_layer_with_bias.resize(hidden_size + 1, batch_size);
    input_with_bias.resize(input_size + 1, batch_size);
}

Eigen::MatrixXd Neural_Network::forward(Eigen::MatrixXd& input, Eigen::MatrixXd& label)
{
    if (input.rows() != input_size) {
        throw std::runtime_error("Input dimension mismatch in forward pass");
    }
    
    const int batch_size = input.cols();
    
    // Resize intermediate matrices if needed
    if (hidden_layer_output.cols() != batch_size) {
        resize_matrices(batch_size);
    }
    
    // Add bias term to input
    input_with_bias.block(0, 0, input_size, batch_size) = input;
    input_with_bias.row(input_size).setOnes();
    
    // Forward to hidden layer
    relu_input = weightsLayer1 * input_with_bias;
    hidden_layer_output = relu_forward(relu_input);
    
    // Add bias term to hidden layer
    hidden_layer_with_bias.block(0, 0, hidden_size, batch_size) = hidden_layer_output;
    hidden_layer_with_bias.row(hidden_size).setOnes();
    
    // Forward to output layer
    output_layer_output = weightsLayer2 * hidden_layer_with_bias;
     
    // Compute softmax
    output_layer_soft_max = soft_max_forward(output_layer_output);
    
    // Compute cross-entropy loss
    output_layer_loss = -label.array() * output_layer_soft_max.array().log();
    
    if (!output_layer_loss.allFinite()) {
        throw std::runtime_error("Non-finite values detected in loss computation");
    }
    
    return output_layer_loss;
}

void Neural_Network::backward(Eigen::MatrixXd& label, double learning_rate)
{
    const int batch_size = label.cols();
    
    // Compute gradients
    Eigen::MatrixXd d_output = cross_loss_backward(output_layer_soft_max, label);
    Eigen::MatrixXd error_tensor_of_soft_max = soft_max_backward(d_output ,output_layer_soft_max);
    Eigen::MatrixXd d_hidden = weightsLayer2.transpose() * error_tensor_of_soft_max;
    d_hidden = d_hidden.block(0, 0, hidden_size, batch_size).array() * relu_backward(relu_input).array();
    
    // Compute weight gradients
    Eigen::MatrixXd grad2 = (d_output * hidden_layer_with_bias.transpose()) / batch_size;
    Eigen::MatrixXd grad1 = (d_hidden * input_with_bias.transpose()) / batch_size;
    
    // Update weights using ADAM
    t++; // Increment time step
    adam_update(weightsLayer2, m2, v2, grad2, learning_rate);
    adam_update(weightsLayer1, m1, v1, grad1, learning_rate);
}

void Neural_Network::adam_update(Eigen::MatrixXd& weights, Eigen::MatrixXd& m, Eigen::MatrixXd& v, 
                               const Eigen::MatrixXd& gradient, double learning_rate)
{
    // Update biased first moment estimate
    m = beta1 * m + (1.0 - beta1) * gradient;
    
    // Update biased second raw moment estimate
    v = beta2 * v + (1.0 - beta2) * gradient.array().square().matrix();
    
    // Compute bias-corrected first moment estimate
    Eigen::MatrixXd m_hat = m / (1.0 - std::pow(beta1, t));
    
    // Compute bias-corrected second raw moment estimate
    Eigen::MatrixXd v_hat = v / (1.0 - std::pow(beta2, t));
    
    // Update parameters
    weights -= learning_rate * (m_hat.array() / (v_hat.array().sqrt() + epsilon)).matrix();
}

Eigen::MatrixXd Neural_Network::relu_forward(const Eigen::MatrixXd& x)
{
    return x.array().max(0.0);
}

Eigen::MatrixXd Neural_Network::relu_backward(const Eigen::MatrixXd& x)
{
    return (x.array()>0).cast<double>();
}

Eigen::MatrixXd Neural_Network::soft_max_forward(const Eigen::MatrixXd& x)
{
    Eigen::MatrixXd colwise_max = x.colwise().maxCoeff().replicate(x.rows(),1);
    Eigen::MatrixXd stable_input = x.array() - colwise_max.array();
    Eigen::MatrixXd exp_input = stable_input.array().exp();
    Eigen::VectorXd sums = exp_input.colwise().sum();
    output_layer_soft_max = exp_input.array().rowwise() / sums.transpose().array();
    return output_layer_soft_max;
}

Eigen::MatrixXd Neural_Network::soft_max_backward(const Eigen::MatrixXd& d_output, const Eigen::MatrixXd& soft_max_output)
{
    // Compute Jacobian-vector product for softmax backward pass
    return soft_max_output.array() * (d_output.array() - (soft_max_output.array() * d_output.array()).colwise().sum().replicate(d_output.rows(), 1));
}

Eigen::MatrixXd Neural_Network::cross_loss_forward(const Eigen::MatrixXd& pred, const Eigen::MatrixXd& label)
{
    return -label.array() * pred.array().log();
}

Eigen::MatrixXd Neural_Network::cross_loss_backward(const Eigen::MatrixXd& pred, const Eigen::MatrixXd& label)
{
    const double epsilon = 1e-8;
    return -1*(label.cwiseQuotient((pred.array()+epsilon).matrix()));
}

void Neural_Network::test(Eigen::MatrixXd& images, Eigen::MatrixXd& labels)
{
    // Process each image
    predicted_label.clear();
    label_values.clear();
    int correct_predictions = 0;
    
    for (int i = 0; i < images.cols(); ++i) {
        Eigen::MatrixXd single_image = images.col(i);
        Eigen::MatrixXd single_label = labels.col(i);
        
        // Forward pass
        Eigen::MatrixXd output = forward(single_image, single_label);
        
        // Get predicted label
        Eigen::Index max_index;
        output_layer_soft_max.col(0).maxCoeff(&max_index);
        predicted_label.push_back(static_cast<int>(max_index));
        
        // Get actual label
        Eigen::Index true_label;
        single_label.col(0).maxCoeff(&true_label);
        label_values.push_back(static_cast<int>(true_label));

        // Check if prediction is correct
        if (predicted_label.back() == label_values.back()) {
            correct_predictions++;
        }
    }
    
    // Calculate accuracy
    std::cout<<"================================================"<<std::endl;
    double accuracy = static_cast<double>(correct_predictions) / images.cols();
    std::cout << "Accuracy: " << accuracy*100 << "%" << std::endl;
    std::cout<<"================================================"<<std::endl;
}

void Neural_Network::train(Eigen::MatrixXd& images, Eigen::MatrixXd& labels, int epochs, double learning_rate, int batch_size)
{
    int num_samples = images.cols();
    int num_batches = (num_samples + batch_size - 1) / batch_size;  // Ceiling division
    
    // Resize intermediate matrices for the batch size
    resize_matrices(batch_size);
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double epoch_loss = 0.0;
        int processed_samples = 0;
        
        // Process each batch
        for (int batch = 0; batch < num_batches; ++batch) {
            int current_batch_size = std::min(batch_size, num_samples - batch * batch_size);
            
            // Create views for the current batch
            Eigen::MatrixXd batch_images = images.block(0, batch * batch_size, images.rows(), current_batch_size);
            Eigen::MatrixXd batch_labels = labels.block(0, batch * batch_size, labels.rows(), current_batch_size);
            
            // Forward pass
            Eigen::MatrixXd loss = forward(batch_images, batch_labels);
            
            // Check for invalid loss values
            if (!loss.allFinite()) {
                std::cerr << "Warning: Non-finite loss detected in epoch " << epoch << ", batch " << batch << std::endl;
                continue;
            }
            
            // Backward pass
            backward(batch_labels, learning_rate);
            
            // Accumulate loss
            epoch_loss += loss.sum();
            processed_samples += current_batch_size;
        }
        
        // Print epoch statistics
        if (processed_samples > 0) {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                      << ", Loss: " << epoch_loss / processed_samples << std::endl;
        }
    }
}

std::vector<int> Neural_Network::getting_label_values() const
{
    return label_values;
}

std::vector<int> Neural_Network::getting_predicted_label() const
{
    return predicted_label;
}