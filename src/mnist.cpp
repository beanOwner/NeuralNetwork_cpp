#include <iostream>
#include "mnist_loader.hpp"
#include "Neural_Network.hpp"
#include<fstream>
#include <Eigen/Dense>
void standardization(Eigen::MatrixXd& images)
{
    Eigen::MatrixXd meanVals = images.rowwise().mean().replicate(1,images.cols());
    Eigen::MatrixXd Xsq = images.array().square().matrix();      // shape (784, num_images)
    Eigen::MatrixXd meanSqVals = Xsq.rowwise().mean().replicate(1,images.cols());
    Eigen::MatrixXd stdVals = (meanSqVals.array() - meanVals.array().square()).sqrt().matrix();
    stdVals.array() += 1e-12;
    images.array() -= meanVals.array();
    images.array() /= stdVals.array();
};
void prediction_logger(std::vector<int> label_values,std::vector<int> predicted_label,std::string log_file,int batch)
{
    std::ofstream file(log_file);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file: " + log_file);
    }
    for (int j= 0; j < label_values.size() / batch; j++)
    {   
        file << "Current batch: " << j << std::endl;
        int start = j * batch;
        int end = (j+1) * batch;
        end = (j+1) * batch;
        for (int i = start; i < end; i++)
        {
            file <<" - "<<"image "<< i << ": Prediction="<< predicted_label[i] << ". Label=" << label_values[i] << std::endl;
        }
    }
    file.close();
};
int main(int argc, char* argv[])
{
    if (argc != 10) {
        std::cerr << "Usage: " << argv[0] << " <train_images> <train_labels> <test_images> <test_labels> <log_file> <epochs> <batch_size> <hidden_size> <learning_rate>" << std::endl;
        return 1;
    }

    std::string path_trainable_images = std::string(argv[1]);
    std::string path_trainable_label = std::string(argv[2]);
    std::string path_testable_images = std::string(argv[3]);
    std::string path_testable_label = std::string(argv[4]);
    std::string log_file = std::string(argv[5]);  // Use log file path as provided without modification
    int epoch = std::stoi(argv[6]);
    int batch = std::stoi(argv[7]);
    int hidden_size = std::stoi(argv[8]);
    double learning_rate = std::stod(argv[9]);
    //===================================================================================================
    // std::string path_trainable_images = "../data/train-images-idx3-ubyte";
    // std::string path_trainable_label = "../data/train-labels-idx1-ubyte";
    // std::string path_testable_images = "../data/t10k-images-idx3-ubyte";
    // std::string path_testable_label = "../data/t10k-labels-idx1-ubyte";
    // std::string log_file = "../logout.txt";
    // int epoch = 10;
    // int batch = 100;
    // int hidden_size = 500;
    // double learning_rate = 1e-3;
//===================================================================================================
    MNIST train_images;
    MNIST train_labels;
    MNIST test_images;
    MNIST test_labels;
    train_images.read_dataset_images(path_trainable_images);
    train_labels.read_dataset_labels(path_trainable_label);
    test_images.read_dataset_images(path_testable_images);
    test_labels.read_dataset_labels(path_testable_label);
    Eigen::MatrixXd matrix_trainable_images = train_images.getting_Images().transpose();
    Eigen::MatrixXd matrix_trainable_labels = train_labels.getting_Labels().transpose();
    Eigen::MatrixXd matrix_testable_images = test_images.getting_Images().transpose();
    Eigen::MatrixXd matrix_testable_labels = test_labels.getting_Labels().transpose();
    standardization(matrix_trainable_images);
    standardization(matrix_testable_images);
    // std::cout << train_images.getting_imageRows() << "    " << train_images.getting_imageCols() << std::endl << std::endl;
    // std::cout << matrix_trainable_images.rows() << "    " << matrix_trainable_images.cols() << std::endl << std::endl;
    // std::cout << matrix_trainable_labels.rows() << "    " << matrix_trainable_labels.cols() << std::endl << std::endl;
    // std::cout << matrix_testable_images.rows() <<  "    " << matrix_testable_images.cols() << std::endl << std::endl;
    // std::cout << matrix_testable_labels.rows() <<  "    " << matrix_testable_labels.cols() << std::endl << std::endl;
    // Eigen::MatrixXd input = matrix_trainable_images.block(0,0,784,1000);
    // Eigen::MatrixXd label = matrix_trainable_labels.block(0,0,10,1000);
    // Eigen::MatrixXd input_test = matrix_testable_images.block(0,0,784,100);
    // Eigen::MatrixXd label_test = matrix_testable_labels.block(0,0,10,100);
    // // Eigen::MatrixXd input_3 = matrix_trainable_images.block(0,1000,784,500);
    // // Eigen::MatrixXd label_3 = matrix_trainable_labels.block(0,1000,10,500);
    // // Eigen::MatrixXd input_4 = matrix_trainable_images.block(0,1500,784,500);
    // // Eigen::MatrixXd label_4 = matrix_trainable_labels.block(0,1500,10,500);
    Neural_Network model1(784,hidden_size,10);
    // //Eigen::MatrixXd weights = model1.getting_data();
    // //std::cout << weights.rows() <<"     "<< weights.cols()<< std::endl << std::endl ;
    model1.train(matrix_trainable_images,matrix_trainable_labels,epoch,learning_rate,batch);
    model1.test(matrix_testable_images,matrix_testable_labels);
    std::vector<int> label_values = model1.getting_label_values();
    std::vector<int> predicted_label = model1.getting_predicted_label();
    prediction_logger(label_values,predicted_label,log_file,batch);

    return 0;
}
