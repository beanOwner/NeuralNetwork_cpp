#include <iostream>
#include "mnist_loader.hpp"


int main(int argc, char* argv[])
{
    std::string input_path_labels = std::string(argv[1]);
    std::string output_path_labels = std::string(argv[2]);
    int index = std::stoi(argv[3]);
    MNIST labels;
    // std::string input_path_labels = "../data/t10k-labels-idx1-ubyte";
    // std::string output_path_labels = "output.txt";
    // int index = 0;
    labels.read_dataset_labels(input_path_labels);

    Eigen::MatrixXd matrix_labels = labels.getting_Labels().transpose();
    std::cout << matrix_labels.rows() << "    " << matrix_labels.cols() << std::endl << std::endl;

    std::ofstream file(output_path_labels);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file: " + input_path_labels);
    }
    // 2) Write the dimention

    // 2) Write the shape: number of rows and columns
    file << matrix_labels.col(index).cols() << "\n";
    file << matrix_labels.col(index).rows() << "\n";

    // 3) Write the data in row-major order
    for (int r = 0; r < matrix_labels.rows(); ++r)
    {
        file << matrix_labels(r, index) << "\n";
        
    }

    file.close(); 

    return 0;
}