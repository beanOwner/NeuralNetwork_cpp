#include <iostream>
#include "mnist_loader.hpp"


int main(int argc, char* argv[])
{
    // assert(argc == 4);
    std::string input_path_images = argv[1];
    std::string output_path_images = argv[2];
    int index = std::stoi(argv[3]);

    MNIST train_images;
    // std::string input_path_images = "../data/train-images-idx3-ubyte";
    // std::string output_path_images = "output.txt";
    // int index = 0;
    train_images.read_dataset_images(input_path_images);

    Eigen::MatrixXd matrix_images = train_images.getting_Images().transpose();
    std::cout << matrix_images.rows() << "    " << matrix_images.cols() << std::endl << std::endl;

    std::ofstream file(output_path_images);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file: " + input_path_images);
    }
    // 2) Write the dimention
    file << 2 << "\n";

    // 2) Write the shape: number of rows and columns
    file << train_images.getting_imageRows() << "\n";
    file << train_images.getting_imageCols() << "\n";

    // 3) Write the data in row-major order
    for (int r = 0; r < matrix_images.rows(); ++r)
    {
        file << matrix_images(r, index) << "\n";
        
    }

    file.close();

    return 0;
}