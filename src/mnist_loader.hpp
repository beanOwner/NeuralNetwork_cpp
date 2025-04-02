// Header file: mnist.hpp
#ifndef MNIST_LOADER_HPP
#define MNIST_LOADER_HPP

#include<iostream>
#include<fstream>
#include<vector>
#include<stdexcept>
#include<string>
#include<cassert>
#include<algorithm>
#include <Eigen/Dense>


class MNIST
{
public:
    // Reads the MNIST image file
    void read_dataset_images(const std::string& file_path);

    // Reads the MNIST label file
    void read_dataset_labels(const std::string& file_path);

    // Accessor for images
    const Eigen::MatrixXd& getting_Images() const;

    // Accessor for number of items
    const Eigen::MatrixXd& getting_Labels() const;

    // Accessor for image rows
    const float& getting_imageRows() const;

    // Accessor for image columns
    const float& getting_imageCols() const;

    //void writeEigen_image_MatrixToFile(const std::string& filename) const;


private:
    Eigen::MatrixXd images2D;
    Eigen::MatrixXd labels2D;
    float magicNumber;
    float numItems; 
    float imageRows;
    float imageCols;
    ///Reads a 4-byte integer from a file in big-endian format
    int readInt(std::ifstream& file);
};

#endif // MNIST_HPP