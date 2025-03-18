#include <iostream>
#include <Eigen/Dense>
#include <random>

std::vector<int> find_index_of_label(Eigen::MatrixXd& label)
{
    std::vector<int> index;
    int value;
    for (int i = 0; i < label.cols(); i++)
    {
        label.col(i).maxCoeff(&value);
        index.push_back(value);
    }
    return index;
};

int main()
{
    // Eigen::MatrixXd weights = Eigen::MatrixXd::Random(1,5);
    // std::cout << weights.array() << std::endl<<std::endl;
    // std::cout << weights.array().mean() << std::endl<<std::endl;

    // Eigen::MatrixXd input_with_bias =  Eigen::MatrixXd::Ones(5,4);
    // std::cout << weights.array() << std::endl<<std::endl;
    // std::cout << input_with_bias.array() << std::endl<<std::endl;
    // weights = weights.block(0, 0, weights.rows()-1, weights.cols());
    // std::cout << weights.array() << std::endl<<std::endl;
    // std::cout << input_with_bias.array() << std::endl<<std::endl;
   //weights_copy = weights;
    // Eigen::MatrixXd k(4,1);
    // k << 2, 2, 3,10;
    // Eigen::MatrixXd c = weights * k;
    // std::cout << weights.array() << std::endl<<std::endl;
    // Eigen::MatrixXd b = weights.cwiseMax(0.0);
    // //weights = weights.array() -weights.minCoeff()/weights.maxCoeff()-weights.minCoeff();
    // std::cout << weights.array() << std::endl<<std::endl;

    // weights = (weights > 0).cast<double>();    
    // std::cout << weights.array() << std::endl<<std::endl;
    // std::cout << b.array() << std::endl<<std::endl;
    // std::cout << &weights << std::endl<<std::endl;
    // std::cout << &weights_copy << std::endl<<std::endl;

    // Eigen::MatrixXd weights2 = Eigen::MatrixXd::Random(4,4);
    // weights_copy = weights2;
    // std::cout << &weights << std::endl<<std::endl;
    // std::cout << &weights_copy << std::endl<<std::endl;
    Eigen::MatrixXd error_tensor(10,10);
    error_tensor << 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                    1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                    0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                    0, 0, 1, 0, 0, 0, 0, 0, 0, 0,       
                    0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                    0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    std::cout << error_tensor.row(0) << std::endl<<std::endl;
    
    // Eigen::VectorXd error_tensor2 = (error_tensor.array() * input.array()).colwise().sum().cast<double>();
    // Eigen::MatrixXd out = input.transpose().array().colwise() * error_tensor2.array();
    // std::cout << error_tensor2 << std::endl<<std::endl;
    // std::cout << input << std::endl<<std::endl;
    // std::cout << out.transpose() << std::endl<<std::endl;

    // Eigen::MatrixXd elementwise_product = error_tensor.array() * y_hat.array();
    // std::cout << elementwise_product << std::endl<<std::endl;

    // Eigen::MatrixXd col_sums = elementwise_product.colwise().sum();
    // std::cout << col_sums << std::endl<<std::endl;

    // Eigen::MatrixXd col_sums_broadcasted = col_sums.replicate(error_tensor.rows(), 1);
    // std::cout << col_sums_broadcasted << std::endl<<std::endl;

    // Eigen::MatrixXd subtracted = error_tensor - col_sums_broadcasted;
    // std::cout << subtracted << std::endl<<std::endl;

    // Eigen::MatrixXd result = y_hat.array() * subtracted.array();
    // std::cout << result << std::endl<<std::endl;
    // Eigen::MatrixXd matrix(6, 10);
    // matrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    //           11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    //           21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    //           31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    //           41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    //           51, 52, 53, 54, 55, 56, 57, 58, 59, 60;

   

    // Calculate the number of steps
    //const int num_steps_cols = matrix.cols() / step_cols;

    //std::cout << matrix.mean() << std::endl;
    // // Iterate over the matrix in steps of 2 columns
    // for (int j = 0; j < num_steps_cols; ++j) {
    //     // Extract the 6x2 submatrix
    //     Eigen::MatrixXd submatrix = matrix.block(0, j * step_cols, step_rows, step_cols);
    //     std::cout << "Submatrix " << j + 1 << ":\n" << submatrix << "\n\n";
    // }






// int count = 0;
    // const int num_samples = images.cols(); // Total number of samples (60000)
    // const int num_batches = num_samples / batch_size; // Number of batches

    // for (int i = 0; i < epoch; i++) {
    //     for (int j = 0; j < num_batches; j++) {
    //         std::cout << "epoch : " << i + 1 << std::endl;
    //         std::cout << "batch : " << j + 1 << std::endl;
    //         count += 1;

    //         // Extract batch of images and labels
    //         int start_col = j * batch_size;
    //         int end_col = start_col + batch_size;

    //         Eigen::MatrixXd batch_images = images.block(0, start_col, images.rows(), batch_size);
    //         Eigen::MatrixXd batch_labels = labels.block(0, start_col, labels.rows(), batch_size);

    //         // Forward pass
    //         Eigen::MatrixXd loss = forward(batch_images, batch_labels);
    //         std::cout << "count : " << count << " cost : " << loss.mean() << std::endl;

    //         // Backward pass
    //         backward(batch_labels, learning_rate);
    //     }
    // }


    return 0;
}