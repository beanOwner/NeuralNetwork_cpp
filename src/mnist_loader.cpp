#include "mnist_loader.hpp"

void MNIST::read_dataset_images(const std::string& file_path)
{
    std::ifstream file(file_path, std::iostream::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open image file: " + file_path);
    }
    //read the file headers 
    magicNumber = readInt(file);
    numItems = readInt(file);
    imageRows = readInt(file);
    imageCols = readInt(file);

    size_t num_pixels_per_image = static_cast<size_t>(imageRows) * static_cast<size_t>(imageCols);

    images2D.resize(static_cast<size_t>(numItems),num_pixels_per_image);
    //std::cout<< std::endl<< std::endl;
    //std::cout << "num_pixels_per_image: " << num_pixels_per_image << std::endl<< std::endl<< std::endl;           
    //std::cout << "numItems: " << numItems << std::endl<< std::endl<< std::endl;            
    //std::cout << "imageRows: " << imageRows << std::endl<< std::endl<< std::endl;
    //std::cout << "imageCols: " << imageCols << std::endl<< std::endl<< std::endl; 
    //read images 
    for(size_t i = 0 ; i < static_cast<size_t>(numItems) ; ++i)
    {
        for (int j = 0; j < num_pixels_per_image; ++j)
        {
            unsigned char pixel = 0 ;
            file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            images2D(i,j) = static_cast<double>(pixel)/255.0;
        }
        //std::cout << i << std::endl ;
        
    }
    //for (int i=0 ; i<num_pixels_per_image;++i)
            //{
                //std::cout<< images2D(4,i) << "  ";
            //}
    file.close();
    //standardization(images2D);
    //std::cout << std::endl<< std::endl<< std::endl<< std::endl;
};
void MNIST::read_dataset_labels(const std::string& file_path)
{
    std::ifstream file(file_path, std::iostream::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open image file: " + file_path);
    }
    magicNumber = readInt(file);
    
    //std::cout << std::endl<< magicNumber<<"\n"<<"\n";   
    

    numItems = readInt(file);

    //std::cout << std::endl<< numItems<<"\n"<<"\n";
    labels2D.resize(static_cast<size_t>(numItems),10);
    labels2D.setZero();

    
    for(int i = 0; i < static_cast<size_t>(numItems) ;++i)
    {   unsigned char label = 0 ;
        file.read(reinterpret_cast<char*>(&label),sizeof(label));
        labels2D(i,static_cast<size_t>(label))=1;
    // for (int i=0;i<10;++i)
    // {
    //     std::cout << std::endl;
    //     std::cout << i << ": " << labels2D(2,i) << std::endl;
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;
    file.close();}
const Eigen::MatrixXd& MNIST::getting_Images() const {return images2D;}
const Eigen::MatrixXd& MNIST::getting_Labels() const {return labels2D;}
const float& MNIST::getting_imageRows() const{return imageRows;}
const float& MNIST::getting_imageCols() const{return imageCols;}
int MNIST::readInt(std::ifstream& file)
{   int value = 0 ;
    file.read(reinterpret_cast<char*>(&value),4); //It needs to be convert to big endian because it takes 4 bytes in little endian
    value = __builtin_bswap32(value);// convert little endian to big endian 
    return static_cast<float>(value);}

