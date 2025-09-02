#include "NeuralNetwork.h"
#include "CMatrix.cuh"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <stdexcept>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void CudaVNonCuda();
std::vector<std::pair<CMatrix, CMatrix>> readTestData();
std::vector<std::pair<CMatrix, CMatrix>> readTrainingData();

int main() {
    //Takes in mnist training sets
    std::vector<std::pair<CMatrix, CMatrix>> testData = readTestData();
    std::vector<std::pair<CMatrix, CMatrix>> trainData = readTrainingData();
    std::cout << "Train Data samples: " << trainData.size() << "\n";
    std::cout << "Test Data samples: " << testData.size() << "\n";

    /*
    Initializes network strucutre. 
    784 (28x28) neurons in layer 1 with 784x10 weights total (CUDA IS GOATED AT THIS)
    10 neurons layer 2 with again 10x10 weights total
    Final layer holds output information. We are choosing 1 out of 10 outputs
    */
    int layer1 = testData[0].first.height;
    const int layer2 = 16;
    const int layer3 = 12;
    const int layer4 = 10;
    const int networkSize = 4;
    const int epochs = 100;
    const int batchSize = 32;
    const double learningRate = .05;
    int networkStructure[] = {layer1, layer2, layer3, layer4};
    NeuralNetwork network = NeuralNetwork(networkStructure, networkSize);
    network.stochasticGradDescent(trainData, epochs, batchSize, learningRate, testData);
    //CudaVNonCuda();
    return 0;
}

//Temporary helper function to test variuos CMat functions
void CudaVNonCuda() {
    //Creates and sets a bunch of CMatrix's (Mainly for testing purposes)
    const int iterations = 1;
    const int matrix_scale_factor = 512;

    std::function<double(int, int)> foo = [](int x, int y) {
        return 1;
    };

    //This does a bunch of Matrix multiplications.
    for(int i = 1; i < iterations*matrix_scale_factor; i+=matrix_scale_factor) {
        CMatrix m1 = createCMatrix(i*matrix_scale_factor, i*matrix_scale_factor);
        CMatrix m2 = createCMatrix(i*matrix_scale_factor, i*matrix_scale_factor);
        setCMatrix(foo, m1);
        setCMatrix(foo, m2);
        std::cout << "Size of Matrix 1 is : " << m1.height << "x" << m1.width << std::endl;

        CMatrix m3 = multiply_cuda(m1,m2);
        std::cout << "Size of Matrix 2 is : " << m3.height << "x" << m3.width << std::endl;

        CMatrix m4 = CMatrixMultiply(m1,m2);
        std::cout << "Size of Matrix 3 is : " << m4.height << "x" << m4.width << std::endl;

        //printCMatrix(m1);
        //printCMatrix(m2);
        //printCMatrix(m3);
        printCMatrix(m4);

        freeCMatrix(m1);
        freeCMatrix(m2);
        freeCMatrix(m3);
   }
}

//Reads in test data for our NN to store
std::vector<std::pair<CMatrix, CMatrix>> readTestData() {
    std::ifstream file("data/mnist_test.csv");

    if (!file.is_open()) {
        throw std::runtime_error("Error: File could not be opened.");
    }

    std::string line;
    std::vector<std::pair<CMatrix, CMatrix>> testData;
    
    //Need to consume the first line since its just header information
    std::getline(file, line);

    const int limit = 100;
    int count = 0;
    while (std::getline(file, line) && (count < limit || limit == -1)) {
        count++;
        std::stringstream ss(line);
        std::string value;

        std::vector<double> row;
        while (std::getline(ss, value, ',')) {
            double dValue = std::stod(value);
            row.push_back(dValue);
        }

        int firstValue = row[0];
        std::function<double(int, int)> firstValueFunc;
        firstValueFunc = [firstValue](int x, int y) {
            if (x == firstValue) {
                return 1.0;
            }
            return 0.0;
        };
        CMatrix expectedOutput = createCMatrix(10, 1);
        setCMatrix(firstValueFunc, expectedOutput);

        CMatrix testingDataCMatrix = createCMatrix(row.size()-1, 1);
        row.erase(row.begin());
        
        std::function<double(int, int)> foo;
        foo = [row](int x, int y) {
            return (row[x]/255.0);
        };
        setCMatrix(foo, testingDataCMatrix);

        testData.push_back(std::make_pair(testingDataCMatrix, expectedOutput));
    }

    file.close();
    return testData;
}

//Reads in training data for our NN to store
std::vector<std::pair<CMatrix, CMatrix>> readTrainingData() {
    std::ifstream file("data/mnist_train.csv");

    if (!file.is_open()) {
        throw std::runtime_error("Error: File could not be opened.");
    }

    std::string line;
    std::vector<std::pair<CMatrix, CMatrix>> testData;
    
    //Need to consume the first line since its just header information
    std::getline(file, line);

    const int limit = 1000;
    int count = 0;
    while (std::getline(file, line) && (count < limit || limit == -1)) {
        count++;
        std::stringstream ss(line);
        std::string value;

        std::vector<double> row;
        while (std::getline(ss, value, ',')) {
            double dValue = std::stod(value);
            row.push_back(dValue);
        }

        int firstValue = row[0];
        std::function<double(int, int)> firstValueFunc;
        firstValueFunc = [firstValue](int x, int y) {
            if (x == firstValue) {
                return 1.0;
            }
            return 0.0;
        };
        CMatrix expectedOutput = createCMatrix(10, 1);
        setCMatrix(firstValueFunc, expectedOutput);

        CMatrix testingDataCMatrix = createCMatrix(row.size()-1, 1);
        row.erase(row.begin());

        std::function<double(int, int)> foo;
        foo = [row](int x, int y) {
            return (row[x]/255.0);
        };
        setCMatrix(foo, testingDataCMatrix);

        testData.push_back(std::make_pair(testingDataCMatrix, expectedOutput));
    }

    file.close();
    return testData;
}