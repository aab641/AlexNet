//
//  main.cpp
//  AlexNet
//
//  Created by Alexander Bedrossian on 2/5/19.
//  Copyright © 2019 Bedrossian. All rights reserved.
//

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include <fstream>
#include <stdio.h>
#include <stdlib.h>     /* atof */
#include <string.h>
#include <time.h>
#include <math.h>
#include <sys/stat.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <iomanip>

using namespace std;

typedef struct {
    int width;
    int height;
    float* elements;
    int stride;
} Matrix;

class dataImage {
public:
    unsigned char pixArray[28 * 28];
    unsigned char label[1];
    
    friend std::ostream & operator<<(std::ostream & _stream, dataImage const & dataImg) {
        _stream << "Label: " << unsigned(dataImg.label[0]) << endl;
        _stream << "Image: " << endl;
        for (int x = 0; x < 28*28; x++) {
            _stream << setfill('0') << setw(1) << (int) dataImg.pixArray[x] << " ";
            if (x % 28 == 0)
                _stream << endl;
            
        }
        return _stream;
    }
};

class neuron {
public:
    
    float bias = 0.0;
    vector<float> weights;
    //float zee = 0.0;
    float dk;
    vector<float> nudges;
    float val  = 0;
    
    neuron() : weights(vector<float>()) {}
    neuron(const int layerSize) :  weights(vector<float>(layerSize,0.01)),nudges(vector<float>(layerSize, 0)) {}
    ~neuron() {
        weights.empty();
        //delete &weights;
    }
};

// Just loads the test files
class database {
private:
    int images_magic_number;
    int labels_magic_number;
    int number_of_images;
    int number_of_labels;
    int nrows;
    int ncolumns;
    
public:
    
    vector<dataImage*> *dataset;
    
    database(const string imageDBfilename, const string labelDBfilename) {
        bool dbgFunction = false;
        std::vector<unsigned char> *image_buffer;
        
        ifstream image_file(imageDBfilename, std::ios::binary | std::ios::ate | std::ios::in);
        
        
        if (image_file.is_open()) {
            std::streamsize size = image_file.tellg();
            image_file.seekg(0, std::ios::beg);
            
            image_buffer = new std::vector<unsigned char>(size);
            if (image_file.read((char*)image_buffer->data(), size))
            {
                if (dbgFunction)
                    cout << "Opened and loaded Image Dataset." << endl;
            }
            image_file.close();
            if (dbgFunction)
                cout << "Image Dataset Closed." << endl;
        }
        else {
            cout << "Unable to open Image Dataset file." << endl;
            exit(0);
        }
        if (dbgFunction)
            cout << "Building training database. . ." << endl;
        
        images_magic_number = int(
                                  (unsigned char)(image_buffer->at(0)) << 24 |
                                  (unsigned char)(image_buffer->at(1)) << 16 |
                                  (unsigned char)(image_buffer->at(2)) << 8 |
                                  (unsigned char)(image_buffer->at(3))
                                  );
        
        number_of_images = int(
                               (unsigned char)(image_buffer->at(4)) << 24 |
                               (unsigned char)(image_buffer->at(5)) << 16 |
                               (unsigned char)(image_buffer->at(6)) << 8 |
                               (unsigned char)(image_buffer->at(7))
                               );
        
        nrows = int(
                    (unsigned char)(image_buffer->at(8)) << 24 |
                    (unsigned char)(image_buffer->at(9)) << 16 |
                    (unsigned char)(image_buffer->at(10)) << 8 |
                    (unsigned char)(image_buffer->at(11))
                    );
        
        ncolumns = int(
                       (unsigned char)(image_buffer->at(12)) << 24 |
                       (unsigned char)(image_buffer->at(13)) << 16 |
                       (unsigned char)(image_buffer->at(14)) << 8 |
                       (unsigned char)(image_buffer->at(15))
                       );
        
        dataset = new vector<dataImage*>(number_of_images);
        
        //cout << "Magic number: " << images_magic_number << endl;
        if (dbgFunction)
            cout << "Total images: " << number_of_images << endl;
        //cout << "ncol: " << ncolumns << endl;
        //cout << "nrow: " << nrows << endl;
        
        int memblockCursor = 16; //Point to the begining of our first image
        
        for (int i = 0; i < number_of_images; i++) {
            dataImage *nImage = new dataImage();
            for (int row = 0; row < nrows*nrows; row++) {
                nImage->pixArray[row] = image_buffer->at(memblockCursor++);
            }
            dataset->at(i) = nImage;
            //nImage->print();
        }
        if (dbgFunction)
            cout << "Releasing Image File Buffer." << endl;
        delete image_buffer;
        
        ifstream label_file(labelDBfilename, std::ios::binary | std::ios::ate);
        
        std::vector<char> *label_buffer;
        
        if (label_file.is_open()) {
            std::streamsize size = label_file.tellg();
            label_file.seekg(0, std::ios::beg);
            
            label_buffer = new std::vector<char>(size);
            if (label_file.read(label_buffer->data(), size))
            {
                if (dbgFunction)
                    cout << "Opened and loaded Label Dataset." << endl;
            }
            label_file.close();
            if (dbgFunction)
                cout << "Label Dataset Closed." << endl;
        }
        else {
            cout << "Unable to open Label Dataset file." << endl;
            exit(0);
        }
        
        labels_magic_number = int(
                                  (unsigned char)(label_buffer->at(0)) << 24 |
                                  (unsigned char)(label_buffer->at(1)) << 16 |
                                  (unsigned char)(label_buffer->at(2)) << 8 |
                                  (unsigned char)(label_buffer->at(3))
                                  );
        
        number_of_labels = int(
                               (unsigned char)(label_buffer->at(4)) << 24 |
                               (unsigned char)(label_buffer->at(5)) << 16 |
                               (unsigned char)(label_buffer->at(6)) << 8 |
                               (unsigned char)(label_buffer->at(7))
                               );
        
        
        memblockCursor = 8; //Point to the begining of our first image
        
        for (int i = 0; i < number_of_labels; i++) {
            dataImage *nImage = dataset->at(i);
            nImage->label[0] = (unsigned char) label_buffer->at(memblockCursor++);
        }
        if (dbgFunction)
            cout << "Releasing Label File Buffer." << endl;
        delete label_buffer;
        
        if (dbgFunction)
            cout << "Built training database!" << endl;
    }
    
    ~database() {
        cout << "! Deleting Training Database from RAM !" << endl;;
        for (dataImage *x : *dataset) {
            delete x;
        }
        delete dataset;
    }
    
};

float sigmoid(float num) {
    bool useTrueSigmoid = true;
    if (useTrueSigmoid) {
        float exp_val = exp(-num);
        return (float) (1 / (float) (1 + exp_val)); // return true sigmoid function.
    }
    else
        return (num / (1 + abs(num))); // This is not the true sigmoid function but we can compute this much faster and it has a very similar mathematical properties.
}

float sigprime(float num) {
    return sigmoid(num) * (1 - sigmoid(num));
}

class neuralLayer {
public:
    vector<neuron> listNeurons;
    
    neuralLayer(vector<neuron> input) : listNeurons(input){}
    
    // Call this function on init of the first hidden layer. If TRUE we also randomize BIAS from -10 to 10.
    void randomizeNeuronWeights(bool andBias) {
        for (int i = 0; i < listNeurons.size(); i++) {
            for (int x = 0; x < listNeurons.at(i).weights.size(); x++) {
                listNeurons.at(i).weights.at(x) = (float)((rand() % 20) - 10) / 10;
            }
            if (andBias)
                listNeurons.at(i).bias = (float)((rand() % 20) - 10);
        }
    }
};

//Forward propagation. Multiplies the VAL(Activation) against the weights of the next layer using matrices. Not working properly.
/*void multLayers(neuralLayer* valLayer, neuralLayer *weightLayer) {
 Matrix A, B, C;
 
 A.height = weightLayer->listNeurons.size();
 A.width = valLayer->listNeurons.size();
 A.elements = (float*)malloc(A.width * A.height * sizeof(float));
 B.height = valLayer->listNeurons.size();
 B.width = 1;
 B.elements = (float*)malloc(B.width * B.height * sizeof(float));
 C.height = weightLayer->listNeurons.size();
 C.width = 1;
 C.elements = (float*)malloc(C.width * C.height * sizeof(float));
 
 for (int i = 0; i < A.height; i++)
 for (int j = 0; j < A.width; j++)
 A.elements[i*A.width + j] = weightLayer->listNeurons.at(i).weights.at(j);
 for (int i = 0; i < B.height; i++)
 for (int j = 0; j < B.width; j++)
 B.elements[i*B.width + j] = valLayer->listNeurons.at(i).val;
 
 MatMul(A, B, C);
 
 for (int i = 0; i < C.height; i++) {
 weightLayer->listNeurons.at(i).val = sigmoid(C.elements[i*C.width] + weightLayer->listNeurons.at(i).bias);
 }
 }*/

// Forward propagate using FOR loops (single threaded CPU work)
void multLoopLayers(neuralLayer* valLayer, neuralLayer *weightLayer) {
    
    
    for (int i = 0; i < weightLayer->listNeurons.size(); i++) {
        float sum = 0;
        for (int j = 0; j < valLayer->listNeurons.size(); j++) {
            float w, a;
            a = valLayer->listNeurons.at(j).val;
            w = weightLayer->listNeurons.at(i).weights.at(j);
            
            sum += w * a;
            
        }
        
        weightLayer->listNeurons.at(i).val = sigmoid(sum + weightLayer->listNeurons.at(i).bias);
        
    }
}

// This is where cost is calculated. 1/2 sumation(y-a)^2
float calculateCost(vector<neuron>* outputLayer, unsigned char* label) {
    float difference = 0;
    for (int i = 0; i < outputLayer->size(); i++) {
        // When calculating cost, you are taking the diference of what you got with regards to what you want; squaring the value and returning half the resultant.
        if (i == *label) {
            difference += pow(outputLayer->at(i).val - 1, 2);
        }
        else {
            difference += pow(outputLayer->at(i).val, 2);
        }
    }
    return difference / 2;
}


long GetFileSize(std::string filename)
{
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

//Load weights from file. They are saved when the program is done with its workload.
//Layers are saved to their respective files.
void loadWeights(neuralLayer *A, neuralLayer *B) {
    ifstream myLayer1file ("neuralNetValuesLayer1.txt");
    if (myLayer1file.is_open())
    {
        
        myLayer1file.seekg(0, std::ios::beg);
        
        //vector<unsigned char> *data = new std::vector<unsigned char>(size);
        
        char temp[10240 / 2];
        
        myLayer1file.getline(temp, 10);
        
        //int layer1Size = atoi(temp);
        
        std::streamsize size = GetFileSize("neuralNetValuesLayer1.txt") - myLayer1file.tellg();
        
        vector<unsigned char> *fileBuf = new std::vector<unsigned char>(size);
        if (myLayer1file.read((char*)fileBuf->data(), size))
        {
        }
        char delimiter = '@'; //delinated neurons with an @ symbol in the save file.
        int index = 0;
        
        std::string s(fileBuf->begin(), fileBuf->end());
        
        size_t pos = 0;
        std::string token;
        while ((pos = s.find(delimiter)) != std::string::npos) {
            token = s.substr(0, pos);
            
            string l = string(token);
            size_t pos2 = 0;
            std::string toke;
            int k = 0;
            while ((pos2 = l.find('|')) != std::string::npos) { //delinated weights with an @ symbol in the save file.
                toke = l.substr(0, pos2);
                float o = atof(toke.c_str());
                //cout << l << std::endl;
                A->listNeurons.at(index).weights.at(k) = o;
                l.erase(0, pos2 + 1);
                k++;
            }
            s.erase(0, pos + 1);
            index++;
        }
        
        myLayer1file.close();
        
    } else cout << "Unable to open file";
    
    ifstream myLayer2file("neuralNetValuesLayer2.txt");
    if (myLayer2file.is_open())
    {
        
        myLayer2file.seekg(0, std::ios::beg);
        
        //vector<unsigned char> *data = new std::vector<unsigned char>(size);
        
        char temp[10240 / 2];
        
        myLayer2file.getline(temp, 10);
        
        std::streamsize size = GetFileSize("neuralNetValuesLayer2.txt") - myLayer2file.tellg();
        
        vector<unsigned char> *fileBuf = new std::vector<unsigned char>(size);
        if (myLayer2file.read((char*)fileBuf->data(), size))
        {
        }
        char delimiter = '@';
        int index = 0;
        
        std::string s(fileBuf->begin(), fileBuf->end());
        
        size_t pos = 0;
        std::string token;
        while ((pos = s.find(delimiter)) != std::string::npos) {
            token = s.substr(0, pos);
            
            string l = string(token);
            size_t pos2 = 0;
            std::string toke;
            int k = 0;
            while ((pos2 = l.find('|')) != std::string::npos) {
                toke = l.substr(0, pos2);
                float o = atof(toke.c_str());
                //cout << l << std::endl;
                B->listNeurons.at(index).weights.at(k) = o;
                l.erase(0, pos2 + 1);
                k++;
            }
            s.erase(0, pos + 1);
            index++;
        }
        
        myLayer2file.close();
        
    }
    else cout << "Unable to open file";
}

int main() {
    
    // Initialize a database object, this stores a vector of images and labels.
    database x("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
    
    cout << "Images in database: " << x.dataset->size() << endl;
    
    //cout << *nImage << endl;
    
    const int layer0_size = 28 * 28; // Input layer size
    
    const int layer1_size = 500; // Layer 1 size
    
    const int layer2_size = 5; // Layer 2 size
    
    //HIDDEN LAYER 1
    neuralLayer layer1 = neuralLayer(vector<neuron>(layer1_size, neuron(layer0_size)));
    layer1.randomizeNeuronWeights(false);
    
    //OUTPUT LAYER
    neuralLayer layer2 = neuralLayer(vector<neuron>(layer2_size, neuron(layer1_size)));
    //layer2.randomizeNeuronWeights(false);
    
    loadWeights(&layer1, &layer2);
    
    int successfullGuesses = 0;
    int totalGuesses = 0;
    
    ////
    for (int r = 0; r < 1; r++ /*,cout << endl << " - next!" << endl << endl*/) {
        for (int index = 0000; index < 1000; index++) {
            dataImage *nImage = x.dataset->at(index);
            if (*nImage->label >= 5) {
                continue;
            }
            neuralLayer layer0 = neuralLayer(vector<neuron>(28 * 28));
            
            for (int i = 0; i < layer0_size; i++)
                layer0.listNeurons.at(i).val = nImage->pixArray[i];
            
            // Use CPU for forward propagation.
            multLoopLayers(&layer0, &layer1);
            multLoopLayers(&layer1, &layer2);
            
            //Calculate the cost of the OUTPUT layer specifically. Taking the square of the difference of achieved and desired activation values for the output layer.
            float Cost = calculateCost(&layer2.listNeurons, nImage->label);
            
            //Ok = value from layer2
            //Tk is 1 or 0 from test
            //Oj = value from layer1
            //Sig(z) = Ok
            
            for (int i = 0; i < layer2.listNeurons.size(); i++) {
                neuron *layer2Neuron = &layer2.listNeurons.at(i);
                for (int x = 0; x < layer2Neuron->weights.size(); x++) {
                    
                    float delta = 0.0;
                    float Ok = layer2.listNeurons.at(i).val;
                    if (i == *nImage->label) {     /* (NeuronErrorLayer2) * (Derivative Sigmoid of Layer2) * ActivationLayer1(x)   */
                        delta = (Ok - 1) * ((1 - Ok) * Ok)  * layer1.listNeurons.at(x).val; // notice the X variable is iterating through neuron wieghts but being used a neuron counter. Thats because the number of weights for one layer are equal the number neurons in the nieghboring layer.
                        layer2.listNeurons.at(i).dk = (Ok - 1) * ((1 - Ok) * Ok);
                    }
                    else {
                        delta = Ok * ((1 - Ok) * Ok) *  layer1.listNeurons.at(x).val;
                        layer2.listNeurons.at(i).dk = Ok * ((1 - Ok) * Ok);
                    }
                    layer2.listNeurons.at(i).nudges.at(x) = delta; //tell the weights to remember how much each weight needs to change, but don't apply the change.  We still need the original values for back propagating all the way through the hidden layers..
                }
            }
            
            //Oi = Layer0 val
            //Oj = Layer1 Val
            //Dk = layer2.dk
            //Wjk = weight of Layer2 node to the Layer1 node.
            for (int i = 0; i < layer1.listNeurons.size(); i++) {
                for (int j = 0; j < layer1.listNeurons.at(i).weights.size(); j++) {
                    float Oi = layer0.listNeurons.at(j).val;
                    float Oj = layer1.listNeurons.at(i).val;
                    
                    //Propagating the hidden layer weights.
                    float delta = Oi * Oj * (1 - Oj); // Used for chainruling.
                    float prod = 0.0;
                    for (int x = 0; x < layer2.listNeurons.size(); x++) {
                        prod += layer2.listNeurons.at(x).dk * layer2.listNeurons.at(x).weights.at(i);  //Used for chainruling
                    }
                    layer1.listNeurons.at(i).weights.at(j) += 0.5 * -delta * prod; //update the nodes activation value
                }
            }
            
            for (int i = 0; i < layer2.listNeurons.size(); i++) {
                neuron *layer2Neuron = &layer2.listNeurons.at(i);
                for (int x = 0; x < layer2Neuron->weights.size(); x++) {
                    layer2.listNeurons.at(i).weights.at(x) += 0.5 * -layer2.listNeurons.at(i).nudges.at(x); //update the layer2 weights after we updated the layer1 weights in the nested FOR loops above.
                }
            }
            
            //Now we've updated our weights and everything. But let's evaluate how we performed before we learned so it doesn't look like we already knew about this image. So just evaluate the Output layers activation values. We never updated them.
            //So lets assume the character is the output layer neuron that light up the most. Take the max of all the Neurons, the index of the neuron in the vector is aligned with regards to the image label.
            int maxIndex = 0;
            for (int indx = 0; indx < layer2.listNeurons.size(); indx++) {
                if (layer2.listNeurons.at(indx).val > layer2.listNeurons.at(maxIndex).val)
                    maxIndex = indx;
            }
            //Increment Guessses
            totalGuesses++;
            
            //cout.precision(6);
            
            cout << "Index: " << index << "\tLabel: " << (int)nImage->label[0] << "\tCost: " << setw(10) << setprecision(6) << Cost << "\t";
            
            if (maxIndex == *nImage->label) {
                successfullGuesses++;
                
                cout << "Accuracy: ";
                
                cout << fixed << setw(9) << (((float)successfullGuesses / totalGuesses) * 100) << "%\t";
                
                cout << "CORRECT GUESS!" << endl;
            } else {
                cout << "Accuracy: ";
                
                cout << fixed << setw(9) << (((float)successfullGuesses / totalGuesses) * 100) << "%\t";
                
                cout << endl;
            }
            /*for (int i = 0; i < layer2.listNeurons.size(); i++) {
                printf("Output Layer Activation Values: %f\n", layer2.listNeurons.at(i).val);
            }*/
        }
    }
    
    cout << "Done" << endl;
    
    ofstream myLayer1file("neuralNetValuesLayer1.txt");
    if (myLayer1file.is_open())
    {
        myLayer1file << layer1.listNeurons.size() << endl; //Layer1
        for (int i = 0; i < layer1.listNeurons.size(); i++) {
            for (int x = 0; x < layer1.listNeurons.at(i).weights.size(); x++) {
                myLayer1file << std::setprecision(32) << (float)layer1.listNeurons.at(i).weights.at(x) << ((x == layer1.listNeurons.at(i).weights.size() - 1) ? "|" : "|");
            }
            myLayer1file << "@";
        }
        myLayer1file.close();
    }
    else cout << "Unable to open file";
    
    ofstream myLayer2file("neuralNetValuesLayer2.txt");
    if (myLayer2file.is_open())
    {
        myLayer2file << layer2.listNeurons.size() << endl; //Layer2
        for (int i = 0; i < layer2.listNeurons.size(); i++) {
            for (int x = 0; x < layer2.listNeurons.at(i).weights.size(); x++) {
                myLayer2file << std::setprecision(32) << (float)layer2.listNeurons.at(i).weights.at(x) << ((x == layer2.listNeurons.at(i).weights.size() - 1) ? "|" : "|");
            }
            myLayer2file << "@";
        }
        
        myLayer2file.close();
    }
    else cout << "Unable to open file";
    
    return 0;
}
