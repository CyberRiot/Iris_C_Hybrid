#include "../include/layer.hpp"

//Implementation of the ConvLayer (Convolutional Layer (for the CNN))

//Constructor: initializing filters with random values
ConvLayer::ConvLayer(int filter_size, int num_filters, double learning_rate) {
    this->filter_size = filter_size;
    this->num_filters = num_filters;
    this->learning_rate = learning_rate;  // Initialize the learning rate
    this->filters = new std::vector<std::vector<neuron*>>(num_filters);

    std::cout << "Initializing Convolutional Layers" << std::endl;

    for (int i = 0; i < num_filters; i++) {
        (*filters)[i] = std::vector<neuron*>(filter_size * filter_size);
        for (int j = 0; j < filter_size * filter_size; j++) {
            (*filters)[i][j] = new neuron(filter_size * filter_size);  // Each neuron gets filter_size * filter_size weights
            if ((*filters)[i][j]->weights->size() != filter_size * filter_size) {
                std::cerr << "Error: Neuron at filter " << i << ", index " << j 
                          << " initialized with incorrect weight size: " 
                          << (*filters)[i][j]->weights->size() << std::endl;
                exit(1);
            }
        }
    }

    std::cout << "ConvLayer initialized with " << num_filters << " filters, each with " 
              << filter_size * filter_size << " neurons." << std::endl;
}

ConvLayer::~ConvLayer() {
    for (auto& filter : *filters) {
        for (neuron* n : filter) {
            delete n;
        }
    }
    delete filters;
}

// Forward Pass: Apply convolution on the input
std::vector<double>* ConvLayer::forward(std::vector<double>* input) {
    std::cout << "Entered forward function" << std::endl;

    // Log raw data for inspection
    std::cout << "Raw input data (first 10 values): ";
    for (int i = 0; i < 10 && i < input->size(); i++) {
        std::cout << (*input)[i] << " ";
    }
    std::cout << std::endl;
    if (input == nullptr || input->empty()) {
        std::cerr << "Error: Input feature vector is empty in ConvLayer at the start!" << std::endl;
        exit(1);
    }

    // Store the input for the backward pass
    this->input = input;

    // Assuming the input size should form a square image
    int image_size = static_cast<int>(std::sqrt(input->size()));
    std::cout << "Reshaping input vector of size " << input->size() << " to 2D matrix of size: " << image_size << "x" << image_size << std::endl;

    if (image_size <= 0) {
        std::cerr << "Error: Invalid input size, unable to reshape input vector." << std::endl;
        exit(1);
    }

    auto reshaped_input = new std::vector<std::vector<double>>(image_size, std::vector<double>(image_size));

    int total_elements = image_size * image_size;
    int progress = 0;

    // Fill reshaped input with a progress bar
    for (int i = 0; i < image_size; i++) {
        for (int j = 0; j < image_size; j++) {
            (*reshaped_input)[i][j] = (*input)[i * image_size + j];

            // Update progress
            progress++;
            if (progress % (total_elements / 100) == 0) {  // Update progress bar at 1% increments
                int progress_percentage = static_cast<int>((static_cast<double>(progress) / total_elements) * 100);
                std::cout << "\rReshaping Progress: [";
                for (int p = 0; p < 50; ++p) {
                    if (p < progress_percentage / 2) std::cout << "=";
                    else std::cout << " ";
                }
                std::cout << "] " << progress_percentage << "%" << std::flush;
            }
        }
    }

    std::cout << "\nReshaping completed!" << std::endl;

    // Convolve and return
    auto output = new std::vector<double>;
    for (int i = 0; i < num_filters; i++) {
        std::vector<std::vector<double>>* filter_output = this->convolve(reshaped_input, &((*filters)[i]));
        for (auto row : *filter_output) {
            output->insert(output->end(), row.begin(), row.end());
        }
        delete filter_output;
    }

    delete reshaped_input;

    // Apply average pooling to the output (reduce size by a factor of 2)
    output = average_pooling(output, 2);  
    std::cout << "First pooling applied. Output size after pooling: " << output->size() << std::endl;

    // Apply additional pooling (reduce size by another factor of 2)
    output = average_pooling(output, 2);  
    std::cout << "Second pooling applied. Output size after second pooling: " << output->size() << std::endl;

    // Now flatten the output before passing to the next layer
    std::cout << "Flattening output for RNN layer..." << std::endl;
    auto flattened_output = new std::vector<double>;
    flattened_output->reserve(output->size());

    for (const double val : *output) {
        flattened_output->push_back(val);
    }

    delete output;
    std::cout << "Flattening completed! Flattened size: " << flattened_output->size() << std::endl;

    return flattened_output;
}

std::vector<double> *ConvLayer::backward(std::vector<double> *gradients) {
    // Assuming d_out is the gradient of the loss w.r.t the output of this layer
    std::vector<double> *d_input = new std::vector<double>(input->size());  // Gradient w.r.t input
    std::vector<std::vector<double>> d_filters(num_filters, std::vector<double>(filter_size * filter_size)); // Gradient w.r.t filters

    // Loop over the filters and inputs to compute gradients
    for (int f = 0; f < num_filters; ++f) {
        for (int i = 0; i < input->size(); ++i) {
            for (int j = 0; j < filter_size * filter_size; ++j) {
                d_filters[f][j] += (*input)[i] * (*gradients)[i];  // Adjust the filters using chain rule
                (*d_input)[i] += (*filters)[f][j]->weights->at(j) * (*gradients)[i];  // Propagate gradient back to input
            }
        }
    }

    // Update the filters using gradients (gradient descent)
    for (int f = 0; f < num_filters; ++f) {
        for (int j = 0; j < filter_size * filter_size; ++j) {
            for (auto &n : (*filters)[f]) {
                n->weights->at(j) -= learning_rate * d_filters[f][j];  // Update weights using the learning rate
            }
        }
    }

    return d_input;  // Return the gradient w.r.t input
}

std::vector<std::vector<double>>* ConvLayer::convolve(std::vector<std::vector<double>>* input, std::vector<neuron*>* filter) {
    int input_size = input->size();  // Input size should match the reshaped 2D matrix
    int output_size = input_size - filter_size + 1;  // Calculate output size

    // Debugging logs for input and filter sizes
    std::cout << "Starting convolution process..." << std::endl;
    std::cout << "Input size: " << input_size << ", filter size: " << filter_size << ", output size: " << output_size << std::endl;
    std::cout << "Filter neuron count: " << filter->size() << std::endl;

    auto result = new std::vector<std::vector<double>>(output_size, std::vector<double>(output_size));

    int total_operations = output_size * output_size * filter_size * filter_size;
    int progress = 0;

    // Perform 2D convolution
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < output_size; j++) {
            double convolved_value = 0;
            int filter_idx = 0;
            for (int fi = 0; fi < filter_size; fi++) {
                for (int fj = 0; fj < filter_size; fj++) {
                    if (filter_idx >= filter->size()) {
                        std::cerr << "Error: Filter neuron out of bounds at index " << filter_idx << std::endl;
                        exit(1);
                    }

                    neuron* n = (*filter)[filter_idx++];
                    if (!n) {
                        std::cerr << "Error: Null neuron at filter index " << filter_idx - 1 << std::endl;
                        exit(1);
                    }

                    if (n->weights->size() != filter_size * filter_size) {
                        std::cerr << "Error: Neuron at filter index " << filter_idx - 1 
                                  << " has incorrect weight size: " << n->weights->size() << std::endl;
                        exit(1);
                    }

                    double input_value = (*input)[i + fi][j + fj];
                    double weight_value = n->weights->at(fi * filter_size + fj);

                    convolved_value += input_value * weight_value;

                    progress++;
                    if (progress % (total_operations / 100) == 0) {
                        int progress_percentage = static_cast<int>((static_cast<double>(progress) / total_operations) * 100);
                        std::cout << "\rConvolution Progress: [";
                        for (int p = 0; p < 50; ++p) {
                            if (p < progress_percentage / 2) std::cout << "=";
                            else std::cout << " ";
                        }
                        std::cout << "] " << progress_percentage << "%" << std::flush;
                    }
                }
            }
            (*result)[i][j] = convolved_value;
        }
    }

    std::cout << "\nConvolution process completed!" << std::endl;
    return result;
}

std::vector<double>* ConvLayer::average_pooling(std::vector<double>* input, int pooling_size) {
    int input_size = static_cast<int>(std::sqrt(input->size()));
    int output_size = input_size / pooling_size;
    auto output = new std::vector<double>(output_size * output_size, 0.0);

    std::cout << "Pooling input size: " << input->size() << " | Pooling output size: " << output->size() << std::endl;

    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < output_size; j++) {
            double sum = 0.0;
            for (int x = 0; x < pooling_size; x++) {
                for (int y = 0; y < pooling_size; y++) {
                    sum += (*input)[(i * pooling_size + x) * input_size + (j * pooling_size + y)];
                }
            }
            (*output)[i * output_size + j] = sum / (pooling_size * pooling_size);
        }
    }

    // Log pooled data for inspection
    std::cout << "Pooled data (first 10 values): ";
    for (int i = 0; i < 10 && i < output->size(); i++) {
        std::cout << (*output)[i] << " ";
    }
    std::cout << std::endl;

    return output;
}

int ConvLayer::get_pooled_output_size() const {
    return this->input->size();  // Return the size after pooling
}

// RNNLayer (LSTM/GRU) Implementation

//Initialize the hidden and cell states for usage
RNNLayer::RNNLayer(int input_size, int hidden_size, double learning_rate) {
    this->input_size = input_size;
    this->hidden_size = hidden_size;
    this->learning_rate = learning_rate;  // Initialize the learning rate

    // Initialize hidden and cell states to zero vectors
    hidden_state = new std::vector<double>(hidden_size, 0.0);
    cell_state = new std::vector<double>(hidden_size, 0.0);

    // Initialize neurons
    hidden_neurons = new std::vector<neuron*>(hidden_size);
    
    std::cout << "Initializing RNN Layer with input size: " << input_size 
              << ", hidden size: " << hidden_size << std::endl;

    int total_neurons = hidden_size;
    int progress = 0;

    // Loop to initialize each hidden neuron
    for (int i = 0; i < hidden_size; i++) {
        (*hidden_neurons)[i] = new neuron(input_size);  // Initialize each hidden neuron
        std::cout << "Neuron " << i << " initialized with input size: " << input_size << std::endl;

        // Progress bar update
        progress++;
        if (progress % (total_neurons / 100) == 0) {  // Update progress bar at 1% increments
            int progress_percentage = static_cast<int>((static_cast<double>(progress) / total_neurons) * 100);
            std::cout << "\rRNN Layer Initialization Progress: [";
            for (int p = 0; p < 50; ++p) {
                if (p < progress_percentage / 2) std::cout << "=";
                else std::cout << " ";
            }
            std::cout << "] " << progress_percentage << "%" << std::flush;
        }
    }

    std::cout << "\nRNN Layer initialization completed!" << std::endl;
}


RNNLayer::~RNNLayer() {
    delete hidden_state;
    delete cell_state;
    for (neuron* n : *hidden_neurons) {
        delete n;
    }
    delete hidden_neurons;
}

std::vector<double>* RNNLayer::forward(std::vector<double>* input) {
    auto output = new std::vector<double>(hidden_size);
    
    std::cout << "RNNLayer forward pass on input size: " << input->size() << std::endl;

    // Progress bar variables
    int progress = 0;
    int total_elements = hidden_size;
    
    for (int i = 0; i < hidden_size; i++) {
        // Forward pass through each hidden neuron
        (*output)[i] = (*hidden_neurons)[i]->activate(input);

        // Update progress bar
        progress++;
        if (progress % (total_elements / 100) == 0) {  // Update every 1%
            int progress_percentage = static_cast<int>((static_cast<double>(progress) / total_elements) * 100);
            std::cout << "\rRNN Forward Progress: [";
            for (int p = 0; p < 50; ++p) {
                if (p < progress_percentage / 2) std::cout << "=";
                else std::cout << " ";
            }
            std::cout << "] " << progress_percentage << "%" << std::flush;
        }
    }

    std::cout << "\nRNN forward pass completed! Output size: " << output->size() << std::endl;

    return output;
}

std::vector<double>* RNNLayer::backward(std::vector<double>* gradients) {
    // Initialize the gradients w.r.t the input and hidden states
    std::vector<double>* d_input = new std::vector<double>(input_size, 0.0);
    std::vector<double>* d_hidden_state = new std::vector<double>(hidden_size, 0.0);
    std::vector<double>* d_cell_state = new std::vector<double>(hidden_size, 0.0);

    // Loop backward through time (if backpropagation through time is needed)
    for (int i = 0; i < hidden_size; ++i) {
        // Calculate the gradients for the cell and hidden states
        double d_output = (*gradients)[i];
        double d_output_gate = d_output * std::tanh((*cell_state)[i]);
        double d_cell = d_output * (*hidden_state)[i];

        // Gradients for the input and forget gates
        double d_input_gate = d_cell * (*cell_state)[i];
        double d_forget_gate = d_cell * (*cell_state)[i];

        // Update the hidden neurons using the calculated gradients
        (*hidden_neurons)[i]->weights->at(i) -= learning_rate * d_input_gate;  // Example weight update

        // Accumulate gradients for the next step
        (*d_hidden_state)[i] = d_output_gate * (*hidden_state)[i];
        (*d_cell_state)[i] = d_cell;
    }

    // Return the gradient w.r.t the input to propagate to the previous layer
    return d_input;
}

std::vector<double>* RNNLayer::lstm_forward(std::vector<double>* input) {
    std::cout << "Entered LSTM_Forward" << std::endl;
    auto forget_gate = new std::vector<double>(hidden_size);
    auto input_gate = new std::vector<double>(hidden_size);
    auto output_gate = new std::vector<double>(hidden_size);
    auto cell_candidate = new std::vector<double>(hidden_size);

    // Output for the current time step
    auto output = new std::vector<double>(hidden_size);

    // Print input and hidden neuron sizes for debugging
    std::cout << "Input size: " << input->size() 
              << ", Hidden size: " << hidden_size 
              << ", Hidden neurons size: " << hidden_neurons->size() 
              << std::endl;

    // Ensure that the sizes match the expected hidden size
    std::cout << "hidden_state size: " << hidden_state->size() << std::endl;
    std::cout << "cell_state size: " << cell_state->size() << std::endl;

    if (hidden_neurons->size() != hidden_size) {
        std::cerr << "Error: Hidden neurons size mismatch. Expected: " << hidden_size 
                  << ", Got: " << hidden_neurons->size() << std::endl;
        exit(1);
    }

    if (hidden_state->size() != hidden_size) {
        std::cerr << "Error: Hidden state size mismatch. Expected: " << hidden_size 
                  << ", Got: " << hidden_state->size() << std::endl;
        exit(1);
    }

    if (cell_state->size() != hidden_size) {
        std::cerr << "Error: Cell state size mismatch. Expected: " << hidden_size 
                  << ", Got: " << cell_state->size() << std::endl;
        exit(1);
    }

    // Loop through the hidden size
    for (int i = 0; i < hidden_size; i++) {
        std::cout << "Processing index: " << i << std::endl;

        if (i >= hidden_neurons->size()) {
            std::cerr << "Error: Hidden neurons out of bounds at index " << i 
                      << " for size " << hidden_neurons->size() << std::endl;
            exit(1);
        }

        if (i >= hidden_state->size()) {
            std::cerr << "Error: Hidden state out of bounds at index " << i 
                      << " for size " << hidden_state->size() << std::endl;
            exit(1);
        }

        if (i >= cell_state->size()) {
            std::cerr << "Error: Cell state out of bounds at index " << i 
                      << " for size " << cell_state->size() << std::endl;
            exit(1);
        }

        int input_index = i % input->size();
        if (input_index >= input->size()) {
            std::cerr << "Error: Input index out of bounds: " << input_index << std::endl;
            exit(1);
        }

        // Get the activation from the corresponding hidden neuron
        neuron* n = (*hidden_neurons)[i];
        double neuron_output = n->activate(input);  // Pass input to the neuron for activation

        // Calculating the gates for LSTM using the neuron output and hidden state
        (*forget_gate)[i] = sigmoid(neuron_output + (*hidden_state)[i]);  // Forget gate
        (*input_gate)[i] = sigmoid(neuron_output + (*hidden_state)[i]);   // Input gate
        (*output_gate)[i] = sigmoid(neuron_output + (*hidden_state)[i]);  // Output gate
        (*cell_candidate)[i] = std::tanh(neuron_output + (*hidden_state)[i]);  // Candidate memory

        // Updating cell state
        (*cell_state)[i] = (*forget_gate)[i] * (*cell_state)[i] + (*input_gate)[i] * (*cell_candidate)[i];

        // Updating hidden state (output of this time step)
        (*hidden_state)[i] = (*output_gate)[i] * std::tanh((*cell_state)[i]);

        // Set output for the current time step
        (*output)[i] = (*hidden_state)[i];
    }

    // Clean up
    delete forget_gate;
    delete input_gate;
    delete output_gate;
    delete cell_candidate;

    return output;
}

std::vector<double>* RNNLayer::lstm_backward(std::vector<double>* d_output, std::vector<double>* d_next_cell_state) {
    std::vector<double>* d_input = new std::vector<double>(input_size, 0.0);  // Gradients w.r.t input
    std::vector<double>* d_hidden_state = new std::vector<double>(hidden_size, 0.0);  // Gradients w.r.t hidden state
    std::vector<double>* d_cell_state = new std::vector<double>(hidden_size, 0.0);  // Gradients w.r.t cell state

    std::vector<double> d_forget_gate(hidden_size, 0.0);
    std::vector<double> d_input_gate(hidden_size, 0.0);
    std::vector<double> d_output_gate(hidden_size, 0.0);
    std::vector<double> d_cell_candidate(hidden_size, 0.0);

    // Loop through each neuron in the hidden layer to compute gradients
    for (int i = 0; i < hidden_size; ++i) {
        neuron* current_neuron = (*hidden_neurons)[i];  // Access the current hidden neuron
        if (!current_neuron) {
            std::cerr << "Error: Null neuron at index " << i << std::endl;
            exit(1);
        }

        // Get the gradient for the output gate and backpropagate through it
        d_output_gate[i] = (*d_output)[i] * std::tanh((*cell_state)[i]);  // Gradient of the output gate
        double d_tanh_cell_state = (*d_output)[i] * (*hidden_state)[i];  // Gradient of tanh(cell state)

        // Compute gradients for forget gate, input gate, and candidate cell state
        d_forget_gate[i] = d_tanh_cell_state * (*cell_state)[i];
        d_input_gate[i] = d_tanh_cell_state * (*cell_state)[i];
        d_cell_candidate[i] = d_tanh_cell_state * (1 - std::pow(std::tanh((*cell_state)[i]), 2));  // Derivative of tanh

        // Accumulate gradients for the next time step
        (*d_cell_state)[i] = (*d_next_cell_state)[i] * d_forget_gate[i];  // Accumulating cell state gradients

        // Update weights for neurons based on the calculated gradients
        for (size_t j = 0; j < current_neuron->weights->size(); ++j) {
            current_neuron->weights->at(j) -= learning_rate * d_forget_gate[i];  // Update weights for forget gate
            current_neuron->weights->at(j) -= learning_rate * d_input_gate[i];   // Update weights for input gate
            current_neuron->weights->at(j) -= learning_rate * d_output_gate[i];  // Update weights for output gate
        }

        // Propagate the gradient back to the input for this time step
        for (size_t k = 0; k < current_neuron->weights->size(); ++k) {
            (*d_input)[k] += current_neuron->weights->at(k) * d_output_gate[i];
        }
    }

    return d_input;  // Return the input gradients for the previous layer
}

double RNNLayer::sigmoid(double x){
    return 1.0 / (1.0 + std::exp(-x));
}