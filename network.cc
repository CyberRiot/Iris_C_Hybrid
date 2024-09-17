#include "network.hpp"

network::network(std::vector<int> *spec, int input_size, int num_classes, double learning_rate) {
    this->input_size = input_size;
    this->num_classes = num_classes;
    this->learning_rate = learning_rate;
    layers = new std::vector<layer *>();

    std::cout << "Starting layer initialization..." << std::endl;

    int current_input_size = input_size;  // Track the input size for each layer

    // Initialize the ConvLayer first
    int conv_filter_size = 3;  // Use filter size 3 for convolution
    int conv_num_filters = spec->at(0);  // First element in spec is the number of filters
    layers->push_back(new ConvLayer(conv_filter_size, conv_num_filters, learning_rate));
    
    std::cout << "ConvLayer initialized with filter size " << conv_filter_size << " and " << conv_num_filters << " filters." << std::endl;

    // Now, get the output size after convolution and pooling
    ConvLayer* conv_layer = static_cast<ConvLayer*>(layers->back());

    // Run a dummy forward pass to get the size after convolution and pooling
    std::vector<double> dummy_input(input_size, 0.0);  // Create a dummy input with the original input size
    std::vector<double>* conv_output = conv_layer->forward(&dummy_input);
    int pooled_output_size = conv_output->size();  // Get the size after convolution and pooling

    std::cout << "Pooled output size from ConvLayer: " << pooled_output_size << std::endl;

    // Use this pooled output size as the input size for the first RNNLayer
    current_input_size = pooled_output_size;
    for (int i = 1; i < spec->size(); i++) {
        int hidden_size = spec->at(i);
        std::cout << "Initializing RNNLayer with input size " << current_input_size << " and hidden size " << hidden_size << "." << std::endl;
        layers->push_back(new RNNLayer(current_input_size, hidden_size, learning_rate));

        current_input_size = hidden_size;  // Update input size for the next RNNLayer
    }

    // Initialize the final output layer (RNNLayer)
    std::cout << "Initializing final RNNLayer with input size " << current_input_size << " and output size " << num_classes << "." << std::endl;
    layers->push_back(new RNNLayer(current_input_size, num_classes, learning_rate));

    std::cout << "Network initialized with " << layers->size() << " layers." << std::endl;
}

network::~network() {
    for (layer *l : *layers) {
        delete l;
    }
    delete layers;
    close_debug_output();
}

std::vector<double>* network::fprop(data *d) {
    std::vector<double>* input = d->get_feature_vector();
    std::cout << "Entered FPROP Function: " << std::endl;

    if (input == nullptr || input->empty()) {
        std::cerr << "Error: Input feature vector is null or empty before forward pass!" << std::endl;
        exit(1);
    }

    std::cout << "Forward Pass | Initial Feature Vector Size: " << input->size() << std::endl;

    for (int i = 0; i < layers->size(); ++i) {
        layer* current_layer = (*layers)[i];

        if (ConvLayer* conv_layer = dynamic_cast<ConvLayer*>(current_layer)) {
            std::cout << "Passing through ConvLayer with input size: " << input->size() << std::endl;

            input = conv_layer->forward(input);

            std::cout << "Pooling applied, new size after ConvLayer: " << input->size() << std::endl;
        } else {
            std::cout << "Passing through layer " << i << " with input size: " << input->size() << std::endl;
            input = current_layer->forward(input);
        }

        if (input == nullptr || input->empty()) {
            std::cerr << "Error: Input feature vector became empty after forward pass in layer " << i << "!" << std::endl;
            exit(1);
        }

        std::cout << "Output size after layer " << i << ": " << input->size() << std::endl;
    }

    return input;
}

void network::bprop(data *d) {
    std::vector<double> *output = fprop(d);
    std::vector<double>* gradients = new std::vector<double>(output->size(), 0.0);

    std::vector<int> *class_vector = d->get_class_vector();
    for(size_t i = 0; i < output->size(); i++){
        double error = (*output)[i] - (*class_vector)[i];
        (*gradients)[i] = error * transfer_derivative((*output)[i]);
    }

    for(int i = layers->size() - 1; i >= 0; i--){
        layer *current_layer = (*layers)[i];
        std::cout << "Backpropagating through layer " << i << std::endl;  // Debugging
        gradients = current_layer->backward(gradients);
    }

    delete gradients;
}

void network::update_weights(data *d) {
    for(int i = 0; i < layers->size(); i++){
        layer* current_layer = (*layers)[i];

        //Check if current layer is a ConvLayer or RNNLayer
        if(ConvLayer* conv_layer = dynamic_cast<ConvLayer*>(current_layer)){
            //This is a Convolutional Layer, update weights for ConvLayer
            for(int f = 0; f < conv_layer->num_filters; f++){
                for(int j = 0; j < conv_layer->filter_size * conv_layer->filter_size; j++){
                    for(auto &n : (*conv_layer->filters)[f]){
                        //This is the rule for the simple weight update
                        n->weights->at(j) -= learning_rate * n->delta;
                    }
                }
            }
        }
        else if(RNNLayer* rnn_layer = dynamic_cast<RNNLayer*>(current_layer)){
            //Update weights for RNNLayer
            for(int h = 0; h < rnn_layer->hidden_size; h++){
                for(int w = 0; w < rnn_layer->input_size; w++){
                    (*rnn_layer->hidden_neurons)[h]->weights->at(w) -= learning_rate * (*rnn_layer->hidden_neurons)[h]->delta;
                }
            }
        }
    }
}

double network::transfer(double activation) {
    return 1.0 / (1.0 + std::exp(-activation));
}

double network::transfer_derivative(double output) {
    return output * (1 - output);
}

int network::predict(data *d) {
    std::vector<double>* output = fprop(d);
    return std::distance(output->begin(), std::max_element(output->begin(), output->end()));
}

// Training: Pass data through the network, forward pass only for now
void network::train(int epochs, double validation_threshold) {
    if (common_training_data == nullptr || common_training_data->empty()) {
        std::cerr << "Error: Training data is empty!" << std::endl;
        return;
    }

    // Open the debug log file for appending
    std::ofstream debug_log("debug_output.txt", std::ios::app);

    int total_samples = common_training_data->size();

    // Training loop over the number of epochs
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << std::endl;
        debug_log << "Epoch " << epoch + 1 << "/" << epochs << std::endl;

        double total_loss = 0.0;
        int sample_index = 0;

        // Iterate over the training data
        for (data* d : *common_training_data) {
            std::vector<double>* input = d->get_feature_vector();

            if (input == nullptr || input->empty()) {
                std::cerr << "Error: Input feature vector is empty for sample " << sample_index << std::endl;
                debug_log << "Error: Input feature vector is empty for sample " << sample_index << std::endl;
                continue;
            }

            // Forward pass through the network
            std::vector<double>* output = fprop(d);

            // Calculate loss using the correct calculate_loss function
            double sample_loss = calculate_loss(output, d->get_class_vector());
            total_loss += sample_loss;

            // Backward pass and update weights
            bprop(d);
            update_weights(d);

            // Progress bar update
            sample_index++;
            int progress = static_cast<int>(static_cast<double>(sample_index) / total_samples * 100);
            std::cout << "\rProgress: [";
            for (int i = 0; i < 50; ++i) {
                if (i < progress / 2) std::cout << "=";
                else std::cout << " ";
            }
            std::cout << "] " << progress << "%";

            // Log the sample loss to the debug log
            debug_log << "Sample " << sample_index << " | Loss: " << sample_loss << std::endl;
        }

        // Log total loss at the end of the epoch
        debug_log << "Epoch " << epoch + 1 << " | Total Loss: " << total_loss << std::endl;
        std::cout << "\nEpoch " << epoch + 1 << " | Total Loss: " << total_loss << std::endl;

        // Validate after each epoch and check if early stopping should be applied
        double validation_accuracy = validate();
        std::cout << "Validation Accuracy after epoch " << epoch + 1 << ": " << validation_accuracy * 100 << "%" << std::endl;
        debug_log << "Validation Accuracy after epoch " << epoch + 1 << ": " << validation_accuracy * 100 << "%" << std::endl;

        // Early stopping condition
        if (validation_accuracy >= validation_threshold) {
            std::cout << "Stopping early as validation accuracy has reached " << validation_accuracy * 100 << "%" << std::endl;
            debug_log << "Early Stopping: Validation Accuracy = " << validation_accuracy * 100 << "%" << std::endl;
            break;
        }
    }

    // Close the debug log after training
    debug_log.close();
}

void network::set_debug_output(const std::string &filename) {
    debug_output.open(filename);
    if (!debug_output.is_open()) {
        std::cerr << "Error opening debug file: " << filename << std::endl;
    }
}

void network::close_debug_output() {
    if (debug_output.is_open()) {
        debug_output.close();
    }
}

void network::save_model(const std::string &filename) {
    std::ofstream outfile(filename);
    if(!outfile.is_open()){
        std::cerr << "Error: Could not open file" << filename << "for saving model." << std::endl;
        exit(1);
    }

    //Save each layer's weights
    for(layer* l : *layers){
        ConvLayer* conv_layer = dynamic_cast<ConvLayer*>(l);
        RNNLayer* rnn_layer = dynamic_cast<RNNLayer*>(l);
        if(conv_layer != nullptr){
            for(auto& filter : *conv_layer->filters){
                for(neuron* n : filter){
                    for(double weight : *n->weights){
                        outfile << weight << " ";
                    }
                    outfile << std::endl;
                }
            }
        }
        if(rnn_layer != nullptr){
            for(neuron* n : *rnn_layer->hidden_neurons){
                for(double weight : *n->weights){
                    outfile << weight << " ";
                }
                outfile << std::endl;
            }
        }
    }
    outfile.close();
    std::cout << "Model saved to " << filename << std::endl;
}

void network::load_model(const std::string &filename) {
    std::ifstream infile(filename);
    if(!infile.is_open()){
        std::cerr << "Error: could not open file " << filename << " for loading model." << std::endl;
        exit(1);
    }
    for(layer* l : *layers){
        ConvLayer* conv_layer = dynamic_cast<ConvLayer*>(l);
        RNNLayer* rnn_layer = dynamic_cast<RNNLayer*>(l);

        if(conv_layer != nullptr){
            for(auto& filter : *conv_layer->filters){
                for(neuron* n : filter){
                    for(double& weight : *n->weights){
                        infile >> weight;
                    }
                }
            }
        }
        if(rnn_layer != nullptr){
            for(neuron* n : *rnn_layer->hidden_neurons){
                for(double& weight : *n->weights){
                    infile >> weight;
                }
            }
        }
    }
    infile.close();
    std::cout << "Model loaded from " << filename << std::endl;
}

double network::validate() {
    if (common_validation_data == nullptr || common_validation_data->empty()) {
        std::cerr << "Error: Validation data is empty!" << std::endl;
        return 0.0;
    }

    int correct = 0;
    int total = 0;

    // Iterate over the validation data in common_validation_data
    for (data* d : *common_validation_data) {
        std::vector<double>* input = d->get_feature_vector();

        if (input == nullptr || input->empty()) {
            std::cerr << "Error: Input feature vector is empty!" << std::endl;
            continue;
        }

        // Forward pass through the network
        std::vector<double>* output = fprop(d);

        // Get the predicted class
        int predicted_class = std::distance(output->begin(), std::max_element(output->begin(), output->end()));

        if (d->get_class_vector()->at(predicted_class) == 1) {
            correct++;
        }
        total++;
    }

    return static_cast<double>(correct) / total;
}

double network::test() {
    if (common_testing_data == nullptr || common_testing_data->empty()) {
        std::cerr << "Error: Testing data is empty!" << std::endl;
        return 0.0;
    }

    int correct = 0;
    int total = 0;

    // Iterate over the testing data in common_testing_data
    for (data* d : *common_testing_data) {
        std::vector<double>* input = d->get_feature_vector();
        
        if (input == nullptr || input->empty()) {
            std::cerr << "Error: Input feature vector is empty!" << std::endl;
            continue;
        }

        // Forward pass through the network
        std::vector<double>* output = fprop(d);

        // Get the predicted class
        int predicted_class = std::distance(output->begin(), std::max_element(output->begin(), output->end()));

        if (d->get_class_vector()->at(predicted_class) == 1) {
            correct++;
        }
        total++;
    }

    return static_cast<double>(correct) / total;
}

void network::output_predictions(const std::string &filename, data_handler *dh) {
    // Output predictions to CSV or other file formats
}

double network::calculate_loss(std::vector<double>* output, std::vector<int>* class_vector) {
    double loss = 0.0;
    for (size_t i = 0; i < class_vector->size(); i++) {
        // Assuming binary cross-entropy loss for this example
        int target = (*class_vector)[i];  // Dereference to access the value
        double prediction = (*output)[i];
        
        // Binary cross-entropy loss calculation
        loss += -target * std::log(prediction) - (1 - target) * std::log(1 - prediction);
    }
    return loss;
}

// In Main
int main() {
    // Initialize the data handler
    data_handler *dh = new data_handler();
    dh->read_data_and_labels("F:\\Code\\VOOD\\data\\binary\\Polyphia.data", "F:\\Code\\VOOD\\data\\labels\\VOOD_labels.csv");
    dh->split_data();

    // Initialize the network with the inherited common_data's training set
    std::vector<int> *spec = new std::vector<int>{128, 64, 3};
    network *net = new network(spec, dh->get_training_data()->at(0)->get_feature_vector()->size(), 3, 0.01);

    // Set debug output file
    net->set_debug_output("debug_output.txt");

    // Assign the split data to the network's inherited common_data members
    net->set_common_training_data(dh->get_training_data());
    net->set_common_testing_data(dh->get_testing_data());
    net->set_common_validation_data(dh->get_validation_data());

    // Train the network
    std::cout << "Starting training..." << std::endl;
    net->train(10, 0.98);  // Train for 10 epochs

    // Save the trained model
    net->save_model("F:\\Code\\VOOD\\data\\saved_model\\trained_model.bin");
    std::cout << "Model saved successfully." << std::endl;

    // Test the model on test data
    double test_accuracy = net->test();
    std::cout << "Test Accuracy: " << test_accuracy << std::endl;

    // Cleanup
    delete dh;
    delete net;
    delete spec;

    return 0;
}
