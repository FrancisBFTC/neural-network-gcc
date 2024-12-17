#include "src/neuralnet.h"
#include "src/tensor.h"
#include "src/files.h"

#define INPUTS 256
#define HIDDEN1 128
#define HIDDEN2 64
#define OUTPUTS 2

typedef struct {
	int layers = 0;
	int input = 0;
	int hidden[2];
	int output = 0;
	double*** weights;
	double** bias;
    char** labels;
} NeuralNetworkData;

struct Lucy {
	
	Neural* network;
	NeuralNetworkData data;
	int networkAmmt = 0;
	int pos = 1;
	bool isConfigured = false;
	int** camadas;
	
	void create_network(int dataset_count){
		network = (Neural*) malloc(dataset_count * sizeof(Neural));
		camadas = (int**) malloc(dataset_count * sizeof(int *));
		networkAmmt = dataset_count;
	}
	
	void initialize_network(int model, int epocas, double learning_rate){	
		network[model].epocas = (epocas != 0) ? epocas : 10000;		
		network[model].learning_rate = (learning_rate != 0) ? learning_rate : 0.5; 
		if(isConfigured)
			network[model].iniciar(camadas[model], network[model].layers_ammount);
	}
	
	void build_layer(int model, int layersize){
		network[model].layers_ammount = (layersize != 0 && layersize >= 3) ? layersize : 4;
		camadas[model] = (int*) malloc(layersize * sizeof(int));
		isConfigured = true;
	}
	
	void build_input(int model, int inputsize){
		camadas[model][0] = inputsize;
		isConfigured = true;
	}
	
	void build_hidden(int model, int hiddenlayer){
		camadas[model][pos++] = hiddenlayer;
		if(pos == network[model].layers_ammount - 1)
			pos = 1;
		isConfigured = true;
	}
	
	void build_output(int model, int outputsize){
		network[model].classes = outputsize;
		camadas[model][network[model].layers_ammount - 1] = outputsize;
		isConfigured = true;
	}
	
	void rename(int model, char* name){
		network[model].name = name;
	}
	
	void show_network(int model){
		printf("Rede Neural [%s] -> Numero de camadas: %d \n", network[model].name, network[model].layers_ammount);
		for(int i = 0; i < network[model].quantCamadas; i++){
			printf("\tCamada %d -> Numero de neuronios: %d \n", i+1, network[model].camadas[i].quantNeuronios);
			for(int j = 0; j < network[model].camadas[i].quantNeuronios; j++){
				printf("\t\tNeuronio %d -> Numero de entradas: %d (camada %d) \n", j, network[model].camadas[i].neuronio[j].quantEntradas, i);
				for(int w = 0; w < network[model].camadas[i].neuronio[j].quantEntradas; w++){
					printf("\t\t\tEntrada %d -> Peso: %f \n", w, network[model].camadas[i].neuronio[j].pesos[w]);
				}
			}
		}
	}
	
	double** create_input(int classes, int *cols, int index, char** file[], char* path[]){
		int length; 
		stringstream pathfile;
		double** tensor = (double**) malloc(classes * sizeof(double *));
		for(int x = 0; x < classes; x++){
			pathfile << path[x] << "/" << file[x][index];
			tensor[x] = create_tensor(pathfile.str().c_str(), &length, true);
			pathfile.str("");
		}
		*cols = length;
		return tensor;
	}

	double** create_label(int rows, int cols){
		double** rotulos = (double**)malloc(rows * sizeof(double*));
	    for (int i = 0; i < rows; i++)
	    	rotulos[i] = (double*)malloc(cols * sizeof(double));
	    return rotulos;
	}
	
	void save_training(int model, const char *filename) {
	    FILE *file = fopen(filename, "wb");
	    if (!file) {
	        perror("Erro ao abrir o arquivo");
	        return;
	    }
	
		int index = network[model].quantCamadas;
		data.layers = network[model].quantCamadas + 1;
		data.input = network[model].camadas[0].neuronio[0].quantEntradas;
		for(int j = 0; j < data.layers - 2; j++)
			data.hidden[j] = network[model].camadas[j].quantNeuronios;
		data.output = network[model].camadas[index - 1].quantNeuronios;
		
		// Escreve as configurações neurais
		fwrite(&data.layers, sizeof(int), 1, file);
		fwrite(&data.input, sizeof(int), 1, file);
		fwrite(&data.hidden, sizeof(int), 2, file);
		fwrite(&data.output, sizeof(int), 1, file);
		
		data.weights = (double***) malloc((data.layers - 1) * sizeof(double**));
		data.bias = (double**) malloc((data.layers - 1) * sizeof(double*));
		for(int i = 0; i < network[model].quantCamadas; i++){
			
			int neuron_length = network[model].camadas[i].quantNeuronios;
	    	int input_length = network[model].camadas[i].neuronio[0].quantEntradas;
			data.weights[i] = (double**) malloc(neuron_length * sizeof(double*));
			data.bias[i] = (double*) malloc(neuron_length * sizeof(double));
	    	
			for(int j = 0; j < neuron_length; j++){
	    		
				data.weights[i][j] = (double*) malloc(input_length * sizeof(double));

				for(int k = 0; k < input_length; k++){
					data.weights[i][j][k] = network[model].camadas[i].neuronio[j].pesos[k];
					fwrite(&data.weights[i][j][k], sizeof(double), 1, file);	// input_length * neuron_length
				}
				data.bias[i][j] = network[model].camadas[i].neuronio[j].bias;
				fwrite(&data.bias[i][j], sizeof(double), 1, file);	// neuron_length
				free(data.weights[i][j]);
			}
			free(data.bias[i]);
			free(data.weights[i]);
		}
		free(data.weights);
		free(data.bias);
	    	
	    data.labels = (char**) malloc(data.output * sizeof(char *));
		for(int i = 0; i < data.output; i++){
			data.labels[i] = (char*) malloc(50 * sizeof(char));
			strcpy(data.labels[i], network[model].labels[i]);
			fwrite(data.labels[i], sizeof(char), 50, file);
			free(data.labels[i]);
		}
		free(data.labels);
	
	    fclose(file);
	    printf("Dados do modelo '%s' foram salvos!\n", filename);
	}
	
	void load_training(int model, const char *filename) {
	    FILE *file = fopen(filename, "rb");
	    if (!file) {
	        perror("Erro ao abrir o arquivo");
	        return;
	    }
		
		fread(&data.layers, sizeof(int), 1, file);
		fread(&data.input, sizeof(int), 1, file);
		fread(&data.hidden, sizeof(int), 2, file);
		fread(&data.output, sizeof(int), 1, file);
		
		if(!isConfigured){
			network[model].name = (char*)filename;
			
			build_layer(model, data.layers);					// Cria 4 camadas no modelo 0
			build_input(model, data.input); 					// Cria 256 entradas iniciais no modelo 0
			for(int j = 0; j < data.layers - 2; j++)
				build_hidden(model, data.hidden[j]);			// Cria 128 e 64 neurônios na 1ª e 2ª camada oculta, respectivamente
			build_output(model, data.output);	 				// Cria 2 neurônios de saída na camada de saída (classes)
			initialize_network(model, 0, 0);					// Inicialize a rede
		}
		
		data.weights = (double***) malloc((data.layers - 1) * sizeof(double**));
		data.bias = (double**) malloc((data.layers - 1) * sizeof(double*));
		for(int i = 0; i < network[model].quantCamadas; i++){
			
			int neuron_length = network[model].camadas[i].quantNeuronios;
	    	int input_length = network[model].camadas[i].neuronio[0].quantEntradas;
	    	data.weights[i] = (double**) malloc(neuron_length * sizeof(double*));
			data.bias[i] = (double*) malloc(neuron_length * sizeof(double));
	    	
			for(int j = 0; j < neuron_length; j++){
	    		
				data.weights[i][j] = (double*) malloc(input_length * sizeof(double));
	    		
				for(int k = 0; k < input_length; k++){
					fread(&data.weights[i][j][k], sizeof(double), 1, file);
					network[model].camadas[i].neuronio[j].pesos[k] = data.weights[i][j][k];
				}
				
				fread(&data.bias[i][j], sizeof(double), 1, file);
				network[model].camadas[i].neuronio[j].bias = data.bias[i][j];
				free(data.weights[i][j]);
			}
			free(data.bias[i]);
			free(data.weights[i]);
		}
		free(data.weights);
		free(data.bias);
		
		network[model].labels = (char**) malloc(data.output * sizeof(char *));
		data.labels = (char**) malloc(data.output * sizeof(char *));
		for(int i = 0; i < data.output; i++){
			network[model].labels[i] = (char*) malloc(50 * sizeof(char));
			data.labels[i] = (char*) malloc(50 * sizeof(char));
			fread(data.labels[i], sizeof(char), 50, file);
			strcpy(network[model].labels[i], data.labels[i]);
			free(data.labels[i]);
		}
		free(data.labels);
	
	    fclose(file);
	    printf("Dados do modelo '%s' foram carregados!\n", filename);
	}
	
	void create_model(int index, int classes, char* name, char* path[], char* labels[]) {
		int layers = 4;
		int cols;
		int ammount[classes] = {0};
		char** files[classes];
		
		if(classes != network[index].classes && network[index].classes != 0){
			printf("Error: A quantidade de classes com neuronios de saida nao se coincidem!");
			return;
		}
		
		network[index].labels = (char**) malloc(classes * sizeof(char *));
		for(int i = 0; i < classes; i++){
			network[index].labels[i] = (char*) malloc(strlen(labels[i]) * sizeof(char));
			strcpy(network[index].labels[i], labels[i]);
		}
		network[index].name = name;
					
		for(int i = 0; i < classes; i++)
	    	files[i] = list_files(path[i], &ammount[i]);
	    	
	    if(files != NULL){
	    	printf("Arquivos encontrados (%d):\n", ammount[0]);
	    		
			double** label = create_label(classes, 2);
			label[0][0] = 1;
			label[0][1] = 0;
			label[1][0] = 0;
			label[1][1] = 1;
				    
	   		for (int i = 0; i < ammount[0]; i++) {
	   			//classes = (classes + i > ammount[0]) ? ammount[0] - i : classes;
	            double** inputs = create_input(classes, &cols, i, files, path);
					
				if(!isConfigured && i == 0){
					camadas[index] = (int*) malloc(layers * sizeof(int));
					camadas[index][0] = cols;
					camadas[index][1] = cols / 2;
					camadas[index][2] = cols / 2 / 2;
					camadas[index][3] = classes;
					network[index].iniciar(camadas[index], layers);
				}
		
				network[index].treinar(classes, inputs, label);
					
		    	close_image();
				close_inputs(inputs, classes);
	        }
	        
	        save_training(index, name);
	        	
	        close_inputs(label, classes);
	        for(int i = 0; i < classes; i++)
	        	close_files(files[i], ammount[i]);
	        isConfigured = true;
		}else{
			printf("Nenhum arquivo encontrado ou erro ao acessar a pasta.\n");
		}
	}
	
	void show_response(int model, const char* filename){
		int size = 0;
		double* tensor = create_tensor(filename, &size, true);
		if(tensor == NULL){
			printf("O arquivo nao foi encontrado!\n");
			return;
		}
		load_training(model, network[model].name);
		int x = network[model].quantCamadas;
		printf("Passando a imagem '%s' para previsao...\n", filename);
	    double** predicao = network[model].predizer(tensor);
	    int resultado = network[model].testar(predicao[x]);
	    printf("Predicao 0: %f, Predicao 1: %f, Resultado: %s\n", predicao[x][0], predicao[x][1], network[model].labels[resultado]);
	    printf("%s -> %.1f%%, %s -> %.1f%%", network[model].labels[0], (predicao[x][0] * 100.0), network[model].labels[1], (predicao[x][1] * 100.0));
	    free(tensor);
	    close_image();
	}
	
	/*
	char* get_response(int model, const char* filename){
		int size = 0;
		double* tensor = create_tensor(filename, &size, true);
		int x = network[model].quantCamadas;
	    double predicao1 = network[model].predizer(tensor)[x][0];
	    int resultado1 = network[model].testar(predicao1);
	    free(tensor);
	    return (char*) network[model].labels[resultado1];
	}
	*/
	
	void close_inputs(double** inputs, int size){
		for (int i = 0; i < size; i++) 
	        free(inputs[i]);
	    free(inputs);
	}
	
	void close_files(char** files, int size){
		for (int i = 0; i < size; i++) 
	        free(files[i]);
	    free(files);
	}
	
	void close_network(){
		for(int i = 0; i < networkAmmt; i++){
			free(network[i].camadas);
			free(camadas[i]);
			
			for(int j = 0; j < data.output; j++){
				free(network[i].labels[j]);
			}
			free(network[i].labels);
		}
		free(network);
		free(camadas);
	}
	
};
