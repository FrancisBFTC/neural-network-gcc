#include "src/neuralnet.h"
#include "src/tensor.h"
#include "src/files.h"

#define INPUTS 256
#define HIDDEN1 128
#define HIDDEN2 64
#define OUTPUTS 2

typedef struct {
    double input_to_hidden1[INPUTS][HIDDEN1];
    double bias_0[HIDDEN1];
    double hidden1_to_hidden2[HIDDEN1][HIDDEN2];
    double bias_1[HIDDEN2];
    double hidden2_to_output[HIDDEN2][OUTPUTS];
    double bias_2[OUTPUTS];
    char labels[OUTPUTS][50];  // Rótulos das saídas (até 50 caracteres cada)
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
	
	double** create_input(int rows, int *cols, int index, char** filename, const char* path){
		int length; 
		stringstream pathfile;
		double** tensor = (double**) malloc(rows * sizeof(double *));
		for(int x = 0; x < rows; x++){
			pathfile << path << "/" << filename[x+index];
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
	
	    for(int i = 0; i < network[model].quantCamadas; i++){
	    	for(int j = 0; j < network[model].camadas[i].quantNeuronios; j++){
	    		for(int l = 0; l < network[model].camadas[i].neuronio[j].quantEntradas; l++){
	    			switch(i){
	    				case 0: data.input_to_hidden1[l][j] = network[model].camadas[i].neuronio[j].pesos[l];
	    						data.bias_0[j] = network[model].camadas[i].neuronio[j].bias;
	    						//printf("Bias %d salvo: %f\n", j, data.bias_0[j]);
	    						break;
	    				case 1: data.hidden1_to_hidden2[l][j] = network[model].camadas[i].neuronio[j].pesos[l];
	    						data.bias_1[j] = network[model].camadas[i].neuronio[j].bias;
	    						//printf("Peso %d salvo: %f\n", l, network[model].camadas[i].neuronio[j].pesos[l]);
	    						break;
	    				case 2: data.hidden2_to_output[l][j] = network[model].camadas[i].neuronio[j].pesos[l];
	    						data.bias_2[j] = network[model].camadas[i].neuronio[j].bias;
	    						break;
					}
				}
			}
		}
		int index = network[model].quantCamadas-1;
		for(int i = 0; i < network[model].camadas[index].quantNeuronios; i++){
			strcpy(data.labels[i], network[model].labels[i]);
		}
	
	    // Escreve os pesos
	    fwrite(data.input_to_hidden1, sizeof(double), INPUTS * HIDDEN1, file);
	    fwrite(data.bias_0, sizeof(double), HIDDEN1, file);
	    fwrite(data.hidden1_to_hidden2, sizeof(double), HIDDEN1 * HIDDEN2, file);
	    fwrite(data.bias_1, sizeof(double), HIDDEN2, file);
	    fwrite(data.hidden2_to_output, sizeof(double), HIDDEN2 * OUTPUTS, file);
	    fwrite(data.bias_2, sizeof(double), OUTPUTS, file);
	
	    // Escreve os labels
	    fwrite(data.labels, sizeof(char), OUTPUTS * 50, file);
	
	    fclose(file);
	    printf("Dados do modelo '%s' foram salvos!\n", filename);
	}
	
	void load_training(int model, const char *filename) {
	    FILE *file = fopen(filename, "rb");
	    if (!file) {
	        perror("Erro ao abrir o arquivo");
	        return;
	    }
	
	    // Lê os pesos
	    fread(&data.input_to_hidden1, sizeof(double), INPUTS * HIDDEN1, file);
	    fread(&data.bias_0, sizeof(double), HIDDEN1, file);
	    fread(&data.hidden1_to_hidden2, sizeof(double), HIDDEN1 * HIDDEN2, file);
	    fread(&data.bias_1, sizeof(double), HIDDEN2, file);
	    fread(&data.hidden2_to_output, sizeof(double), HIDDEN2 * OUTPUTS, file);
	    fread(&data.bias_2, sizeof(double), OUTPUTS, file);
	    
	    // Lê os labels
	    fread(&data.labels, sizeof(char), OUTPUTS * 50, file);
	    
	    for(int i = 0; i < network[model].quantCamadas; i++){
	    	for(int j = 0; j < network[model].camadas[i].quantNeuronios; j++){
	    		for(int l = 0; l < network[model].camadas[i].neuronio[j].quantEntradas; l++){
	    			switch(i){
	    				case 0: network[model].camadas[i].neuronio[j].pesos[l] = data.input_to_hidden1[l][j];
	    						network[model].camadas[i].neuronio[j].bias = data.bias_0[j];
	    						//printf("Bias %d lido: %f\n", j, network[model].camadas[i].neuronio[j].bias);
	    						break;
	    				case 1: network[model].camadas[i].neuronio[j].pesos[l] = data.hidden1_to_hidden2[l][j];
	    						network[model].camadas[i].neuronio[j].bias = data.bias_1[j];
	    						//printf("Peso %d lido: %f\n", l, network[model].camadas[i].neuronio[j].pesos[l]);
	    						break;
	    				case 2: network[model].camadas[i].neuronio[j].pesos[l] = data.hidden2_to_output[l][j];
	    						network[model].camadas[i].neuronio[j].bias = data.bias_2[j];
	    						break;
					}
				}
			}
		}
		int index = network[model].quantCamadas-1;
		for(int i = 0; i < network[model].camadas[index].quantNeuronios; i++){
			strcpy(network[model].labels[i], data.labels[i]);
		}
	
	    fclose(file);
	    printf("Dados do modelo '%s' foram carregados!\n", filename);
	}
	
	void create_model(int index, int classes, char* name, const char* path, const char* pos_label, const char* neg_label){
		int layers = 4;			// Quantidade de camadas para este modelo
		//int classes = 2;		// Quantidade de classes para carregar: 2 -> Quadrado, Triangulo
		int cols;
		int ammount = 0;
		
		if(classes != network[index].classes && network[index].classes != 0){
			printf("Error: A quantidade de classes com neuronios de saida nao se coincidem!");
			return;
		}
		
		strcpy(network[index].labels[1], pos_label);
		strcpy(network[index].labels[0], neg_label);
		network[index].name = name;
					
		// Chama a função para listar arquivos
	    char** files = list_files(path, &ammount);
	    	
	    if(files != NULL){
	    	printf("Arquivos encontrados (%d):\n", ammount);
	    		
			double** label = create_label(classes, 2);
			label[0][0] = 1;
			label[0][1] = 0;
			label[1][0] = 0;
			label[1][1] = 1;
			
			int classes_old = classes;
				    
	   		for (int i = 0; i < ammount; i+=classes) {
	   			classes = (classes + i > ammount) ? ammount - i : classes;
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
	        	
	        close_inputs(label, classes_old);
	        close_files(files, ammount);
		}else{
			printf("Nenhum arquivo encontrado ou erro ao acessar a pasta.\n");
		}
	}
	
	void show_response(int model, const char* filename){
		int size = 0;
		int x = network[model].quantCamadas;
		double* tensor = create_tensor(filename, &size, true);
		load_training(model, network[model].name);
		printf("Passando a imagem '%s' para previsao...\n", filename);
	    double** predicao = network[model].predizer(tensor);
	    int resultado = network[model].testar(predicao[x][0]);
	    printf("Predicao 0: %f, Predicao 1: %f, Resultado: %s\n", predicao[x][0], predicao[x][1], network[model].labels[resultado]);
	    printf("%s -> %.1f%%, %s -> %.1f%%", network[model].labels[1], (predicao[x][0] * 100.0), network[model].labels[0], (predicao[x][1] * 100.0));
	    free(tensor);
	    close_image();
	}
	
	char* get_response(int model, const char* filename){
		int size = 0;
		double* tensor = create_tensor(filename, &size, true);
		int x = network[model].quantCamadas;
	    double predicao1 = network[model].predizer(tensor)[x][0];
	    int resultado1 = network[model].testar(predicao1);
	    free(tensor);
	    return (char*) network[model].labels[resultado1];
	}
	
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
		}
		free(network);
		free(camadas);
	}
	
};
