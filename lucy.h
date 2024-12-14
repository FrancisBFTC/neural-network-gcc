#include "src/neuralnet.h"
#include "src/tensor.h"
#include "src/files.h"

struct Lucy {
	
	Neural* network;
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
	
	void create_model(int index, int classes, char* name, const char* path, const char* pos_label, const char* neg_label){
		int layers = 4;			// Quantidade de camadas para este modelo
		//int classes = 2;		// Quantidade de classes para carregar: 2 -> Quadrado, Triangulo
		int cols;
		int ammount = 0;
		
		if(classes != network[index].classes && network[index].classes != 0){
			printf("Error: A quantidade de classes com neuronios de saida nao se coincidem!");
			return;
		}
		
		network[index].labels[1] = pos_label;
		network[index].labels[0] = neg_label;
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
	        	
	        close_inputs(label, classes_old);
	        close_files(files, ammount);
		}else{
			printf("Nenhum arquivo encontrado ou erro ao acessar a pasta.\n");
		}
	}
	
	void show_response(int model, const char* filename){
		int size = 0;
		double* tensor = create_tensor(filename, &size, true);
		int x = network[model].quantCamadas;
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
