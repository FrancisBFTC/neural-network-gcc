#include "src/neuralnet.h"
#include "src/tensor.h"
#include "src/files.h"

struct Lucy {
	
	Neural* network;
	int networkAmmt = 0;
	
	void initialize_network(int dataset_count){
		network = (Neural*) malloc(dataset_count * sizeof(Neural));
		networkAmmt = dataset_count;
	}
	
	double** create_input(int rows, int *cols, int index, char** filename, const char* path){
		int length; 
		stringstream pathfile;
		double** tensor = (double**) malloc(rows * sizeof(double *));
		for(int x = 0; x < rows; x++){
			pathfile << path << "/" << filename[x+index];
			const char* imagename = pathfile.str().c_str();
			printf("Carregando arquivo: %s ...\n", imagename);
			tensor[x] = create_tensor(imagename, &length, true);
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
	
	void create_model(int index, const char* path, const char* pos_label, const char* neg_label){
		int size = 4;
		int rows = 2;
		int cols;
		int ammount = 0;
		
		// Chama a função para listar arquivos
    	char** files = list_files(path, &ammount);
    	
    	if(files != NULL){
    		printf("Arquivos encontrados (%d):\n", ammount);
    		
			double** rotulos = create_label(rows, 1);
			rotulos[0][0] = 1;
			rotulos[1][0] = 0;
			
			network[index].labels[1] = pos_label; //descriptions[0];
			network[index].labels[0] = neg_label; //desc;
			int rows_old = rows;
			    
    		for (int i = 0; i < ammount; i+=rows) {
    			rows = (rows + i > ammount) ? ammount - i : rows;
	            double** inputs = create_input(rows, &cols, i, files, path);
	            
				if(i == 0){
					int camadas[size] = {cols, 8, 5, 1}; 
					network[index].iniciar(camadas, size);
				}
				
				network[index].treinar(rows, inputs, rotulos, 20000, 0.01);
				
				// Liberar recursos
	    		close_image();
				close_inputs(inputs, rows);
        	}
        	
        	// Liberar recursos
        	close_inputs(rotulos, rows_old);
        	close_files(files, ammount);
		}else{
			printf("Nenhum arquivo encontrado ou erro ao acessar a pasta.\n");
		}
	
	}
	
	void show_response(int model, const char* filename){
		int size = 0;
		double* tensor = create_tensor(filename, &size, true);
	    double predicao1 = network[model].predizer(tensor)[0];
	    int resultado1 = network[model].testar(predicao1);
	    printf("Predicao: %f, Resultado: %s\n", predicao1, network[model].labels[resultado1]);
	    free(tensor);
	}
	
	char* get_response(int model, const char* filename){
		int size = 0;
		double* tensor = create_tensor(filename, &size, true);
	    double predicao1 = network[model].predizer(tensor)[0];
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
		for(int i = 0; i < networkAmmt; i++)
			free(network[i].camadas);
		free(network);
	}
	
};
