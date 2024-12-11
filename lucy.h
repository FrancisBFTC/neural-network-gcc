#include "src/neuralnet.h"
#include "src/tensor.h"

#include <windows.h>
#include <string.h>

char** list_files(const char*, int*);

struct Lucy {
	
	double** create_input(int rows, int *cols, int index, char** filename){
		int length; 
		double** tensor = (double**) malloc(rows * sizeof(double *));
		for(int x = 0; x < rows; x++){
			printf("Carregando arquivo: %s ...\n", filename[x+index]);
			tensor[x] = create_tensor(filename[x+index], &length, false);
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
	
	Neural create_model(const char* path, const char* pos_label, const char* neg_label){
		int size = 4;
		int rows = 2;
		int cols;
		int ammount = 0;
		int camadas[size];
		Neural RedeNeural;
		
		// Chama a função para listar arquivos
    	char** files = list_files(path, &ammount);
    	
    	if(files != NULL){
    		printf("Arquivos encontrados (%d):\n", ammount);
    		
			double** rotulos = create_label(rows, 1);
			rotulos[0][0] = 1;
			rotulos[1][0] = 0;
			RedeNeural.labels[1] = pos_label;
			RedeNeural.labels[0] = neg_label;
			int rows_old = rows;
			    
    		for (int i = 0; i < ammount; i+=rows) {
    			rows = (rows + i > ammount) ? ammount - i : rows;
	            double** inputs = create_input(rows, &cols, i, files);
	            printf("Valor: %d\n", i);
	            
				if(!i){
					camadas[0] = cols;
					camadas[1] = 8;
					camadas[2] = 5;
					camadas[3] = 1;
					RedeNeural.iniciar(camadas, size);
				}
				
				RedeNeural.treinar(rows, inputs, rotulos, 10, 0.01);
				
				// Liberar recursos
	    		close_image();
				close_inputs(inputs, rows);
        	}
        	
        	close_inputs(rotulos, rows_old);
        	close_files(files, ammount);
		}else{
			printf("Nenhum arquivo encontrado ou erro ao acessar a pasta.\n");
		}
		
	    return RedeNeural;
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
	
};

// Função para listar os arquivos em uma pasta
char** list_files(const char* folder, int* ammount) {
    WIN32_FIND_DATA datas;
    HANDLE hFind;
    char path[256];
    char** files = NULL;
    int counter = 0;

    // Monta o caminho com o padrão "*"
    snprintf(path, sizeof(path), "%s\\*", folder);

    // Inicia a busca
    hFind = FindFirstFile(path, &datas);
    if (hFind == INVALID_HANDLE_VALUE) {
        printf("Erro ao acessar o diretorio.\n");
        *ammount = 0;
        return NULL;
    }

    do {
        // Ignora diretórios
        if (!(datas.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            counter++;
            files = (char**) realloc(files, counter * sizeof(char*));
            if (files == NULL) {
                perror("Erro ao alocar memoria");
                FindClose(hFind);
                *ammount = 0;
                return NULL;
            }
            files[counter - 1] = strdup(datas.cFileName); // Duplica o nome do arquivo
        }
    } while (FindNextFile(hFind, &datas) != 0);

    FindClose(hFind);

    *ammount = counter;
    return files;
}
