#include "src/neuralnet.h"
#include <stdio.h>
#include <string>
#include <sstream>

// Inclui a biblioteca para leitura
#define STB_IMAGE_IMPLEMENTATION
#include "src/img/stb_image.h"

// Inclui a biblioteca para escrita
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "src/img/stb_image_write.h"

// Função para converter uma imagem RGB para escala de cinza
void rgb_to_grayscale(unsigned char* rgb_image, unsigned char* gray_image, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        int r = rgb_image[3 * i];
        int g = rgb_image[3 * i + 1];
        int b = rgb_image[3 * i + 2];
        // Fórmula para converter para escala de cinza
        gray_image[i] = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
    }
}

void create_grayscale_tensor(unsigned char* gray_image, double* tensor, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        tensor[i] = gray_image[i] / 255.0f; // Normalização
    }
}

double* get_tensor(char const* filename){
	stringstream imgs;
	imgs << filename << ".png";
	int width1, height1, channels1;
	unsigned char *rgb_image1 = stbi_load("imagem_16x16.png", &width1, &height1, &channels1, 3);
    if (!rgb_image1) {
        printf("Erro ao carregar a imagem\n");
        return {};
    }
    
    printf("Imagem carregada: %dx%d, canais: %d\n", width1, height1, channels1);
    
    // Criar uma imagem em escala de cinza
    unsigned char *gray_image1 = (unsigned char*)malloc(width1 * height1);
    rgb_to_grayscale(rgb_image1, gray_image1, width1, height1);
    
     // Exibir os valores de escala de cinza
    printf("Imagem em escala de cinza:\n");
    for (int y = 0; y < height1; y++) {
        for (int x = 0; x < width1; x++) {
            printf("%3d ", gray_image1[y * width1 + x]);
        }
        printf("\n");
    }
    
    imgs.str("");
    imgs << filename << "_gray" << ".png";
    
    // Salva a imagem em escala de cinza
    if (stbi_write_png("imagem_16x16_cinza.png", width1, height1, 1, gray_image1, width1) == 0) {
        printf("Erro ao salvar a imagem em escala de cinza\n");
    } else {
        printf("Imagem em escala de cinza salva como 'imagem_16x16_cinza.png'\n\n");
    }
	
	// Criar o tensor de escala de cinza
    double* tensor1 = (double*)malloc(width1 * height1 * sizeof(double));
    create_grayscale_tensor(gray_image1, tensor1, width1, height1);   
    
    // Exibir os valores do tensor
    printf("Tensor de escala de cinza (normalizado):\n");
    for (int y = 0; y < height1; y++) {
        for (int x = 0; x < width1; x++) {
            printf("%.2f ", tensor1[y * width1 + x]);
        }
        printf("\n");
    }
    //stbi_image_free(rgb_image1);
    //free(gray_image1);
    
    return tensor1;
}

int main()
{
	/*
	char const *filename = "imagem_16x16.png";
	char const *filename_gray = "imagem_16x16_cinza.png";
	
	int width, height, channels;
    // Carregar a imagem
    unsigned char* rgb_image = stbi_load(filename, &width, &height, &channels, 3);
    if (!rgb_image) {
        printf("Erro ao carregar a imagem\n");
        return 1;
    }
    
    printf("Imagem carregada: %dx%d, canais: %d\n", width, height, channels);
    
    // Criar uma imagem em escala de cinza
    unsigned char* gray_image = (unsigned char*)malloc(width * height);
    rgb_to_grayscale(rgb_image, gray_image, width, height);
    
    // Exibir os valores de escala de cinza
    printf("Imagem em escala de cinza:\n");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%3d ", gray_image[y * width + x]);
        }
        printf("\n");
    }
	
	// Salva a imagem em escala de cinza
    if (stbi_write_png(filename_gray, width, height, 1, gray_image, width) == 0) {
        printf("Erro ao salvar a imagem em escala de cinza\n");
    } else {
        printf("Imagem em escala de cinza salva como '%s'\n\n", filename_gray);
    }
	
	// Criar o tensor de escala de cinza
    double* tensor = (double*)malloc(width * height * sizeof(double));
    create_grayscale_tensor(gray_image, tensor, width, height);   
    
    // Exibir os valores do tensor
    printf("Tensor de escala de cinza (normalizado):\n");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%.2f ", tensor[y * width + x]);
        }
        printf("\n");
    }
    */


	double* tensor = get_tensor("imagem_16x16");
	int length = 256;
	
	// Alocar dinamicamente a matriz de entradas
	//int length = width * height;
	int rows = 1, cols = length;
	double **entradas = (double**) malloc(rows * sizeof(double *));
	for (int i = 0; i < rows; i++)
	    entradas[i] = tensor; //(double*) malloc(cols * sizeof(double));
	
	// Preencher as entradas com os valores específicos
	//entradas[0][0] = 0.8; entradas[0][1] = 0.8; entradas[0][2] = 0.8;
	//entradas[1][0] = 0.4; entradas[1][1] = 0.4; entradas[1][2] = 0.4;
	//entradas[2][0] = 0.2; entradas[2][1] = 0.2; entradas[2][2] = 0.2;
	
	// Alocar dinamicamente a matriz de rótulos
	rows = 1; cols = 1;
	double **rotulos = (double**) malloc(rows * sizeof(double *));
	for (int i = 0; i < rows; i++)
	    rotulos[i] = (double*) malloc(cols * sizeof(double));
	    
	// Preencher os rótulos com os valores específicos
	rotulos[0][0] = 1; //rotulos[1][0] = 0; rotulos[2][0] = 0;
    
    int camadas[3];
    camadas[0] = length;	// Quantidade de entradas iniciais
    camadas[1] = 4;			// Quantidade de neurônios na camada oculta
    camadas[2] = 1;			// Quantidade de neurônios na camada de saída
    double taxaAprendizagem = 0.5;
    int epocas = 100;

    Neural RedeNeural;
    RedeNeural.iniciaCamadas(camadas, sizeof(camadas) / sizeof(camadas[0]));
    RedeNeural.treinar(1, (double**)entradas, (double**)rotulos, epocas, taxaAprendizagem);
	
	//double* tensor1 = get_tensor("cruz_16x16");
    
    //double teste1[3] = {0.8, 0.8, 0.8};
    //double teste2[3] = {0.4, 0.4, 0.4};
    double predicao1 = RedeNeural.predizer(tensor)[0];
    //double predicao2 = RedeNeural.predizer(tensor1)[0];
    int resultado1 = RedeNeural.testar(predicao1);
    //int resultado2 = RedeNeural.testar(predicao2);

	if(resultado1 != -1){	// && resultado2 != -1
		cout << endl;
	    printf("Predicao 1: %f, Resultado 1: %d\n", predicao1, resultado1);
	    //printf("Predicao 2: %f, Resultado 2: %d\n", predicao2, resultado2);		
	}
    
	free(RedeNeural.camadas);
	
	// Liberar memória
    //stbi_image_free(rgb_image);
    //free(gray_image);
    free(tensor);
    //free(tensor1);
	
    return 0;
}

