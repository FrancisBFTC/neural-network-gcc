// Inclui a biblioteca para leitura
#define STB_IMAGE_IMPLEMENTATION
#include "img/stb_image.h"

// Inclui a biblioteca para escrita
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "img/stb_image_write.h"

// Variáveis globais para imagens
unsigned char* rgb_image;
unsigned char* gray_image;

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

// Cria tensor da escala de cinza
void create_grayscale_tensor(unsigned char* gray_image, double* tensor, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        tensor[i] = gray_image[i] / 255.0f; // Normalização
    }
}

// Exibir os valores de escala de cinza
void show_matrix(unsigned char* gray_image, int width, int height){
    printf("Imagem em escala de cinza:\n");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%3d ", gray_image[y * width + x]);
        }
        printf("\n");
    }
    printf("\n");
}

// Exibir os valores do tensor
void show_tensor(double tensor[256], int width, int height){
    printf("Tensor de escala de cinza (normalizado):\n");
    for (int y = 0; y < width; y++) {
        for (int x = 0; x < height; x++) {
            printf("%.2f ", tensor[y * width + x]);
        }
        printf("\n");
    }
    printf("\n");
}

double* create_tensor(char const *filename, int *length, bool verbose){
	int width, height, channels;
	rgb_image = stbi_load(filename, &width, &height, &channels, 3);
	*length = width * height;
	double* tensor = (double*) malloc(*length * sizeof(double));
    gray_image = (unsigned char*) malloc(*length * sizeof(char));
    rgb_to_grayscale(rgb_image, gray_image, width, height);
    create_grayscale_tensor(gray_image, tensor, width, height);
    if(verbose){
    	printf("\nImagem carregada: %dx%d, canais: %d\n", width, height, channels);
    	printf("Tamanho da imagem: %d\n\n", *length);
    	show_matrix(gray_image, width, height);
    	show_tensor(tensor, width, height);
	}
    
    return tensor;
}

void close_image(){
	stbi_image_free(rgb_image);
    free(gray_image);
}
