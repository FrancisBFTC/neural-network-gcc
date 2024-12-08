#include "src/neuralnet.h"

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

void show_matrix(unsigned char* gray_image, int width, int height){
	// Exibir os valores de escala de cinza
    printf("Imagem em escala de cinza:\n");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%3d ", gray_image[y * width + x]);
        }
        printf("\n");
    }
}

int main()
{
	const char *filename = "imagem_16x16.png";
	const char *filename_gray = "imagem_16x16_cinza.png";
	int width, height, channels;
	
    // Carregar a imagem
    unsigned char* rgb_image = stbi_load("imagem_16x16.png", &width, &height, &channels, 3);
    printf("Imagem carregada: %dx%d, canais: %d\n", width, height, channels);
    unsigned char* gray_image = (unsigned char*)malloc(width * height * sizeof(char));
    rgb_to_grayscale(rgb_image, gray_image, width, height);
	show_matrix(gray_image, width, height);
	
	// Salva a imagem em escala de cinza
    if (stbi_write_png(filename_gray, width, height, 1, gray_image, width) == 0) {
        printf("Erro ao salvar a imagem em escala de cinza\n");
    } else {
        printf("Imagem em escala de cinza salva como 'imagem_16x16_cinza.png'\n\n");
    }
	
	// Criar o tensor de escala de cinza
    double* tensor = (double*)malloc(width * height * sizeof(double));
    create_grayscale_tensor(gray_image, tensor, width, height);   
    // Exibir os valores do tensor
    printf("Tensor de escala de cinza (normalizado):\n");
    for (int y = 0; y < width; y++) {
        for (int x = 0; x < height; x++) {
            printf("%.2f ", tensor[y * width + x]);
        }
        printf("\n");
    }

	// Alocar dinamicamente a matriz de entradas
	int length = width * height;
	int rows = 1, cols = length;
	double **entradas = (double**) malloc(rows * sizeof(double *));
	for (int i = 0; i < rows; i++)
	    entradas[i] = tensor; //(double*) malloc(cols * sizeof(double));
	
	// Criar rótulos
    double** rotulos = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        rotulos[i] = (double*)malloc(1 * sizeof(double));
        rotulos[i][0] = 1; // Exemplo de rótulo
    }
    
	// Preencher os rótulos com os valores específicos
	rotulos[0][0] = 1; //rotulos[1][0] = 0; rotulos[2][0] = 0;
    
    int camadas[3];
    camadas[0] = length;	// Quantidade de entradas iniciais
    camadas[1] = 4;			// Quantidade de neurônios na camada oculta
    camadas[2] = 1;			// Quantidade de neurônios na camada de saída
    double taxaAprendizagem = 0.5;
    int epocas = 200;

	Neural RedeNeural;
    RedeNeural.iniciaCamadas(camadas, sizeof(camadas) / sizeof(camadas[0]));
    RedeNeural.treinar(rows, entradas, rotulos, epocas, taxaAprendizagem);
    
    // Carregar a imagem
    rgb_image = stbi_load("preto_16x16.png", &width, &height, &channels, 3);
    printf("\nImagem carregada: %dx%d, canais: %d\n", width, height, channels);
    rgb_to_grayscale(rgb_image, gray_image, width, height);
    create_grayscale_tensor(gray_image, tensor, width, height); 
    
    double predicao1 = RedeNeural.predizer(tensor)[0];
    //double predicao2 = RedeNeural.predizer(tensor1)[0];
    int resultado1 = RedeNeural.testar(predicao1);
    //int resultado2 = RedeNeural.testar(predicao2);

	if(resultado1 != -1){	// && resultado2 != -1
		cout << endl;
	    printf("Predicao 1: %f, Resultado 1: %d\n", predicao1, resultado1);
	    //printf("Predicao 2: %f, Resultado 2: %d\n", predicao2, resultado2);		
	}	
	
	// Liberar recursos
    free(RedeNeural.camadas);
    for (int i = 0; i < rows; i++) {
        free(rotulos[i]);
    }
    free(rotulos);
    free(tensor);
    stbi_image_free(rgb_image);
    free(gray_image);
    free(tensor);
    //free(tensor1);
	
    return 0;
}

