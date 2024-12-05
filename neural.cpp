#include "src/neuralnet.h"

int main()
{	

	int rows = 2, cols = 3;
    double **entradas = (double**) malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        entradas[i] = (double*) malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            entradas[i][j] = 0.70 * (i + 1);
        }
    }
    
    rows = 2, cols = 1;
    double **rotulos = (double**)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        rotulos[i] = (double*)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            rotulos[i][j] = i;
        }
    }
	
	int camadas[] = {3, 4, 1};
	double aprendizagem = 0.55;		// Ajuste conforme o necessário
	int epocas = 10;
    int tamanho = sizeof(camadas) / sizeof(camadas[0]);
    int tamanhoEntradas = rows;
    
	Neural RedeNeural;
	RedeNeural.iniciaCamadas(camadas, tamanho);
	RedeNeural.treinar(tamanhoEntradas, entradas, rotulos, epocas, aprendizagem);
	
	cout << endl;
	double* saidas = RedeNeural.predizer(entradas[0]);
	cout << "Predicao : " << saidas[0] << endl;
	
	int result = RedeNeural.testar(entradas[0]);
	cout << "Resultado : " << result << endl;
	
	free(entradas);
	free(rotulos);
	free(RedeNeural.camadas);
	
    return 0;
}

