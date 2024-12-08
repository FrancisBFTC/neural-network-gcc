#include "src/neuralnet.h"

int main()
{

	// Alocar dinamicamente a matriz de entradas
	int rows = 3, cols = 3;
	double **entradas = (double**) malloc(rows * sizeof(double *));
	for (int i = 0; i < rows; i++)
	    entradas[i] = (double*) malloc(cols * sizeof(double));
	
	// Preencher as entradas com os valores específicos
	entradas[0][0] = 0.8; entradas[0][1] = 0.8; entradas[0][2] = 0.8;
	entradas[1][0] = 0.4; entradas[1][1] = 0.4; entradas[1][2] = 0.4;
	entradas[2][0] = 0.2; entradas[2][1] = 0.2; entradas[2][2] = 0.2;
	
	// Alocar dinamicamente a matriz de rótulos
	rows = 3; cols = 1;
	double **rotulos = (double**) malloc(rows * sizeof(double *));
	for (int i = 0; i < rows; i++)
	    rotulos[i] = (double*) malloc(cols * sizeof(double));
	    
	// Preencher os rótulos com os valores específicos
	rotulos[0][0] = 1; rotulos[1][0] = 0; rotulos[2][0] = 0;
    
    int camadas[] = {3, 4, 1};
    double taxaAprendizagem = 0.5;
    int epocas = 100;

    Neural RedeNeural;
    RedeNeural.iniciaCamadas(camadas, sizeof(camadas) / sizeof(camadas[0]));
    RedeNeural.treinar(3, (double**)entradas, (double**)rotulos, epocas, taxaAprendizagem);
	
    double teste1[3] = {0.8, 0.8, 0.8};
    double teste2[3] = {0.4, 0.4, 0.4};
    double predicao1 = RedeNeural.predizer(teste1)[0];
    double predicao2 = RedeNeural.predizer(teste2)[0];
    int resultado1 = RedeNeural.testar(predicao1);
    int resultado2 = RedeNeural.testar(predicao2);

	if(resultado1 != -1 && resultado2 != -1){
		cout << endl;
	    printf("Predicao 1: %f, Resultado 1: %d\n", predicao1, resultado1);
	    printf("Predicao 2: %f, Resultado 2: %d\n", predicao2, resultado2);		
	}
    
	free(RedeNeural.camadas);
	
    return 0;
}

