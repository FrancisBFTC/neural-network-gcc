#include "lucy.h" 

int main()
{	
	Lucy lucy;
	lucy.initialize_network(1);									// Inicialize 1 rede neural
	lucy.create_model(0, "quadrados", "QUADRADO", NULL);		// Modelo 0 - treina a pasta "quadrados" e atribua label "QUADRADO"
    lucy.show_response(0, "quadrados/#2-quadrado1.png");		// Colete a resposta do modelo 0 baseado na imagem de entrada
	lucy.close_network();										// Fecha todas as redes neurais
	
	//char* label = lucy.get_response(0, "quadrados/#3-nao-quadrado1.png");	// retorna a resposta da label
	//printf("Resposta: %s\n", label);
    return 0;
}

