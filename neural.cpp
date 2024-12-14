#include "lucy.h" 

int main()
{	
	Lucy lucy;
	lucy.create_network(1);							// Crie uma rede neural
	
	lucy.build_layer(0, 4);						// Cria 4 camadas no modelo 0
	lucy.build_input(0, 256); 						// Cria 256 entradas iniciais no modelo 0
	lucy.build_hidden(0, 128);						// Cria 128 neur�nios na 1� camada oculta do modelo 0
	lucy.build_hidden(0, 64);						// Cria 64 neur�nios na 2� camada oculta do modelo 0
	lucy.build_output(0, 2); 						// Cria 2 neur�nios de sa�da na camada de sa�da (classes)
	lucy.rename(0, "fignet");						// Atribua um nome para a rede
	
	lucy.initialize_network(0, 5000, 0.5);			// Inicialize a rede 0 com 5000 �pocas e 0.5 de aprendizado
	//lucy.show_network(0);							// Apresente as configura��es manuais da rede neural 0			
	
	//lucy.create_model(0, 2, "fignet", "quadrados", "QUADRADO", "TRIANGULO");	// Modelo 0 - treina a pasta "quadrados" e atribua label "QUADRADO"
	//lucy.show_network(0);
	
	lucy.show_response(0, "quadrados/3-quadrado.png");	// Colete a resposta do modelo 0 baseado na imagem de entrada
	lucy.close_network();								// Fecha todas as redes neurais
	
	//char* label = lucy.get_response(0, "quadrados/#3-nao-quadrado1.png");	// retorna a resposta da label
	//printf("Resposta: %s\n", label);
    return 0;
}

