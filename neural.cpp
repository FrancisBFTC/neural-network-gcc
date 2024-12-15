#include "lucy.h" 

int main(int argc, char** argv)
{	
	if(argc == 1){
		printf("Lucy API version 1.0.0\n");
		printf("Author: By Wenderson Francisco\n");
		printf("usage: lucy [--train] [--test]");
		return 0;
	}
	
	Lucy lucy;
	lucy.create_network(1);							// Crie uma rede neural
	
	for(int i = 1; i < argc; i++){
		if(strcmp(argv[i], "--train") == 0){
			
			lucy.initialize_network(0, 5000, 0.5);	// Inicialize a rede 0 com 5000 épocas e 0.5 de aprendizado
			lucy.create_model(0, 2, "fignet", "quadrados", "QUADRADO", "TRIANGULO");	// Treinar o modelo 0
		}
		if(strcmp(argv[i], "--test") == 0){
			lucy.build_layer(0, 4);					// Cria 4 camadas no modelo 0
			lucy.build_input(0, 256); 				// Cria 256 entradas iniciais no modelo 0
			lucy.build_hidden(0, 128);				// Cria 128 neurônios na 1ª camada oculta do modelo 0
			lucy.build_hidden(0, 64);				// Cria 64 neurônios na 2ª camada oculta do modelo 0
			lucy.build_output(0, 2); 				// Cria 2 neurônios de saída na camada de saída (classes)
			lucy.rename(0, "fignet");				// Atribua um nome para a rede
			lucy.initialize_network(0, 0, 0);		// Inicialize a rede 0
			
			if(argv[++i] != NULL)
				lucy.show_response(0, argv[i]);		// Faça a previsão
			else
				printf("Forneca uma imagem de entrada!\n");
		}
		//if(strcmp(argv[i], "--info") == 0){
			//lucy.show_network(0);					// Apresente as configurações manuais da rede neural 0
		//}
	}
	
	lucy.close_network();							// Fecha todas as redes neurais
	
    return 0;
}

