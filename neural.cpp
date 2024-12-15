#include "lucy.h" 

int main(int argc, char** argv)
{	
	if(argc == 1){
		printf("Lucy API version 1.0.0\n");
		printf("Author: By Wenderson Francisco\n");
		printf("usage: lucy [--train] [--pred] [--info]");
		return 0;
	}
	
	Lucy lucy;
	lucy.create_network(1);							// Crie uma rede neural
	
	for(int i = 1; i < argc; i++){
		if(strcmp(argv[i], "--train") == 0){
			lucy.initialize_network(0, 5000, 0.5);	// Inicialize a rede 0 com 5000 épocas e 0.5 de aprendizado
			lucy.create_model(0, 2, argv[++i], "quadrados", "QUADRADO", "TRIANGULO");	// Treinar o modelo 0
		}
		
		if(strcmp(argv[i], "--pred") == 0){
			lucy.rename(0, argv[++i]);						// Atribua um nome para a rede
			
			if(argv[++i] != NULL)
				lucy.show_response(0, argv[i]);	// Faça a previsão da entrada na rede fignet
			else
				printf("Forneca uma imagem de entrada!\n");
		}
		
		if(strcmp(argv[i], "--info") == 0){
			if(argv[++i] != NULL){
				lucy.load_training(0, argv[i]);			// Carrega dados de treinamento
				
				FILE *file;
				if(argv[++i] != NULL){
					// Redireciona stdout para um arquivo
				    file = freopen(argv[i], "w", stdout);
				    if (!file) {
				        perror("Erro ao redirecionar stdout");
				        return 1;
				    }
				}
				
				lucy.show_network(0);					// Apresente as configurações manuais da rede neural 0
				
				if(argv[i] != NULL){
					// Restaurar stdout (opcional, útil se quiser voltar a imprimir no console)
				    freopen("CON", "w", stdout); // No Windows, use "CON"
				    printf("As configuracoes foram salvas em '%s'\n", argv[i]);
				}
			}else{
				printf("Forneca o nome do modelo!\n");	
			}
		}
	}
	
	lucy.close_network();							// Fecha todas as redes neurais
	
    return 0;
}

