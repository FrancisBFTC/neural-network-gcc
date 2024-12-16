struct Neural {
	Layer* camadas;
	int quantCamadas = 0;
	int layers_ammount = 4;
	int classes = 0;
	int epocas = 10000;
	double learning_rate = 0.5;
	double erroTotal = 0.0;
	char* name;
	char** labels;//[2][50];
	
	// Inicia as camadas
	void iniciar(int layers[], int size){
		quantCamadas = size-1;
		
		camadas = (Layer*) malloc(sizeof(Layer) * quantCamadas);
		for(int i = 0; i < quantCamadas; i++)
			camadas[i].inicializa(layers[i + 1], layers[i]);
	}
	
	// Predizer os resultados da rede neural
	double** predizer(double* input){
		double** output = (double**) malloc((quantCamadas + 1) * sizeof(double *));	
		output[0] = input;
		for(int i = 0; i < quantCamadas; i++)
			output[i + 1] = camadas[i].avancar(output[i]);
			
		softmax(output[quantCamadas]);
		return output;
	}
	
	void softmax(double* logits) {
	    double somaExp = 0;
	
	    // Calcula as exponenciais e a soma das exponenciais
	    for (int i = 0; i < 2; i++) {
	        logits[i] = exp(logits[i]); // calcula e armazena exp(x)
	        somaExp += logits[i];       // acumula a soma
	    }
	
	    // Divide cada exponencial pela soma total para normalizar
	    for (int i = 0; i < 2; i++) {
	        logits[i] /= somaExp;
	    }
	}
	
	int testar(double* predicao){
		int index = 0;
		while(!round(predicao[index])) index++;
		return index;
	}
	
	// Treinamento da rede neural
	void treinar(int quantEntradas, double** entradas, double** rotulos){	//double** rotulos
		printf("Treinando o modelo");
		for(int epoca = 0; epoca < epocas; epoca++){
			for(int i = 0; i < quantEntradas; i++){
				double** saidas = predizer(entradas[i]);
				double** deltas = retropropagar(saidas, rotulos[i]);
				atualizaPesos(deltas, saidas);
	            learning_rate = learning_rate / (1 + epoca / epocas);
	            
	            for(int i = 0; i < quantCamadas; i++)
	            	free(deltas[i]);
	            free(deltas);
	            free(saidas);
			}
			if(epoca % 1000 == 0)	printf(".");
			
			//cout << "Epoca: " << epoca + 1 << ", Erro Total: " << erroTotal << endl;
		}
		printf("\n");
	}
	
	// Propagação para trás (Back Propagation - Otimização de erros)
	double** retropropagar(double** saidas, double* rotulos){	//double* rotulos
		double** deltas = (double**) malloc(quantCamadas * sizeof(double));
		erroTotal = 0.0;
		
		for(int j = quantCamadas - 1; j >= 0; j--){
			deltas[j] = (double*) malloc(camadas[j].quantNeuronios * sizeof(double));
			for(int k = 0; k < camadas[j].quantNeuronios; k++){
				if(j == quantCamadas - 1){
					deltas[j][k] = saidas[j + 1][k] - rotulos[k];
					erroTotal += pow(deltas[j][k], 2);
				}else{
					double error = 0.0;
                    for (int l = 0; l < camadas[j + 1].quantNeuronios; l++)
                        error += deltas[j + 1][l] * camadas[j + 1].neuronio[l].pesos[k];
                    double ativacao = saidas[j + 1][k];
                    double otimizacao = sigmoidDerivative(ativacao);
                    deltas[j][k] = error * otimizacao;
				}
				
			}
		}
		
		return deltas;
	}
	
	// Atualização de pesos baseado nos erros
	void atualizaPesos(double** deltas, double** entradas){
		for(int j = 0; j < quantCamadas; j++){
			double* camadaEntradas = entradas[j];
			for(int k = 0; k < camadas[j].quantNeuronios; k++){
				for(int l = 0; l < camadas[j].neuronio[k].quantEntradas; l++){
					camadas[j].neuronio[k].pesos[l] -= learning_rate * deltas[j][k] * camadaEntradas[l];
				}
				camadas[j].neuronio[k].bias -= learning_rate * deltas[j][k];
			}
		}
	}
	
	// Otimização de saída da atualização
	double sigmoidDerivative(double x){
		return x * (1 - x);
	}
};
