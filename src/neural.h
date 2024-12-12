struct Neural {
	Layer* camadas;
	int quantCamadas = 0;
	double erroTotal = 0.0;
	const char* labels[2];
	
	// Inicia as camadas
	void iniciar(int layers[], int size){
		quantCamadas = size-1;
		//cout << "Rede Neural [Main] -> Numero de camadas: " << size << endl;
		
		camadas = (Layer*) malloc(sizeof(Layer) * quantCamadas);
		for(int i = 0; i < quantCamadas; i++)
			camadas[i].inicializa(layers[i + 1], layers[i]);
		
		for(int i = 0; i < quantCamadas; i++){
			//cout << "\tCamada " << i+1 << " -> Numero de neuronios: " << camadas[i].quantNeuronios << endl;
			for(int j = 0; j < camadas[i].quantNeuronios; j++){
				//cout << "\t\tNeuronio " << j << " -> Numero de entradas: " << camadas[i].neuronio[j].quantEntradas << " (camada " << i << ")"<< endl;
				for(int w = 0; w < camadas[i].neuronio[j].quantEntradas; w++){
					//cout << "\t\t\tEntrada " << w << " -> Peso: " << camadas[i].neuronio[j].pesos[w] << endl;
				}
			}
		}
	}
	
	// Predizer os resultados da rede neural
	double* predizer(double* entradas){	
		double* saidas = entradas;
		for(int i = 0; i < quantCamadas; i++)
			saidas = camadas[i].avancar(saidas);
		return saidas;
	}
	
	int testar(double predicao){
		//if (predicao > 0.4 && predicao <= 0.6) {
		//    cout << "Indefinido. Precisa ajustar os parametros da rede neural." << endl;
		//    return -1;
		//}
		return round(predicao);
	}
	
	// Treinamento da rede neural
	void treinar(int quantEntradas, double** entradas, double** rotulos, int epocas, double taxaAprendizagem){	//double** rotulos
		for(int epoca = 0; epoca < epocas; epoca++){
			for(int i = 0; i < quantEntradas; i++){
				double* saidas = predizer(entradas[i]);
				double** deltas = retropropagar(saidas, rotulos[i]);
				atualizaPesos(deltas, entradas[i], taxaAprendizagem);
	            taxaAprendizagem = taxaAprendizagem / (1 + epoca / epocas);
	            
	            for(int i = 0; i < quantCamadas; i++)
	            	free(deltas[i]);
	            free(deltas);
	            free(saidas);
			}
			cout << "Epoca: " << epoca + 1 << ", Erro Total: " << erroTotal << endl;
		}
	}
	
	// Propagação para trás (Back Propagation - Otimização de erros)
	double** retropropagar(double* saidas, double* rotulos){	//double* rotulos
		double** deltas = (double**) malloc(quantCamadas * sizeof(double));
		erroTotal = 0.0;
		
		for(int j = quantCamadas - 1; j >= 0; j--){
			deltas[j] = (double*) malloc(camadas[j].quantNeuronios * sizeof(double));
			for(int k = 0; k < camadas[j].quantNeuronios; k++){
				if(j == quantCamadas - 1){
					deltas[j][k] = saidas[k] - rotulos[k];
					erroTotal += pow(deltas[j][k], 2);
					
					//cout << "quantNeuronios: " << camadas[j].quantNeuronios<< endl;
					//cout << "saida: " << saidas[k] << ", rotulo: " << rotulos[k] << ", delta: " << saidas[k] - rotulos[k] << endl;
					//cout << "\nCAMADA OCULTA DA BACKPROPAGATION:" << endl;
					//cout << "\tDelta {"<< j <<"}{"<< k <<"}:" << deltas[j][k] << endl;
					//printf("\t Saidas {%d} = %f, Rotulos {%d} = %f\n", k, saidas[k], k, rotulos[k]);
				}else{
					double error = 0.0;
                    for (int l = 0; l < camadas[j + 1].quantNeuronios; l++)
                        error += deltas[j + 1][l] * camadas[j + 1].neuronio[l].pesos[k];
                    //cout << "ENTRADA DA RETROPROPAGACAO" << endl;
                    double ativacao = camadas[j].neuronio[k].ativar(saidas, 1);
                    double otimizacao = sigmoidDerivative(ativacao);
                    deltas[j][k] = error * otimizacao;
                    //cout << "\tDelta {"<< j <<"}{"<< k <<"}: " << deltas[j][k] << ", Error: " << error << ", Optimization: " << otimizacao << endl;
				}
				
			}
		}
		
		return deltas;
	}
	
	// Atualização de pesos baseado nos erros
	void atualizaPesos(double** deltas, double* entradas, double taxaAprendizagem){
		for(int j = 0; j < quantCamadas; j++){
			double* camadaEntradas = (j == 0) ? entradas : camadas[j - 1].avancar(entradas);
			int sizeCamadaEntradas = sizeof(camadaEntradas) / sizeof(camadaEntradas[0]);
			for(int k = 0; k < camadas[j].quantNeuronios; k++){
				for(int l = 0; l < sizeCamadaEntradas; l++){
					camadas[j].neuronio[k].pesos[l] -= taxaAprendizagem * deltas[j][k] * camadaEntradas[l];
				}
				camadas[j].neuronio[k].bias -= taxaAprendizagem * deltas[j][k];
			}
		}
	}
	
	// Otimização de saída da atualização
	double sigmoidDerivative(double x){
		return x * (1 - x);
	}
};
