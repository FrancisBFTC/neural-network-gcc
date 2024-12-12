struct Neuron {
    double bias;
    double* pesos;
    int quantEntradas = 0;

    // Inicializa pesos
    void iniciaPesos(int tamanho){
        quantEntradas = tamanho;
        bias = 0.0;
        pesos = (double*) malloc(sizeof(double) * tamanho);
        srand(21);
        for(int i = 0; i < tamanho; i++)
            pesos[i] = inicializacaoXavier(tamanho);
    }
    // Inicialização de Xavier
    double inicializacaoXavier(int tamanho){
    	return ((double) rand() / RAND_MAX) * sqrt(2.0 / (tamanho * 2));
	}

    // Ativa o neurônio
    double ativar(double* entradas, int size){
        double sum = 0;
        //cout << "quantEntradas: " << quantEntradas << endl;
        for(int i = 0; i < size; i++){
        	sum += entradas[i] * pesos[i];
        	//cout << "Entradas: " << entradas[i] << ", pesos: " << pesos[i] << endl;
		}
		//cout << "Bias: " << bias << endl;
		//cout << "Soma antes do bias: " << sum << endl;
        sum += bias;
        //cout << "Soma depois do bias: " << sum << endl;
        return sigmoid(sum);
    }

    // Ajusta as saídas da ativação
    double sigmoid(double x){
        return 1 / (1 + exp(-x));
    }
};
