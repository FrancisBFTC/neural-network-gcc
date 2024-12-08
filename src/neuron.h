struct Neuron {
    double bias;
    double* pesos;
    int quantEntradas = 0;

    // Inicializa pesos
    void iniciaPesos(int tamanho){
        quantEntradas = tamanho;
        pesos = (double*) malloc(sizeof(double) * tamanho);
        srand(21);
        for(int i = 0; i < tamanho; i++)
            pesos[i] = inicializacaoXavier(tamanho);
    }
    // Inicializa��o de Xavier
    double inicializacaoXavier(int tamanho){
    	return ((double) rand() / RAND_MAX) * sqrt(2.0 / (tamanho * 2));
	}

    // Ativa o neur�nio
    double ativar(double* entradas){
        double sum = 0;
        for(int i = 0; i < quantEntradas; i++)
            sum += entradas[i] * pesos[i];
        sum += bias;
        return sigmoid(sum);
    }

    // Ajusta as sa�das da ativa��o
    double sigmoid(double x){
        return 1 / (1 + exp(-x));
    }
};
