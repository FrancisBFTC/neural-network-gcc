struct Layer{
    Neuron* neuronio;
    int quantNeuronios = 0;

    // Inicializa os neurônios
    void inicializa(int numeroNeuronios, int quantEntradas){
    	//cout << "quantidade de entradas 1: " << quantEntradas << endl;
        quantNeuronios = numeroNeuronios;
        neuronio = (Neuron*) malloc(sizeof(Neuron) * quantNeuronios);
        for(int i = 0; i < numeroNeuronios; i++)
            neuronio[i].iniciaPesos(quantEntradas);
    }

    // Avançar camadas (Forward Propagation)
    double* avancar(double* entradas){
        double* saidas = (double*) malloc(quantNeuronios * sizeof(double));
        for(int i = 0; i < quantNeuronios; i++)
        	saidas[i] = neuronio[i].ativar(entradas, neuronio[i].quantEntradas);
        return saidas;
    }
};
