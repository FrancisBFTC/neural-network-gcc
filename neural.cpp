#include "lucy.h" 

int main()
{

	Lucy lucy;

	int size = 0;
	Neural modelo = lucy.create_model("quadrados", "E UM QUADRADO", "NAO E UM QUADRADO");
    double* tensor3 = create_tensor("quadrado0.png", &size, false);
    double predicao1 = modelo.predizer(tensor3)[0];
    int resultado1 = modelo.testar(predicao1);
    printf("Predicao: %f, Resultado: %s\n", predicao1, modelo.labels[resultado1]);
	
    free(tensor3);
	free(modelo.camadas);
	
    return 0;
}

