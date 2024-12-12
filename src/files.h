#include <windows.h>
#include <string.h>

// Função para listar os arquivos em uma pasta
char** list_files(const char* folder, int* ammount) {
    WIN32_FIND_DATA datas;
    HANDLE hFind;
    char path[256];
    char** files = NULL;
    int counter = 0;

    // Monta o caminho com o padrão "*"
    snprintf(path, sizeof(path), "%s\\*", folder);

    // Inicia a busca
    hFind = FindFirstFile(path, &datas);
    if (hFind == INVALID_HANDLE_VALUE) {
        printf("Erro ao acessar o diretorio.\n");
        *ammount = 0;
        return NULL;
    }

    do {
        // Ignora diretórios
        if (!(datas.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            counter++;
            files = (char**) realloc(files, counter * sizeof(char*));
            if (files == NULL) {
                perror("Erro ao alocar memoria");
                FindClose(hFind);
                *ammount = 0;
                return NULL;
            }
            files[counter - 1] = strdup(datas.cFileName); // Duplica o nome do arquivo
        }
    } while (FindNextFile(hFind, &datas) != 0);

    FindClose(hFind);

    *ammount = counter;
    return files;
}
