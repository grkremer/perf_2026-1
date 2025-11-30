# perf_2026-1
### Introdução
Este repositório contém os arquivos do projeto referente ao semestre 2025-2 da disciplina de Analise de Performance.

### Dos arquivos necessários para execução
Na raiz do diretório há os arquivos functions.py e train_and_eval.py, que contêm as funções e o script principal para treinamento e avaliação do modelo, respectivamente.
Além disso, o repositório contém o arquivo run_test.sbatch, que é o script de submissão para a fila de processamento.

Para rodar o script principal localmente, você pode usar o seguinte comando no terminal:
```run_test.sbatch -<NOME DO ARQUIVO DE SAÍDA>```
Caso queira modificar as configurações do teste, você pode editar o arquivo train_and_eval.py diretamente.

### Dos dados de saída
Os resultados dos testes serão salvos em arquivos CSV, cujo nome é especificado como argumento ao rodar o script run_test.sbatch.
Esses arquivos conterão métricas de desempenho do modelo treinado e avaliado.
Eles ficam na pasta results.

Na pasta out ficam arvazenados as saídas dos log dos testes realizados.

### Do ambiente
Há um arquivo chamado requirements.txt, que lista todas as dependências necessárias para executar o projeto.
O ambiente deve ser criado automaticamente ao submeter o teste na fila via sbatch (talvez seja necessário ajustar o script, na duvida rode create_env.sbatch).

### Das imagens e gráficos
O arquivo notebook plotting.ipynb contém códigos para gerar gráficos e visualizar os resultados dos testes.
Os arquivos ficam salvos na pasta figs


