# Python - aprendizado de máquina

## Exemplo: Máquina de Boltzmann Restrita (RBM)

Este repositório agora inclui um exemplo completo de **Máquina de Boltzmann Restrita** em Python puro, sem dependências externas.

### Arquivo
- `boltzmann_machine.py`: implementação da RBM com treinamento por **Contrastive Divergence (CD-1)**.

### Como executar
```bash
python3 boltzmann_machine.py
```

O script:
1. treina a RBM em um pequeno conjunto de dados binários;
2. imprime o erro médio de reconstrução ao longo das épocas;
3. mostra os pesos aprendidos;
4. reconstrói um exemplo de entrada.
