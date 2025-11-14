# Rota Inteligente: Otimização de Entregas com Algoritmos de IA
**Projeto Final - Artificial Intelligence Fundamentals**

Este projeto implementa uma solução de Inteligência Artificial para o desafio da empresa de delivery "Sabor Express", otimizando suas rotas de entrega e aumentando a eficiência operacional.

## 1. Descrição do Problema
A "Sabor Express" é uma empresa de delivery local que enfrenta desafios logísticos significativos:

* **Rotas Ineficientes:** O planejamento de rotas é manual, baseado apenas na experiência dos entregadores.
* **Altos Custos:** A ineficiência gera alto custo de combustível e atrasos frequentes.
* **Insatisfação do Cliente:** Atrasos prejudicam a reputação da empresa.

O objetivo é desenvolver uma solução de IA que sugira rotas otimizadas e agrupe entregas de forma inteligente, tornando a operação mais rápida e econômica.

## 2. Abordagem da Solução
A solução proposta ataca o problema em duas frentes principais: primeiro, agrupamos os pedidos por proximidade; segundo, roteamos o caminho mais curto para atender esses grupos.

### Modelagem do Problema: A Cidade como um Grafo
Conforme os conceitos da **Unidade 2**, modelamos a cidade como um **grafo ponderado**:

* **Nós (Vértices):** Representam a base, bairros e locais de entrega.
* **Arestas:** Representam as ruas que conectam os nós.
* **Pesos:** Representam a distância ou tempo estimado para percorrer uma aresta.

### Etapa 1: Agrupamento de Entregas com K-Means (Unidade 3)
* **O Problema:** Em horários de pico, é necessário dividir os pedidos de forma lógica entre múltiplos entregadores.
* **A Solução:** Utilizamos o **K-Means Clustering**.
* **Justificativa:** O K-Means é um algoritmo de aprendizado não supervisionado que particiona dados em *k* grupos, minimizando a variabilidade dentro de cada grupo. Isso cria "zonas de entrega" geograficamente coesas, garantindo que um entregador não precise cruzar a cidade inteira desnecessariamente.

### Etapa 2: Roteamento Otimizado com A* (Unidade 2)
* **O Problema:** Definida a zona, o entregador precisa da rota mais eficiente entre os pontos.
* **A Solução:** Utilizamos o **Algoritmo A* (A-Star)**.
* **A* vs. BFS (Busca em Largura):**
    * A **Busca em Largura (BFS)** explora o grafo nível por nível e é ideal para grafos *não ponderados*. Como nossas ruas têm custos diferentes (distâncias), o BFS não garante a rota mais econômica.
    * O **A*** é uma busca informada que utiliza uma heurística ($h(n)$) combinada com o custo real ($g(n)$) através da fórmula $f(n) = g(n) + h(n)$. Isso permite encontrar o caminho de menor custo de forma muito mais eficiente que buscas cegas.

## 3. Diagrama do Grafo (Modelo Utilizado)
O diagrama abaixo representa a topologia da cidade simulada no código:

```ascii
      (C) --5-- (E)
     / | \     /
    4  3  3   4
   /   |   \ /
 (A)   |   (D) --8-- (F) --6-- (G)
  |    |   / \     / \
  5    |  7   \   5   7
  |    | /     \ /     \
(Base)-4-(B)--3--(I)--3--(H)
  \         /
   ---9----
