# Sabor_Express
-----------------------------------------------------------------------------------------------------
# Rota Inteligente: Otimização de Entregas com Algoritmos de IA
**Projeto Final - Artificial Intelligence Fundamentals**

[cite_start]Este projeto implementa uma solução de Inteligência Artificial para o desafio da empresa de delivery "Sabor Express"[cite: 2135], otimizando suas rotas de entrega e aumentando a eficiência operacional.

## 1. Descrição do Problema

[cite_start]A "Sabor Express" é uma empresa de delivery local que enfrenta desafios logísticos significativos[cite: 2136]:
* [cite_start]**Rotas Ineficientes:** O planejamento de rotas é manual, baseado na experiência dos entregadores.
* **Altos Custos:** Ineficiência gera alto custo de combustível e atrasos.
* [cite_start]**Insatisfação do Cliente:** Atrasos frequentes prejudicam a reputação da empresa[cite: 2137].

[cite_start]O **objetivo** [cite: 2140] é desenvolver uma solução de IA que sugira rotas otimizadas e agrupe entregas de forma inteligente, tornando a operação mais rápida, econômica e confiável.

## 2. Abordagem da Solução

A solução proposta ataca o problema em duas frentes principais: primeiro, **agrupamos** os pedidos por proximidade; segundo, **roteamos** o caminho mais curto para atender esses grupos.

### Modelagem do Problema: A Cidade como um Grafo
[cite_start]Conforme o desafio [cite: 2141] [cite_start]e os conceitos da **Unidade 2**[cite: 748, 765], modelamos a cidade como um **grafo ponderado**:
* **Nós (Vértices):** Pontos de interesse (Base, bairros, locais de entrega).
* **Arestas:** As ruas que conectam os nós.
* **Pesos:** A distância ou tempo estimado para percorrer uma aresta.

### Etapa 1: Agrupamento de Entregas com K-Means (Unidade 3)

**O Problema:** Em horários de pico, é impossível para um único entregador atender todos os pedidos. Precisamos dividi-los de forma lógica entre múltiplos entregadores.

[cite_start]**A Solução:** Utilizamos o **K-Means Clustering**, um conceito fundamental da **Unidade 3**[cite: 7].
* [cite_start]**Por que K-Means?** K-Means é um algoritmo de aprendizado **não supervisionado** [cite: 136] ideal para esta tarefa. [cite_start]Seu objetivo é "particiona[r] um conjunto de dados em k grupos, minimizando a variabilidade intra-grupo" [cite: 184-186].
* [cite_start]**Aplicação:** Ao definir $k$ como o número de entregadores disponíveis, o K-Means identifica $k$ "centróides" [cite: 188] e agrupa os pedidos (nós) com base em sua proximidade geográfica (usando suas coordenadas). [cite_start]Isso cria "zonas de entrega" [cite: 2147] coesas e eficientes para cada entregador.

### Etapa 2: Roteamento Otimizado com A* (Unidade 2)

**O Problema:** Uma vez que um entregador tem sua "zona", ele ainda precisa saber a rota mais eficiente (de menor custo) entre a Base e os pontos de entrega.

**A Solução:** Utilizamos o **Algoritmo A* (A-Star)**, uma técnica de busca avançada da **Unidade 2**[cite: 696].

**Por que A* e não BFS (Busca em Largura)?**
* [cite_start]A **Busca em Largura (BFS)**, conforme a **Unidade 2**[cite: 651], é uma busca "cega" (não informada). [cite_start]Ela "encontra caminhos mais curtos em grafos *não ponderados*"[cite: 733]. Nosso grafo *é* ponderado (as ruas têm distâncias diferentes), então o BFS não garantiria a rota de menor custo.
* [cite_start]O **A*** é uma **busca informada (heurística)**[cite: 630, 699]. [cite_start]Ele combina o custo real já percorrido (função $g(n)$) com uma "heurística admissível" (função $h(n)$) — uma estimativa inteligente que nunca superestima o custo restante[cite: 700].
* [cite_start]Ao usar a função $f(n) = g(n) + h(n)$[cite: 700], o A* explora de forma inteligente apenas os caminhos mais promissores, garantindo a rota de menor custo (ótima) de forma muito mais eficiente que algoritmos de busca cega.

## 3. Diagrama do Grafo (Modelo Fictício)

O grafo a seguir (baseado nos dados do código) ilustra a estrutura da cidade:

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
