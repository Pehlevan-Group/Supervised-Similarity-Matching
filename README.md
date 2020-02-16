# Supervised-Similarity-Matching
This repository contains code accompanying the Supervised Similarity Matching Equilibrium Propagation paper. The code in this repo modifies the code for Equilibrium Propagation, released by Scellier & Bengio 2017 [Equilibrium Propagation: Bridging the Gap between Energy-Based Models and Backpropagation](https://www.frontiersin.org/articles/10.3389/fncom.2017.00024/full).

EP, 'betasigned': Equilibrium Propagation without lateral connections, beta (the nudge parameter) is randomly assigned a positive or negative sign post each batch. In the other learning rules, beta is always taken to be positive.
EP, 'betapos': Equilibrium Propagation without lateral connections, beta (the nudge parameter) is positive. 
EP, Lateral: Equilibrium Propagation with lateral connections.
SMEP: Similarity Matching update for lateral connections, Equilibrium Propagation update for forward connections.

Code, result files and documentation for Supervised Similarity Matching. This directory is organized as follows:
  * Results: Contains saved network model results for all runs discussed in paper / supplementary information.
  * Structured: Code for networks with structured connectivity.
  * Main: Code for networks without structured connectivity.
  
