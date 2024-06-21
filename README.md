Inspired by [Machine learning for modular multiplication](https://arxiv.org/abs/2402.19254) we investigate the ability of GNN's to learn multiplication modulo primes $p$.
More precisely, let $p$ be a prime and $s \in \mathbb{Z} / (p)$. Then we want to see how effective a neural network can learn to compute $as \pmod{p}$ for a given positive integer $a$. Previous literature suggests
that the answer is not very well, the linked paper discussess this issue in more depth. 

Modular multiplication show epicycloid patterns when viewed as a graph [as shown by Nathaniel Wroblewski](https://www.nathaniel.ai/modular-multiplication/). Perhaps 
if we used a graph neural network, we could learn whether integer pairs $(a,b)$ follow the relation $b=as \pmod{p}$. for a fixed integer $s$. In more detail, we consider a graph
with vertices given by the classes of $\mathbb{Z}/(p)$. The edges in the graph will be given by some number of pairs $(a,b)$ such that $b = as \pmod{p}$. Importantly, we do not include
all such pairs $(a,b)$ as we want to predict later on whether some unseen pair $(a',b')$ follows $b' = a's \pmod{p}$. This is the task of link prediction, and we utilize the PyTorch Geometric library
to do this. Unfortunately, this attempt is unsuccessful, following the previous attempts described in the paper linked above.
