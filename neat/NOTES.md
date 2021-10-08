# A bag for some of my ideas and scratch notes

S - search space (phenotype)
N - navigation space (genotype)
D - distribution on N
F: D -> D  - distribution shift
f: N -> S  - search mapping
K(S) = f(D(N)) - distribution on S

At a high level, it's all about finding the right representation
of distribution on S. Then implement the right mutation mechanism via
F. Iterative application of F should move the distribution in a way that
eventually explores every corner of S.

The phenotype does not have to be any physical thing. For example, you might
take S to be the space of all human bodies. Then you can try to come up with
F that will explore all possible human bodies. That is the way most people
think of phenotype. But actually it can be something much more abstract or general.
We might take S to be the space of all places on earth. Then we will try to create
evolutionary system that could eventually visit every natural niche, like oceans, mountains, caves,
plains, air (or even outer space, just like humans have explored it). Another example
would be to search the space of gameplay strategies in chess.

The genotype and phenotype need not be different. In fact we could have defined evolution purely
by means of S and K. For example let S be a maze and all the places in it expressed as X and Y coordinates.
We could define K directly on 2D plane and ignore the walls of the maze. But then the search problem
would become trivial. It's much more interesting when you take N to be the space of paths in the maze,
and define D on those paths. Then the problem of exploring S via means of D is more challenging.
It becomes even more difficult when you define D to be a neural network and try to explore the space of
paths indirectly via F on those neural nets. In fact it leads us to the notion of search space composition.

Let S1 be some search space and S2 be some navigation space with distribution D2 on S2 and
F2: D2 -> D2. Similarly let S3 another navigation space with distribution D3 and
F3: D3 -> D3. Given mapping f2 : S2 -> S1 and f3 : S3 -> S2 we can define a composed search
mapping f2∘f3 : S3 -> S1 . Thus we can search S1 by navigating in S3

Sometimes we want to explore some space, but we don't necessarily need to explore all of it.
If we defined the search problem as visiting every element of S in the limit, then we would just have
an exhaustive search. Instead the search problem must be biased towards those places that are
interesting to us. The key issue is that very often in practice we don't know how to define
"interestingness" (in fact, if we could formally define it, then there would be no search problem anymore).
Hence some proxy concept for interestingness is required. In evolution, this concept is captured by survival.
Things that die are not very interesting. The primary feature of all living systems is their
self-organisation property, that is, all living creatures seek to minimise the entropy of their internal
systems. Maintaining low entropy in static environments is easy, but the more dangerous it becomes
and the scarcer the resources are, the more complex organisms need to evolve in order to put up resistance.

Hence interestingness should be defined by the proxy of another distribution.
E - the distribution on S
We try to design F in such a way that after enough iterations E=K.
... ???

The more random the mutations become the more difficult it is to maintain stability in population.
As a result, even if we find an optimal solution, we might struggle to maintain it for long.
On top of that, if we keep increasing the randomness of system, then at a certain point,
we will end up with purely random search. On the other hand, if we decrease randomness
to bare minimum, we will end up with plain old gradient-descent.
These were the problems plaguing evolutionary algorithms when applied to optimisation problems.
But here, we are not trying to do that. Here we investigate evolutionary algorithms
as applied to search problems. Optimisation and search are two opposites, analogically to the
exploration & exploitation dilemma in reinforcement learning.

Imagine we are trying to navigate a maze. The difficulty lies in the fact that
path configurations do not translate easily to final 2D coordinates. If we could find
path representation such that small changes yield predictable changes in final results,
then we could indeed use gradient descent to solve any maze.
But the problem is that mapping from maze path to final coordinate is intractable for any
more complicated maze. We can formalise it as follows

d_S - distance metric on search space
d_N - distance metric on navigation space
E[d_S(f(n))]
