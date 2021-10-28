# NUTS

## The No-U-Turn Sampler

* Hamiltonian Monte Carlo (HMC)는 매우 효과적인 MCMC 알고리즘이지만 hand tuning을 필요로 한다.
* NUTS는 adaptive extension of HMC로 hand tuning의 영향을 줄일 수 있음.

## HMC

* Hybrid Monte Carlo(HMC)는 DUane et al. 1987, Neal 1993 등에서 제안되었으며 헤밀토니안 역학을 simulating하여 사후분포에서 표본을 추출
* $ -\log p(\theta|x)$ 를 potential energy로, $r$로 momentum vector with quadratic kinetic energy $r^T r/2$ 라 한 뒤 
  1. $r$을 다차원 정규분포에서 resampling (gibbs sampling step)
  2. $(\theta, r)$ 에서 L-discretized stemps of the hamiltonian dynamics of the fictitious physical system 을 통해 전진
  3. Accept or reject based on accuracy of simulation (metropolis-hastings algorithm)

Hamiltonian = Potential + Kinetic 

* Gibbs보다 더 효과적으로 표본공간을 돌아다님이 알려져 있음
* Random walk MH 보다는 훨씬 효율적

### Pros

* HMC supperesses random walk behavior, and can be much more efficient than basic MH or (Usually) Gibbs sampler, which require $O(d^2)$ steps to travel a distance $d$.
* Scales much better with dimension than basic MH or (usually) Gibbs.

### Cons

* 이산형 확률변수에는 적용 불가능 (확률밀도함수를 미분해야하므로)
* $\epsilon$, $L$ 의 튜닝이 제대로 되어있지 않으면 낮은 성능을 가짐
* 가령 $L$ 이 너무 작은 경우 모수공간을 잘 돌아다니지 못하고 너무 크면 낮은 acceptance rate, 왔다갔다 하며 계산상의 낭비

## U-Turn

HMC에서 기존의 $\theta$ 와 새로운 $\theta^\prime$ 사이의 거리가 줄어드는 때를 **U-turn** 으로 정의, 즉, 

$$\frac{d}{dt} \left[ 0.5 (\theta - \theta^\prime)^T (\theta-\theta^\prime) \right] =  \frac{d\theta^\prime}{dt} (\theta-\theta^\prime) = r^T  (\theta-\theta^\prime) < 0$$

일때를 U-turn이라 생각. 

## Overview

각 iteration은 다음과 같이 이루어짐

1. Sampling a momentum vector $r \sim \textrm{Normal}(0, 1)$ 
2. Sampling a slice variable $u \sim \textrm{Uniform([0, p(\theta, r|x)])}$
3. Tracing out the Hamiltonian dynamics of $\theta, r$ forwards and backwards in time, going forwards or backwards 1 temp, then forwards or backwards 2 steps, then 4 steps, etc. (doubling step!)
4. Dobuling stops when a sub-trajectory makes a U-turn, at which point we sample (carefully) from among the points on the trajectory

* L의 결정을 NUTS로, epsilon의 결정은 Nestrov의 dual-everaging, 0.6~0.65 정도의 acceptance rate을 갖도록
* NUTS eliminates HMC's need to twak simulation lengths
* NUTS performs as well as (or better than) HMC, even ignoring the cost of finding a good simulation length for HMC
* In conjuction with a method to automatically tune $\epsilon$, NUTS needs no hand-tuning at all.
* Core inference algorithm in Stan, which aims to be a BUGS, JAGS replacement.
