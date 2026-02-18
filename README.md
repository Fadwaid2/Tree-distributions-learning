# Learning-Tree-distributions
Python repository for the paper "Learning Tree-structured Distributions". It contains the implementation of the two following Online Learning algorithms:


1. **[Online Forest Density Estimation](https://www.auai.org/uai2016/proceedings/papers/116.pdf)**
   Frédéric Koriche. *UAI 2016*.

2. **[Distribution Learning Meets Graph Structure Sampling](https://arxiv.org/abs/2405.07914)**
   Arnab Bhattacharyya, Sutanu Gayen, Philips George John, Sayantan Sen, and N. V. Vinodchandran. *arXiv 2024*.

3. A variant of (2) **[Distribution Learning Meets Graph Structure Sampling](https://arxiv.org/abs/2405.07914)**, that uses Loop-erased random walk - Wilson's sampling algorithm.

It also includes an implementation of the classic Chow-Liu algorithm and evaluation metrics that were used in the paper, specifically, log-likelihood and structural hamming distance (SHD) computations.

---

## Methods

Each algorithm has two versions: a standard version for synthetic/small datasets and a fast vectorized version for large real-world datasets.

| `--method` | Description |
|---|---|
| `Chow-Liu` | Offline maximum-weight spanning tree via mutual information (Kruskal, descending sort) | 
| `OFDE` | Online Forest Density Estimation. FPL + Kruskal (ascending/argmin) + iterative swap rounding. Precomputes cumulative counts and phi for all T steps. |
| `OFDEFast` | Same algorithm as OFDE but with incremental vectorized phi. | 
| `RWM` | Randomized Weighted Majority with arborescence sampler. Python-loop DPT precomputation. | 
| `RWMFast` | Same algorithm as RWM but with vectorized batch DPT precomputation (log-space cumsum, no T×n²×k² loops) and index-based weight access. | 
| `RWM_Wilson` | RWM variant using Wilson's loop-erased random walk for spanning tree sampling instead of the arborescence sampler. Same Python-loop DPT as RWM. | 
| `RWM_WilsonFast` | Same as RWM_Wilson but with numba-compiled Wilson sampler (uniform fallback for stuck walks) + vectorized DPT + pre-stacked (n,n,T+1) weight matrix for O(1) updates. | 

---

## Usage

### Generate synthetic data and run a method

```bash
python run.py \
  --synthetic \
  --tree \
  --n 10 \
  --T 500 \
  --k 2 \
  --seed 42 \
  --noise 0.3 \
  --method OFDE \
  --output_folder results/ofde_synthetic
```

`--synthetic` generates train/test data on the fly instead of loading CSVs.
`--tree` samples from a random tree graph (omit for an Erdos-Renyi random graph).

### Load a real dataset and run a fast method

```bash
python run.py \
  --train_data path/to/train.csv \
  --test_data path/to/test.csv \
  --true_graph path/to/true_graph.txt \
  --n 50 \
  --T 1000 \
  --k 2 \
  --method OFDEFast \
  --output_folder results/ofde_fast
```

### Run RWM with Wilson's sampler (fast version)

```bash
python run.py \
  --train_data path/to/train.csv \
  --test_data path/to/test.csv \
  --true_graph path/to/true_graph.txt \
  --n 50 \
  --T 1000 \
  --k 2 \
  --method RWM_WilsonFast \
  --epsilon 0.9 \
  --output_folder results/rwm_wilson_fast
```

### Output

Results are saved to `--output_folder`:
- `results.json` — `log-likelihood` and `shd` (structural hamming distance)
- `arguments.json` — full record of all arguments used
