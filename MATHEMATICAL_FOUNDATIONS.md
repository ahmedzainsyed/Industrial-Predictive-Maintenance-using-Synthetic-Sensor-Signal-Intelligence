# Research Mathematical Foundations
## Industrial Predictive Maintenance — Signal Processing & AI

---

## 1. Spectral Analysis

### 1.1 Discrete Fourier Transform

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j2\pi kn/N}, \quad k = 0,1,\ldots,N-1$$

The inverse DFT:
$$x[n] = \frac{1}{N}\sum_{k=0}^{N-1} X[k] \cdot e^{j2\pi kn/N}$$

Frequency resolution: $\Delta f = f_s / N$ Hz

### 1.2 Welch Power Spectral Density

For $K$ overlapping segments of length $M$ with window $w[n]$:

$$\hat{S}_{xx}(f) = \frac{1}{K \cdot U} \sum_{k=0}^{K-1} \left| \sum_{n=0}^{M-1} x_k[n] \cdot w[n] \cdot e^{-j2\pi fn} \right|^2$$

where $U = \frac{1}{M}\sum_{n=0}^{M-1} w^2[n]$ is the window normalization factor.

### 1.3 Short-Time Fourier Transform

$$\mathcal{X}(\tau, \omega) = \int_{-\infty}^{\infty} x(t) \cdot w(t-\tau) \cdot e^{-j\omega t} \, dt$$

Discrete STFT:
$$X[m, k] = \sum_{n=0}^{N-1} x[n + mH] \cdot w[n] \cdot e^{-j2\pi kn/N}$$

where $H$ = hop size, $m$ = frame index.

### 1.4 Spectral Entropy

Normalize PSD to a probability distribution:
$$p(f_k) = \frac{S(f_k)}{\sum_{k} S(f_k)}$$

Shannon spectral entropy:
$$H_s = -\sum_{k} p(f_k) \log_2 p(f_k)$$

**Interpretation:**
- $H_s \to 0$: Energy concentrated at few frequencies (fault-like)
- $H_s \to \log_2 N/2$: Energy spread uniformly (noise-like)

### 1.5 Spectral Kurtosis

$$K(f) = \frac{\langle |H(t,f)|^4 \rangle}{\langle |H(t,f)|^2 \rangle^2} - 2$$

where $H(t,f)$ is the STFT at time $t$, frequency $f$.
High $K(f)$ indicates impulsive content at frequency $f$ — a bearing fault signature.

---

## 2. Wavelet Transform Mathematics

### 2.1 Continuous Wavelet Transform

$$W_\psi(a, b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} x(t) \cdot \psi^*\!\left(\frac{t-b}{a}\right) dt$$

**Morlet wavelet** (optimal time-frequency resolution):
$$\psi(t) = \pi^{-1/4} \left(e^{j\omega_0 t} - e^{-\omega_0^2/2}\right) e^{-t^2/2}$$

Frequency-scale relationship: $f = \frac{f_\psi}{a \cdot \Delta t}$ where $f_\psi$ is the center frequency.

### 2.2 Discrete Wavelet Transform (Mallat Algorithm)

Decomposition (analysis filter bank):
$$\text{cA}_j[n] = \sum_{k} h[k - 2n] \cdot \text{cA}_{j-1}[k] \quad \text{(approximation)}$$
$$\text{cD}_j[n] = \sum_{k} g[k - 2n] \cdot \text{cA}_{j-1}[k] \quad \text{(detail)}$$

where $h[n]$ = low-pass filter, $g[n]$ = high-pass filter, $g[n] = (-1)^n h[L-1-n]$ (quadrature mirror).

Reconstruction (synthesis):
$$\text{cA}_{j-1}[n] = \sum_{k} \tilde{h}[n-2k] \cdot \text{cA}_j[k] + \sum_{k} \tilde{g}[n-2k] \cdot \text{cD}_j[k]$$

**Daubechies-8** has 8 vanishing moments:
$$\int_{-\infty}^{\infty} t^k \psi(t) \, dt = 0, \quad k = 0,1,\ldots,7$$

### 2.3 Wavelet Energy and Entropy

Energy at level $j$:
$$E_j = \sum_{n} |\text{cD}_j[n]|^2$$

Normalized energy ratios:
$$p_j = \frac{E_j}{\sum_{j} E_j}$$

**Wavelet Shannon entropy:**
$$WE = -\sum_{j} p_j \ln p_j$$

### 2.4 Wavelet Denoising — BayesShrink Threshold

For level $j$ with noise estimate $\hat{\sigma}_j = \text{median}(|d_j|)/0.6745$:

$$\hat{\sigma}_{s,j}^2 = \max\left(0, \frac{\sum_n d_j^2[n] / N_j - \hat{\sigma}_j^2}{1}\right)$$

$$\lambda_j^{\text{Bayes}} = \frac{\hat{\sigma}_j^2}{\hat{\sigma}_{s,j}}$$

Apply soft thresholding:
$$\hat{d}_j[n] = \text{sign}(d_j[n]) \cdot \max(0, |d_j[n]| - \lambda_j)$$

---

## 3. Deep Learning Models

### 3.1 LSTM Cell Equations

$$\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) \quad &\text{(forget gate)} \\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) \quad &\text{(input gate)} \\
\tilde{C}_t &= \tanh(W_C [h_{t-1}, x_t] + b_C) \quad &\text{(candidate)} \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad &\text{(cell state)} \\
o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) \quad &\text{(output gate)} \\
h_t &= o_t \odot \tanh(C_t) \quad &\text{(hidden state)}
\end{aligned}$$

### 3.2 Transformer Self-Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

Multi-head attention:
$$\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h) W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### 3.3 NASA C-MAPSS Scoring Function

$$d_i = \hat{y}_i - y_i \quad \text{(prediction error)}$$

$$s_i = \begin{cases} e^{-d_i/13} - 1 & \text{if } d_i < 0 \text{ (late prediction)} \\ e^{d_i/10} - 1 & \text{if } d_i \geq 0 \text{ (early prediction)} \end{cases}$$

$$\text{Score} = \sum_{i=1}^{n} e^{s_i}$$

The asymmetry penalizes late predictions more severely, reflecting that under-predicting RUL is more dangerous in industrial practice.

### 3.4 Variational Autoencoder (LSTM-VAE) ELBO

$$\mathcal{L}(\theta, \phi; x) = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{Reconstruction}} - \underbrace{\beta \cdot D_{KL}(q_\phi(z|x) \| p(z))}_{\text{Regularization}}$$

KL divergence (diagonal Gaussian):
$$D_{KL}(q \| p) = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

Reparameterization trick:
$$z = \mu + \epsilon \odot \sigma, \quad \epsilon \sim \mathcal{N}(0, I)$$

---

## 4. Bayesian Uncertainty Quantification

### 4.1 Monte Carlo Dropout

Approximate predictive distribution via $T$ stochastic forward passes:

$$p(y^* | x^*, \mathcal{D}) \approx \frac{1}{T} \sum_{t=1}^{T} p(y^* | x^*, \hat{\omega}_t)$$

Predictive mean and variance:
$$\mathbb{E}[y^*] \approx \frac{1}{T} \sum_t \hat{y}_t$$

$$\text{Var}[y^*] \approx \underbrace{\sigma^2}_{\text{aleatoric}} + \underbrace{\frac{1}{T}\sum_t \hat{y}_t^2 - \left(\frac{1}{T}\sum_t \hat{y}_t\right)^2}_{\text{epistemic}}$$

### 4.2 Conformal Prediction

Given calibration set $\{(x_i, y_i)\}_{i=1}^n$, nonconformity scores:
$$s_i = 1 - \hat{p}(y_i | x_i)$$

Quantile: $\hat{q} = \text{Quantile}\left(\{s_i\}, \frac{\lceil(n+1)(1-\alpha)\rceil}{n}\right)$

Prediction set: $\mathcal{C}(x) = \{y : 1 - \hat{p}(y|x) \leq \hat{q}\}$

**Coverage guarantee:** $\mathbb{P}(Y_{n+1} \in \mathcal{C}(X_{n+1})) \geq 1 - \alpha$

---

## 5. Reliability Engineering Mathematics

### 5.1 Weibull Distribution

CDF (probability of failure by time $t$):
$$F(t) = 1 - e^{-(t/\eta)^\beta}$$

Reliability function:
$$R(t) = e^{-(t/\eta)^\beta}$$

Hazard rate (instantaneous failure rate):
$$h(t) = \frac{\beta}{\eta}\left(\frac{t}{\eta}\right)^{\beta-1}$$

Mean Time To Failure:
$$\text{MTTF} = \eta \cdot \Gamma\!\left(1 + \frac{1}{\beta}\right)$$

**Shape parameter interpretation:**
- $\beta < 1$: Decreasing failure rate (infant mortality)
- $\beta = 1$: Constant failure rate (exponential = random failures)
- $\beta > 1$: Increasing failure rate (wear-out)
- $\beta \approx 3.5$: Approximately normal (typical bearing wear-out)

### 5.2 B10/B50 Life

$B_x$ life = time at which $x\%$ of a population will have failed:

$$B_x = \eta \cdot \left(-\ln\left(1 - \frac{x}{100}\right)\right)^{1/\beta}$$

### 5.3 Maintenance Optimization

Expected cost per unit time:
$$C(T) = \frac{C_p \cdot R(T) + C_f \cdot [1 - R(T)]}{T_p \cdot R(T) + T_f \cdot [1 - R(T)]} \cdot \frac{1}{T}$$

Optimal preventive maintenance interval $T^*$ minimizes $C(T)$.

---

## 6. Population Stability Index (PSI) Drift Detection

Bins features into $B$ equal-frequency buckets:

$$\text{PSI} = \sum_{b=1}^{B} \left(A_b - E_b\right) \cdot \ln\left(\frac{A_b}{E_b}\right)$$

where $A_b$ = fraction of current population in bin $b$,  
$E_b$ = fraction of reference population in bin $b$.

| PSI | Interpretation |
|-----|---------------|
| < 0.10 | No significant change |
| 0.10–0.20 | Minor change, monitor |
| > 0.20 | Major change, **retrain** |

---

*All equations implemented in production code — see source files for full implementations.*
