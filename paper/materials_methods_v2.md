# 2. Materials and Methods

> **Draft v2.1 — status legend.** Values implemented in `ctnet_pd/` (configuration `ctnet_pd/configs/default.yaml`) are now written in the text. `[TBD: …]` marks values that still depend on the pilot runs. `[VERIFY: …]` marks statements taken from the preliminary conference paper that must be checked against the corpus metadata or the code before submission. Remove this box and all markers before submission.

## 2.1. Study Design and Relation to the Preliminary Study

This study extends our preliminary conference work on attention maps for CTNet-based Parkinson's disease (PD) classification [Patiño-Bedoya et al., LNCS, TBD]. The extended study answers four questions:

- **Q1 — Prediction.** Does CTNet discriminate PD from healthy controls (HC) at the subject level under speaker-disjoint and cross-cohort validation, compared with baselines retrained under the same protocol?
- **Q2 — Location of evidence.** Is the evidence used by the network located consistently before and after the self-attention stage?
- **Q3 — Validity of explanations.** Are the explanations faithful (they change the prediction when perturbed), sensitive to model parameters and labels (sanity checks), and reproducible across independent cohorts beyond what generic spectral structure explains?
- **Q4 — Acoustic prior.** Does softly aligning the explanation with acoustic priors defined independently of the model improve explanation validity without degrading prediction or introducing demographic dependencies?

Table 0 lists the methodological changes with respect to the preliminary study. Because of these changes, the performance values reported in the preliminary study are **not comparable** with those reported here.

**Table 0.** Changes with respect to the preliminary study.

| Aspect | Preliminary study | This study |
|---|---|---|
| Data partitioning | Stratified 5-fold over recordings | Speaker-disjoint, repeated stratified group 5-fold, nested |
| Hyperparameter selection | Per fold, final evaluation on the same folds | Inner subject-disjoint validation only |
| Classification head | Flatten (22,400) → Dense(128) → Dense(2) | Global average pooling → Linear(64→2), 130 parameters |
| Input length | Fixed per corpus (661 frames in NeuroVoz); short recordings stretched by spectral interpolation, long ones truncated | Overlapping 229-frame windows; no stretching or truncation (Section 2.5) |
| Positional encoding | None | Learned (Eq. 2) |
| Baselines | Values quoted from studies with other protocols | All retrained under the same folds and information boundary |
| Explanations | Qualitative LRP, Grad-CAM and Score-CAM maps | Pre-/post-attention maps; deletion/insertion, ROAR, sanity checks, cross-cohort tests against null references |
| Evaluation unit | Recording | Subject |
| External validation | None (qualitative comparison between cohorts) | Two-way, task-matched |
| Acoustic prior | None | Optional soft regularizer, accepted only under pre-specified criteria (Section 2.10.4) |

---

## 2.2. Speech Corpora

Two independent Spanish speech corpora were used (Table 1).

**PC-GITA** [Orozco-Arroyave et al., 2014] contains recordings of 50 PD patients and 50 HC, all native Colombian Spanish speakers, balanced by sex and age. PD patients were recorded in the ON-medication state. This study used:

- diadochokinetic (DDK) tasks: /pa-ta-ka/, /pe-ta-ka/, /pa-ka-ta/, /pa/, /ta/ and /ka/;
- a read text of 36 words;
- a spontaneous monologue.

`[VERIFY: whether sustained vowels, isolated words and sentences of PC-GITA were excluded and why.]`

**NeuroVoz** [Mendes-Laureano et al., 2024] contains 3,010 recordings (26.88 ± 3.35 per participant) `[VERIFY: the Kaggle copy used in the preliminary study indexes 2,976 WAV files (1,509 HC, 1,467 PD)]` of 112 native Castilian Spanish speakers (54 PD, 58 HC), all PD patients in the ON state. Tasks:

- sustained vowels /a/, /e/, /i/, /o/, /u/;
- DDK /pa-ta-ka/;
- 16 listen-and-repeat utterances;
- a spontaneous monologue.

Metadata include age, sex, disease duration, MDS-UPDRS-III, Hoehn & Yahr stage, symptoms and treatment.

`[VERIFY: recording conditions of NeuroVoz (room, microphone). The preliminary paper states a sound-proof booth at 44.1 kHz/16 bit for both corpora.]`

**Table 1.** Demographic and clinical characteristics.

| | PC-GITA PD | PC-GITA HC | NeuroVoz PD | NeuroVoz HC |
|---|---|---|---|---|
| Subjects (M/F) | 25/25 | 25/25 | `[TBD]` | `[TBD]` |
| Age, M, mean (SD) | 61.3 (11.4) | 60.5 (11.6) | `[TBD]` | `[TBD]` |
| Age, F, mean (SD) | 60.7 (7.3) | 61.4 (7.0) | `[TBD]` | `[TBD]` |
| Disease duration, years, mean (SD) | `[TBD]` | – | `[TBD]` | – |
| MDS-UPDRS-III, mean (SD) | `[TBD]` | – | `[TBD]` | – |
| Hoehn & Yahr, median (range) | `[TBD]` | – | `[TBD]` | – |

`[VERIFY: Table 1 of the preliminary paper is internally inconsistent for NeuroVoz. Its counts (33+20 PD, 28+26 HC = 107) differ from the 54/58 stated in the text. Its disease-duration and UPDRS values are identical for men and women. Recompute all cells directly from the corpus metadata, and test the PD–HC age and sex differences (Welch t / χ²), because they define the demographic-confounding risk.]`

**Ethics.** Both corpora were collected under the approval of their institutional ethics committees with written informed consent [cite the corpus papers]. This study is a secondary analysis of de-identified data. `[TBD: MDPI Institutional Review Board, Informed Consent and Data Availability statements; PC-GITA access is granted under a license agreement.]`

**Statistical unit.** Each participant contributes many recordings. The **subject** is therefore the independent unit for partitioning, aggregation, resampling and inference throughout.

---

## 2.3. Speech Tasks and Task-Matched Subsets

Mixing tasks with very different acoustic content (sustained vowels versus monologues) in one model makes the decision depend on how many recordings of each task a subject contributes. Analyses were therefore organized by task family (Table 2).

**Table 2.** Task families and their use in each experiment.

| Task family | PC-GITA | NeuroVoz | Within-cohort (Exp. I) | External, primary (Exp. III) |
|---|---|---|---|---|
| DDK /pa-ta-ka/ | ✓ | ✓ | ✓ | ✓ |
| Other DDK (/pe-ta-ka/, /pa-ka-ta/, /pa/, /ta/, /ka/) | ✓ | – | ✓ | – |
| Monologue | ✓ | ✓ | ✓ | ✓ |
| Read text / listen-and-repeat | ✓ / – | – / ✓ | ✓ | – (not equivalent) |
| Sustained vowels | `[VERIFY]` | ✓ | ✓ (NeuroVoz) | – |

The **primary** analyses were run per task family on the two shared families: /pa-ta-ka/ and monologue. A model pooling all available tasks was reported as a **secondary**, more heterogeneous analysis, using task-balanced aggregation (Section 2.7).

---

## 2.4. Subject-Independent Partitioning

All recordings of a subject were assigned to a single partition, so that $\mathcal S_{\text{train}}$, $\mathcal S_{\text{val}}$ and $\mathcal S_{\text{test}}$ are pairwise disjoint in every split.

- **Outer loop.** Stratified group 5-fold cross-validation (stratified by diagnosis, grouped by subject), **repeated 10 times** with different seeds. With 100–112 subjects, each test fold holds about 20 subjects, so repetition is needed to stabilize the estimates.
- **Inner loop.** Within each outer training set, a stratified, subject-disjoint hold-out of 20% of the training subjects was used for hyperparameter selection, checkpoint selection, early stopping and the regularization weights. A single inner split keeps the 50 outer folds computationally feasible; an inner 4-fold variant is available in the code (`cv.inner = kfold`). The outer test fold was never accessed during model selection.
- **Leakage boundary.** Normalization statistics (Section 2.5) and acoustic-concept standardization (Section 2.10.3) were computed on the corresponding training partition only.

---

## 2.5. Preprocessing and Log-Mel Representation

1. **Curation.** Recordings were manually curated to remove coughs and artifacts. `[VERIFY: this step is not part of the preliminary code; state whether it was done and how.]`
2. **Resampling.** Signals were loaded as mono and resampled from 44.1 kHz to 22.05 kHz (librosa).
3. **Silence trimming.** Leading and trailing silences were removed with an energy threshold of 30 dB below the recording peak (`librosa.effects.trim`). `[TBD: confirm the threshold on pilot data.]`
4. **Amplitude normalization.** Each recording was peak-normalized to $[-1,1]$.
5. **Log-Mel spectrogram.** Each signal $\tilde x$ was converted into
$$
X(f,t)=\max\Big(10\log_{10}\frac{E(f,t)}{\max_{f',t'}E(f',t')},\;-80\Big),\qquad E(f,t)=\sum_{k} B_f(k)\,\big|\mathrm{STFT}(\tilde x)(k,t)\big|^2, \tag{1}
$$
for $f=1,\dots,128$, where $B_f$ are triangular Mel filters spanning 0–11,025 Hz. The STFT used a 2,048-sample Hann window (92.9 ms) with a hop of 512 samples (23.2 ms) and centred frames. The decibel reference is the maximum of each recording, so the level is normalized per recording and the dynamic range is limited to 80 dB.
6. **Fixed-length segments.** The operator $\mathcal R$ cuts each spectrogram $\mathbb R^{128\times T}$ into windows of 229 frames (5.3 s) with a hop of 115 frames (50% overlap). A final partial window is kept, right-padded with the recording minimum (−80 dB, i.e., silence), only if at least 50% of it is real signal. A recording shorter than one window yields a single padded window, so no recording is discarded. Recordings are never stretched or truncated. In the preliminary study, short recordings were stretched by spectral interpolation (`ifft(fft(x), n)`), which alters $F_0$ and speech rate. Recording- and subject-level probabilities were obtained by aggregating segment probabilities (Section 2.7).
7. **Input standardization.** Each Mel band was standardized with the mean and SD computed on the training segments of the corresponding split.

**Table 3.** Preprocessing parameters.

| Parameter | Value |
|---|---|
| Sampling rate | 22,050 Hz |
| Silence trimming | leading/trailing, 30 dB below peak `[TBD: confirm]` |
| Amplitude normalization | peak, per recording |
| STFT window / length / hop | Hann / 2,048 / 512 samples (92.9 / 23.2 ms), centred |
| Mel bands / range | 128 / 0–11,025 Hz |
| Log compression | dB re recording maximum, floor −80 dB |
| Segment length | 229 frames (5.3 s) |
| Segment hop | 115 frames (50% overlap) |
| Padding | recording minimum; partial window kept if ≥ 50% real |
| Input standardization | per Mel band, training-partition mean and SD |

---

## 2.6. CTNet Architecture

CTNet maps a log-Mel segment $X\in\mathbb R^{128\times229}$ to class probabilities in four steps (Figure 1, Table 4):

1. **Convolutional stem.** Two blocks, each a 3×3 convolution with 64 filters and same padding, batch normalization, a nonlinearity (GELU or ReLU, tuned) and 3×3 average pooling with stride 3, produce $C\in\mathbb R^{14\times 25\times 64}$.
2. **Tokenization.** Reshaping $C$ gives $N_{\text{tok}}=14\times25=350$ tokens of dimension $d=64$, and a learned positional embedding $E_{\text{pos}}\in\mathbb R^{350\times64}$ is added. The preliminary model had no positional encoding, which makes self-attention invariant to the order of the tokens.
$$
H^{(0)}=\mathcal V(C)+E_{\text{pos}}\in\mathbb R^{350\times 64}. \tag{2}
$$
3. **Transformer encoder.** $L=1$ post-norm encoder layer (as in the preliminary study) with multi-head self-attention ($H$ heads, key dimension $d/H$), a two-layer feed-forward network, dropout, residual connections and layer normalization produce $H^{(L)}=[h_1,\dots,h_{350}]^\top$, with $h_i\in\mathbb R^{64}$.
4. **Classification head.** Global average pooling followed by a linear layer:
$$
z=\frac{1}{350}\sum_{i=1}^{350}h_i\in\mathbb R^{64},\qquad o_c=w_c^\top z+b_c,\qquad \hat p(c\mid X)=\operatorname{softmax}(o)_c, \tag{3}
$$
with $W\in\mathbb R^{2\times64}$. The head therefore has $64\times2+2=130$ parameters.

This head replaces the flatten–dense head of the preliminary study, which had about 2.87 M parameters and was disproportionate to the sample size. It also yields an exact token-level decomposition of the logit (Section 2.8).

**Architecture-defining hyperparameters were fixed:** two convolutional blocks and $d=64$. All folds therefore share the same $14\times25$ token grid, which the explanation analyses require.

**Table 4.** CTNet configuration.

| Block | Output | Specification |
|---|---|---|
| Input (log-Mel) | 128×229×1 | Section 2.5 |
| Conv block 1 | 42×76×64 | Conv 3×3 (64) – BN – GELU/ReLU – AvgPool 3×3 |
| Conv block 2 | 14×25×64 | Conv 3×3 (64) – BN – GELU/ReLU – AvgPool 3×3 |
| Tokens + positional encoding | 350×64 | Learned embedding (22,400 parameters) |
| Transformer encoder | 350×64 | $L=1$, post-norm; heads $\in\{1,2,4\}$, FFN $\in\{64,128,256\}$, dropout $\in\{0.1,0.2,0.3\}$ (tuned) |
| Global average pooling | 64 | – |
| Linear + softmax | 2 | 130 parameters |
| **Total parameters** | | 85,826–110,594 depending on the FFN width (94,082 with FFN = 128); 256 of them are non-trainable BN statistics |

---

## 2.7. Subject-Level Aggregation

Let $\hat p_{s r j}$ be the PD probability of segment $j$ of recording $r$ of subject $s$.

- **Recording level.** $\hat p_{sr}$ is the mean of $\hat p_{srj}$ over the recording's segments.
- **Subject level (primary, single task family).** $\hat p_s$ is the mean of $\hat p_{sr}$ over the subject's recordings.
- **Subject level (all-task analysis).** Recordings are first averaged within each task family, and the family means are then averaged with equal weight.

The decision was $\hat y_s=\mathbb 1[\hat p_s\ge\tau]$, with $\tau=0.5$ (Section 2.13 gives the external-validation thresholds). The same aggregation was applied to every model.

---

## 2.8. Pre- and Post-Attention Relevance Maps

All maps were computed on the **native $14\times25$ token grid**. Upsampling to $128\times229$ was used only for visualization and to build input-space perturbation masks (Section 2.11).

**Explained quantity.** For class $c$ and its complement $\bar c$, the logit contrast $o^{(c)}_\delta=o_c-o_{\bar c}$ was explained. At test time $c$ is the predicted class; in the prior loss (Section 2.10) it is the true class.

**Pre-attention map (Grad-CAM on the convolutional stem)** [Selvaraju et al., 2017]:
$$
\alpha_u^{(c)}=\frac{1}{350}\sum_{i,j}\frac{\partial o^{(c)}_\delta}{\partial C_{iju}},\qquad
R_{\text{pre}}^{(c)}=\operatorname{ReLU}\Big(\sum_{u=1}^{64}\alpha_u^{(c)}C_{::u}\Big)\in\mathbb R_{\ge0}^{14\times25}. \tag{4}
$$
The gradient flows through the Transformer, so $R_{\text{pre}}$ is the relevance of the convolutional representation **for the whole network**, not of a separate branch.

**Post-attention map (exact token contribution / CAM)** [Zhou et al., 2016]. Under Eq. (3), the logit contrast decomposes exactly over tokens:
$$
o^{(c)}_\delta-(b_c-b_{\bar c})=\sum_{i=1}^{350}\phi_i^{(c)},\qquad \phi_i^{(c)}=\tfrac{1}{350}(w_c-w_{\bar c})^\top h_i, \tag{5}
$$
$$
R_{\text{post}}^{(c)}=\operatorname{ReLU}\big(\mathcal V^{-1}(\phi^{(c)})\big)\in\mathbb R_{\ge0}^{14\times25}.
$$
After attention, each token mixes information from other positions. $R_{\text{post}}$ is therefore anchored at each token's stem position but is not strictly local, which is why its validity is tested empirically (Section 2.11).

**Joint map.** With the sum-normalization $\mathcal N_1(R)=(R+\epsilon)/\sum_{i,j}(R_{ij}+\epsilon)$:
$$
q_{\text{pre}}=\mathcal N_1(R_{\text{pre}}),\qquad q_{\text{post}}=\mathcal N_1(R_{\text{post}}),\qquad
\boxed{\,q_J=\tfrac12\,(q_{\text{pre}}+q_{\text{post}})\,} \tag{6}
$$
$q_J$ is a probability distribution over the 350 cells. The fusion has no learnable parameters, so it cannot be trained to mimic the prior independently of the network. For display, min–max scaled versions $M_{\text{pre}}, M_{\text{post}}, M_J\in[0,1]^{128\times229}$ were used.

**Computational note.** Using $q_{\text{pre}}$ inside a loss requires second-order gradients (double backpropagation). $q_{\text{post}}$ is first order. Training time is reported for every variant.

---

## 2.9. Comparator Explanation Methods

- **LRP for Transformers (AttnLRP)** [Achtibat et al., 2024]. `[TBD: implementation; not yet in ctnet_pd.]` The maps labelled LRP in the preliminary study came from `tf_keras_vis.saliency.Saliency`, which computes vanilla gradient saliency. Gradient saliency $|\partial o_\delta/\partial X|$ is reported here under its correct name.
- **Score-CAM** [Wang et al., 2020] on the stem output.
- **Gradient-weighted attention rollout** [Chefer et al., 2021; Abnar & Zuidema, 2020]. `[TBD: not yet in ctnet_pd.]`

Grad-CAM is not an independent check of $q_{\text{pre}}$, which is itself Grad-CAM. AttnLRP, Score-CAM and the perturbation tests serve as the independent controls.

---

## 2.10. Biomarker-Guided Acoustic Prior (Optional Regularizer)

### 2.10.1. Acoustic concept maps

For each segment, $K$ soft concept maps $\Pi_k^{\text{high}}\in[0,1]^{128\times229}$ were computed from the preprocessed signal alone (Table 5): $F_0$ and voicing with probabilistic YIN (pYIN, 50–500 Hz) on the same frame grid as the spectrogram, and formants with Praat/Parselmouth [Boersma & Weenink; Jadoul et al., 2018]. Concept maps were segmented with the same operator $\mathcal R$ (padding carries no prior). Their definition, and every parameter in Table 5, was **frozen before any prior-regularized model was trained**.

The concepts were justified only by the literature on hypokinetic dysarthria. They were **not** derived from the saliency maps of the preliminary study, which were obtained on the same corpora with the same model family and would make validation circular.

**Table 5.** Candidate acoustic concepts. `[TBD: final selection and parameters, frozen before the prior experiments.]`

| $k$ | Concept | Spectro-temporal support of $\Pi_k^{\text{high}}$ | Rationale (cite) |
|---|---|---|---|
| 1 | Phonatory band | Voiced frames; Mel bins covering $F_0(t)$ to $3F_0(t)$ (Gaussian edges, SD 2 semitones) | Jitter, $F_0$ variability, monopitch [Little 2009; Rusz 2011] |
| 2 | Formant regions | Voiced frames; Gaussian bands (SD 150 Hz) around $F_1(t)$ and $F_2(t)$ | Articulatory undershoot, vowel space [Skodda; Rusz] |
| 3 | Voicing onset/offset transitions | All bins; triangular window of ±3 frames around voicing changes | Onset/offset deficits in DDK [Vásquez-Correa et al.] |
| 4 | Aperiodic energy | Voiced frames; bins above 4 kHz `[TBD: confirm]` | HNR, breathiness [Tsanas; Rusz] |

Pause-based prosodic concepts were excluded because silences are trimmed during preprocessing.

### 2.10.2. Receptive-field downsampling and prior distribution

Each concept map was projected onto the token grid by
$$
\mathcal D_{\text{RF}}:[0,1]^{128\times229}\to[0,1]^{14\times25},\qquad
\Pi_{k,ij}=\sum_{f,t}a_{ij,ft}\,\Pi_k^{\text{high}}(f,t),\qquad a_{ij,ft}\ge0,\;\;\sum_{f,t}a_{ij,ft}=1, \tag{7}
$$
where $a_{ij,ft}\neq0$ only if $(f,t)$ lies within the receptive field of cell $(i,j)$ of the convolutional stem.

- **Primary choice:** $a$ is uniform over the stride tile $\Omega_{ij}$ of cell $(i,j)$, equivalent to adaptive average pooling. The tiles partition the input, and $\Omega_{ij}$ lies inside the receptive field.
- **Sensitivity analysis:** weights proportional to the effective receptive field.

For the post-attention map, the cell is the token's positional anchor at the stem.

The prior was $\Pi=\sum_k\omega_k\Pi_k$ with **uniform, pre-specified** $\omega_k=1/K$. Leave-one-concept-out was run as a sensitivity analysis, and $\omega_k$ was never estimated from data. The prior distribution is $p=\mathcal N_1(\Pi)$.

### 2.10.3. Losses

**Prior alignment** (PD subjects, explained class $c=\text{PD}$, subject-balanced):
$$
\mathcal L_{\text{prior}}=\frac{1}{|\mathcal S_{\text{PD}}|}\sum_{s\in\mathcal S_{\text{PD}}}\frac{1}{R_s}\sum_{r=1}^{R_s}D_{\mathrm{JS}}\big(q_{J,sr}\,\|\,p_{sr}\big). \tag{8}
$$

**Optional concept supervision.** This is an auxiliary multitask head, not a concept bottleneck. Recording-level acoustic measures $b_{sr}\in\mathbb R^K$ (e.g., jitter, shimmer, HNR, $F_0$ SD, DDK rate) were standardized with training-partition statistics. A head $h_\psi:\mathbb R^{64}\to\mathbb R^K$ on $z$ was trained with
$$
\mathcal L_{\text{concept}}=\frac{\sum_{sr,k}m_{sr,k}\,\mathrm{Huber}(\hat b_{sr,k},\tilde b_{sr,k})}{\sum_{sr,k}m_{sr,k}}, \tag{9}
$$
where $m_{sr,k}\in\{0,1\}$ indicates whether the measure is available.

**Total objective.** The classification loss is subject-balanced cross-entropy:
$$
\mathcal L_{\text{cls}}=-\frac{1}{|\mathcal S_{\text{train}}|}\sum_s\frac{1}{R_s}\sum_r\log\hat p_{sr}(y_s),\qquad
\boxed{\mathcal L=\mathcal L_{\text{cls}}+\lambda_P\mathcal L_{\text{prior}}+\lambda_C\mathcal L_{\text{concept}}}. \tag{10}
$$
Pre/post agreement and explanation stability were **not** optimized; they were used only as evaluation measures (Section 2.11), to avoid reporting quantities that the model was trained to maximize.

### 2.10.4. Pre-specified acceptance criterion for the prior

Let $G$ be the faithfulness gap of Section 2.11.2, computed on outer-test subjects. The prior-regularized model (Joint-Prior) is considered **useful** only if all four conditions hold:

1. **(A1)** $G_{\text{Joint-Prior}}>G^{\text{rand}}_{\text{Joint-Prior}}$, with a 95% CI excluding 0.
2. **(A2)** $G_{\text{Joint-Prior}}\ge G_{\text{CTNet}}-0.02$ (non-inferior faithfulness).
3. **(A3)** $\mathrm{AUROC}_{\text{Joint-Prior}}\ge\mathrm{AUROC}_{\text{CTNet}}-0.05$ (non-inferior prediction).
4. **(A4)** Joint-Prior outperforms the Random-Prior control in $G$, or in task-matched external AUROC.

`[TBD: confirm the margins 0.02 and 0.05 before running.]` If any condition fails, the prior is reported as not beneficial. This negative result is informative in its own right.

---

## 2.11. Quantitative Evaluation of Explanations

### 2.11.1. Perturbation baselines

Perturbations replaced cells with a reference spectrogram $B$:

- **Primary:** band-wise Gaussian noise, $B_{f,t}=\mu_f+\sigma_f\varepsilon_{f,t}$ with $\varepsilon_{f,t}\sim\mathcal N(0,1)$, where $\mu_f,\sigma_f$ are the log-Mel band statistics of the training partition.
- **Secondary:** band-wise mean, $B_{f,t}=\mu_f$.

Unlike zero filling, both baselines keep perturbed inputs within the range of the training distribution.

### 2.11.2. Deletion and insertion curves

Let $\mathbf 1_q$ be the binary input-space mask obtained by upsampling (nearest neighbor) the top-$q$ fraction of the 350 cells of a map. Then
$$
X_{\text{del}}(q)=X\odot(1-\mathbf 1_q)+B\odot\mathbf 1_q,\qquad X_{\text{ins}}(q)=B\odot(1-\mathbf 1_q)+X\odot\mathbf 1_q, \tag{11}
$$
for $q\in\{0,0.02,\dots,1\}$, i.e., steps of 7 cells. The curves of $\hat p_c$ for the explained class yield $\mathrm{AUC}_{\text{del}}$ (lower is better) and $\mathrm{AUC}_{\text{ins}}$ (higher is better). The faithfulness gap is
$$
G=\mathrm{AUC}_{\text{ins}}-\mathrm{AUC}_{\text{del}}. \tag{12}
$$
Curves were averaged per subject before any statistic was computed. The single-$q$ effect $\Delta^{\text{sal}}_q=\hat p_c(X)-\hat p_c(X_{\text{del}}(q))$ was also reported at $q\in\{0.05,0.10,0.20\}$.

### 2.11.3. Matched random masks

The random reference must match the salient mask in area, shape and frequency distribution, not only in area.

- **Primary — frequency-preserving circular time shift.** $\mathbf 1_q^{\text{rand}}(i,j)=\mathbf 1_q(i,(j+\tau)\bmod 25)$, with $\tau$ uniform in $\{6,\dots,19\}$ and 10 draws per segment. This keeps the same Mel bands, area and shape. The overlap (IoU) with the salient mask was reported.
- **Secondary — within-row cell permutation.** Preserves the exact frequency marginal.

This yields $G^{\text{rand}}$ and $\Delta^{\text{rand}}_q$, and the faithfulness hypothesis is
$$
H_{\text{faith}}:\;G>G^{\text{rand}}\quad(\text{equivalently }\Delta^{\text{sal}}_q>\Delta^{\text{rand}}_q). \tag{13}
$$

### 2.11.4. ROAR (confirmatory, restricted)

To rule out drops caused only by out-of-distribution inputs, Remove-And-Retrain (ROAR) [Hooker et al., 2019] was run:

1. For $q\in\{0.1,0.3,0.5\}$, the top-$q$ cells of $q_J$ (computed by the original fold model) were replaced with $B$ in all training and test segments.
2. CTNet was retrained from scratch with the same hyperparameters.
3. The result was compared with the same procedure using matched random masks.

The expected result is $\mathrm{Perf}_{q_J}(q)<\mathrm{Perf}_{\text{rand}}(q)$.

**Scope, to limit cost:** one repetition of the outer 5-fold CV, both cohorts, CTNet and Joint-Prior only. `[Optional: ROAD (Rong et al., 2022) as a cheaper complement.]`

### 2.11.5. Sanity checks

Sanity checks [Adebayo et al., 2018] were applied to $q_{\text{pre}}$, $q_{\text{post}}$, $q_J$, AttnLRP and Score-CAM.

1. **Cascading parameter randomization.** Weights were re-initialized from the top down: classifier → Transformer layers $L,\dots,1$ → conv block 2 → conv block 1. Similarity $\rho(M^\star,M^{(k)})$ to the trained map was measured with absolute Spearman correlation, SSIM and top-10% IoU, and is expected to decrease. Independent (single-layer) randomization was also reported.
2. **Label randomization.** Models were retrained with subject-level labels permuted within the training folds. The test is **passed** if the upper 95% CI bound of $\rho(M_{\text{true}},M_{\text{perm}})$ lies below the lower 95% CI bound of the **seed-to-seed similarity** $\rho(M^{\text{seed}_1}_{\text{true}},M^{\text{seed}_2}_{\text{true}})$. This gives an operational threshold instead of "$\rho\ll1$".

### 2.11.6. Pre/post agreement and explanation stability (descriptive)

**Agreement.** Spearman $\rho(q_{\text{pre}},q_{\text{post}})$ and top-$q$ Jaccard index at $q\in\{0.05,0.10,0.20\}$, on the $14\times25$ grid.

**Stability.** $D_{\mathrm{JS}}\big(q_J(X),q_J(a(X))\big)$ under label-preserving perturbations $a$:
- small time shifts, realigned with $\mathcal T_a^{-1}$;
- additive noise at 30 dB SNR;
- gain changes of ±3 dB.

Pitch shifting was excluded because it alters $F_0$-related biomarkers.

---

## 2.12. Cross-Cohort Reproducibility of Explanations

### 2.12.1. Frequency profiles

The time axis of different utterances is not aligned, so maps were marginalized over time. For subject $s$ (averaged over segments and recordings) and frequency row $i$:
$$
\varphi_s^{(c)}(i)=\frac{1}{25}\sum_{j=1}^{25}\bar q_s^{(c)}(i,j),\qquad i=1,\dots,14. \tag{14}
$$
Profiles at 128 Mel bins were computed for the pixel-level comparators (AttnLRP).

From these profiles, the following were computed for cohort $d$:
- the **class prototypes** $\bar\varphi_d^{(c)}$, i.e., the mean profile explaining class $c$ over subjects of class $c$;
- the **class-contrast profile** $\delta\varphi_d=\bar\varphi_d^{(\text{PD})}-\bar\varphi_d^{(\text{HC})}$.

The contrast removes the class-agnostic structure shared by both groups (spectral envelope, upsampling bias) and was the **primary** quantity:
$$
\rho_{\text{cross}}=\rho_S\big(\delta\varphi_{\text{PC-GITA}},\,\delta\varphi_{\text{NeuroVoz}}\big). \tag{15}
$$
The same statistic was also reported on each class prototype.

### 2.12.2. Which models explain which subjects

- **Scenario B (primary) — independently learned evidence.** PC-GITA maps came from PC-GITA out-of-fold models and NeuroVoz maps from NeuroVoz out-of-fold models (Experiment I).
- **Scenario A — stability of one model under domain shift.** Source-cohort maps came from out-of-fold models, and target-cohort maps from the external model of Experiment III.

### 2.12.3. Null and reference distributions

Similarity between two average profiles can arise from generic structure. $\rho_{\text{cross}}$ was therefore interpreted only against the following references:

1. **Label-permutation null.** Diagnosis labels were permuted across subjects within each cohort independently, and $\delta\varphi_d$ and $\rho_{\text{cross}}$ were recomputed ($10^4$ permutations; for class prototypes, each subject's map was recomputed for its permuted class). One-sided $p=(1+\#\{\rho^{\text{null}}\ge\rho_{\text{obs}}\})/(1+10^4)$. This tests whether the **diagnosis-related** part of the explanation reproduces across cohorts.
2. **Model-randomization floor.** $\rho_{\text{cross}}$ was computed from maps of untrained CTNets with the same architecture (10 random initializations). The observed value must exceed the 95th percentile of this floor, the similarity produced by the architecture and input statistics alone.
3. **Acoustic reference.** $\rho_{\text{cross}}$ of the class-contrast log-Mel **energy** profile $\delta\bar e_d$, pooled to the same 14 rows. The partial Spearman correlation of the explanation profiles controlling for $\delta\bar e$ was also reported. If the explanations reproduce no better than raw spectral differences, they add no information beyond spectral averages.
4. **Noise ceiling.** The subjects of each cohort were split into two stratified halves, and the split-half $\rho$ of $\delta\varphi_d$ was corrected with Spearman–Brown, $\rho_{\text{SB}}=2\rho/(1+\rho)$ (1,000 splits). The reported normalized similarity is $\rho_{\text{cross}}/\sqrt{\rho_{\text{SB},\text{GITA}}\,\rho_{\text{SB},\text{NV}}}$.

95% CIs were obtained by subject bootstrap within each cohort (2,000 resamples). With only 14 frequency rows, Pearson correlation and the complete permutation distribution were also reported.

---

## 2.13. Experimental Design

### Experiment I — Within-cohort, subject-independent validation

Repeated nested CV (Section 2.4) was run separately in PC-GITA and NeuroVoz, for each task family (Table 2) and for the all-task model.

### Experiment II — Same-protocol baselines

Every baseline used identical folds, segments, aggregation, metrics and information boundary, with an equal hyperparameter-search budget (Table 6). Values from studies with other protocols were not used for comparison.

**Table 6.** Baselines.

| Model | Input | Training |
|---|---|---|
| SVM (RBF) | eGeMAPS [Eyben et al., 2016] functionals, per recording | Subject-grouped inner 4-fold grid over $C\in\{0.1,1,10,100\}$, $\gamma$ |
| CNN-only | Same log-Mel segments | CTNet stem + GAP + linear head, no Transformer (60,610 parameters); same search |
| AST, frozen | Log-Mel per AST specification [Gong et al., 2021] | AudioSet-pretrained embeddings + logistic regression |
| Speech SSL, frozen (**recommended**) | Raw waveform, 16 kHz | WavLM-Base+ or XLS-R embeddings (mean-pooled, best layer chosen in subject-grouped inner CV) + logistic regression |
| CTNet-Flatten | Same log-Mel segments | Preliminary-study head, as an ablation of the GAP head |

`[TBD: fine-tuned AST only if the computational budget allows.]`

### Experiment III — Two-way external validation

The two directions were PC-GITA → NeuroVoz and NeuroVoz → PC-GITA.

- **Source training.** Hyperparameters were selected by inner CV on the source cohort. The final model was trained on all source subjects for the median number of epochs selected in the inner folds. No target subject was used for fitting, early stopping, tuning or choosing $\lambda$.
- **Analyses.** The primary analysis was task-matched (/pa-ta-ka/ and monologue, separately); the secondary analysis used all tasks.
- **Thresholds.** Both $\tau=0.5$ and the Youden-optimal threshold estimated on source out-of-fold predictions were reported, together with threshold-free metrics.

### Experiment IV — Ablation of the prior

**Table 7.** Variants. All variants share the architecture of Section 2.6.

| Variant | Map used in $\mathcal L_{\text{prior}}$ | Prior | Concept head |
|---|---|---|---|
| CTNet | – | – | – |
| Pre-Prior | $q_{\text{pre}}$ | ✓ | – |
| Post-Prior | $q_{\text{post}}$ | ✓ | – |
| Joint-Prior | $q_J$ | ✓ | – |
| Joint-Prior-Concept | $q_J$ | ✓ | ✓ |
| CTNet-Concept | – | – | ✓ |
| Random-Prior | $q_J$ | Shuffled | – |

The **Random-Prior** control used the prior of a different, randomly chosen training recording of the same task. This preserves the area, sparsity and population-level spectral marginals while breaking the recording-specific acoustic correspondence. A circular time shift alone would not randomize time-stationary concepts such as the phonatory band.

---

## 2.14. Training Configuration

- **Optimization.** Adam, batch size 32, at most 100 epochs, ReduceLROnPlateau on the inner-validation loss (factor 0.5, patience 5, minimum 10⁻⁶), early stopping (patience 15) restoring the weights with the lowest **inner-validation** subject-balanced loss. In the preliminary code the minimum learning rate equalled the initial one, so the schedule never acted.
- **Hyperparameter search.** Bayesian optimization (KerasTuner) on the inner split, objective inner-validation loss, 10 trials of at most 40 epochs, with the same budget for all models. The search space was:
  - learning rate $\in\{10^{-3},4\cdot10^{-4},10^{-4}\}$;
  - dropout $\in\{0.1,0.2,0.3\}$;
  - attention heads $\in\{1,2,4\}$;
  - activation $\in\{\text{ReLU},\text{GELU}\}$;
  - $\lambda_P,\lambda_C\in\{0.01,0.1,1\}$, prior variants only.
- **Seeds.** Outer folds of repetition $r$ use seed $42+r$; the model of repetition $r$, fold $k$ uses seed $42+1000r+k$.
- **Software and hardware.** Python 3, TensorFlow 2 with Keras 3, KerasTuner, librosa, scikit-learn, Parselmouth and openSMILE; Kaggle, 2× NVIDIA T4. `[TBD: exact versions, logged by the experiment notebook to config_used.json.]` Training time per variant was reported.

---

## 2.15. Predictive Metrics

All metrics were computed at the subject level:

- AUROC (primary predictive endpoint);
- balanced accuracy;
- sensitivity and specificity;
- F1;
- PR-AUC;
- Brier score;
- calibration intercept and slope.

For repeated CV, out-of-fold predictions were pooled within each repetition, and the metrics were averaged across repetitions. 95% CIs were obtained by subject-level bootstrap (2,000 resamples), with all recordings of a subject resampled together.

---

## 2.16. Statistical Analysis

**Pre-specified primary hypotheses.** These were fixed before the test results were examined:

- **H1 (faithfulness):** for CTNet, $G(q_J)>G^{\text{rand}}$ on outer-test subjects, in both cohorts. Tested with a one-sided Wilcoxon signed-rank test on subject-level paired differences.
- **H2 (prior):** the acceptance criterion of Section 2.10.4.

The **primary predictive endpoint** was the subject-level AUROC in task-matched external validation, reported as an estimate with its 95% CI.

**Exploratory (post hoc) analyses.** All other comparisons were labeled exploratory and corrected for multiplicity with the Holm procedure within each family (predictive, explanation, subgroup). They were interpreted through effect sizes and CIs rather than $p$-values alone:

- **Model comparisons on identical subjects.** Paired DeLong for AUROC (within a repetition); paired subject-level bootstrap or permutation for the other metrics.
- **Confounding.** A logistic mixed-effects model of subject-level correctness:
$$
\operatorname{logit}P(E_{sm}=1)=\beta_0+\beta_1\text{Model}_m+\beta_2\text{Age}_s+\beta_3\text{Sex}_s+\beta_4\text{Cohort}_s+\beta_5\text{Class}_s+\boldsymbol\beta_6^\top(\text{Model}_m\times[\text{Age}_s,\text{Sex}_s,\text{Cohort}_s,\text{Class}_s])+u_s, \tag{16}
$$
where $E_{sm}\in\{0,1\}$ indicates a correct prediction and $u_s\sim\mathcal N(0,\sigma_u^2)$ is a subject random effect.
- **Demographic probing.** Linear probes on $z$ predicting age and sex (subject-level CV), together with sensitivity and specificity differences by sex.
- **Subgroups.** Age strata ≤ 60, 61–70 and > 70 years, fixed a priori, plus sex; paired subject-level bootstrap.
- **Clinical association (PD subjects).** Spearman correlation, partial on age, of $\hat p_s$ and of the concept-head outputs with MDS-UPDRS-III, disease duration and Hoehn & Yahr.

---

## 2.17. Reproducibility

Code (`ctnet_pd/`), the complete configuration, fold assignments (subject IDs per fold and repetition, `folds.csv`), frozen concept definitions and trained weights will be released at `[TBD: public repository]`. The corpora are available from their owners under their respective licenses.

---

### Pending items before submission (checklist)

- [x] STFT parameters, log compression, operator $\mathcal R$ and segment hop (Table 3)
- [x] Convolutional-block specification, $L$, FFN size, positional encoding (Table 4)
- [ ] Confirm the silence-trimming threshold and the aperiodic cutoff on pilot data
- [ ] Attention rollout and AttnLRP comparators
- [ ] Final acoustic concepts and their parameters (Table 5), **frozen before the prior experiments**
- [ ] Recompute Table 1 from metadata; verify NeuroVoz recording conditions and PC-GITA task selection
- [ ] Real LRP implementation (AttnLRP)
- [ ] Margins of the acceptance criterion (Section 2.10.4)
- [x] Number of Bayesian-optimization trials and seeds
- [ ] Exact software versions (from `config_used.json`)
- [ ] MDPI back matter: IRB, informed consent, data availability, conflicts of interest
