# LITOBZOR.md — Обзор литературы SciLibMath_v2

> **Версия:** 1.4 | **Фаза:** 2b | **Дата:** 2026-03-08
> **Papers:** A, B, C

---

## Часть I: Развёрнутый анализ

### 1. CLIP $\to$ обобщение на $M$ модальностей

#### 1.1 От siamese networks к InfoNCE

Контрастное обучение как парадигма берёт начало от работы Hadsell, Chopra и LeCun [HADSELL-2006], в которой siamese network обучалась создавать инвариантные отображения: "похожие" входы $\to$ близкие точки в низкоразмерном пространстве, "непохожие" $\to$ далёкие. Лосс-функция имела вид:

$$L(Y, D) = (1 - Y) \cdot \tfrac{1}{2} D^2 + Y \cdot \tfrac{1}{2} \bigl[\max(0, \text{margin} - D)\bigr]^2$$

где $D$ — евклидово расстояние между представлениями, $Y \in \{0, 1\}$ — метка.

Эта конструкция работает для пар, но не масштабируется: нужно явно задавать позитивные и негативные пары.

Van den Oord et al. [INFONCE-2018] решили проблему масштабирования, предложив InfoNCE — лосс, основанный на оценке взаимной информации:

$$\mathcal{L}_{\text{NCE}}(i) = -\log \frac{\exp(f(x_i, c_i) / \tau)}{\sum_{j=1}^N \exp(f(x_j, c_i) / \tau)}$$

[Интуиция] InfoNCE превращает задачу обучения представлений в задачу классификации: из $N$ кандидатов выбрать "правильную" пару. Чем больше батч (больше негативов), тем точнее оценка mutual information.

#### 1.2 SimCLR и MoCo: два полюса

SimCLR [SIMCLR-2020] показал, что простой фреймворк (аугментации + projection head + NT-Xent) создаёт сильные визуальные представления. Ключевой результат: projection head (MLP поверх encoder) критически важен — без него качество падает на 10%+. Мы наследуем этот паттерн: все 5 энкодеров в Family A имеют shared projection layer $g: \mathbb{R}^{d'} \to \mathbb{R}^d$.

MoCo [MOCO-2020] решил проблему большого батча через momentum encoder и queue негативов. Для нашей задачи ($M = 5$ модальностей, ~1M записей) это потенциально релевантно при масштабировании, но на текущем этапе мы используем стандартный in-batch negatives подход (как в CLIP).

#### 1.3 CLIP: $M = 2$ baseline

CLIP [CLIP-2021] — поворотный момент: контрастное обучение на 400M пар (изображение, текст) создаёт пространство с zero-shot трансфером. Архитектурно CLIP прост:

1. Image encoder (ResNet или ViT) $\to$ вектор $z_{\text{img}} \in \mathbb{R}^d$
2. Text encoder (Transformer) $\to$ вектор $z_{\text{text}} \in \mathbb{R}^d$
3. Симметричный InfoNCE:

$$\mathcal{L} = \tfrac{1}{2}\bigl(\mathcal{L}_{\text{NCE}}(\text{img} \to \text{text}) + \mathcal{L}_{\text{NCE}}(\text{text} \to \text{img})\bigr)$$

**Что CLIP НЕ делает:**
- Не работает с $M > 2$ модальностями (нет обобщения InfoNCE на 5+ входов).
- Не использует центроидные представления (retrieval по попарному сходству).
- Не имеет механизма балансировки модальностей (обе модальности равноправны).

ALIGN [ALIGN-2021] масштабировал CLIP-подход на 1.8B шумных пар, показав, что шумные данные работают при достаточном масштабе. Для нас это релевантно: аугментация SciLibModal_v2 через LLM создаёт шумные пары.

LiT [LIT-2022] предложил замораживать visual encoder и обучать только текстовый — locked-image tuning. Наша стратегия `text_unfreeze_ratio` (частичная разморозка) в v_1 [KHALOV-2025] — вариация этого подхода.

#### 1.4 Gap: от $M = 2$ к $M = 5$

**Ни одна из перечисленных работ не решает нашу задачу:** контрастное обучение для $M \geq 3$ модальностей одного математического объекта с центроидным retrieval. Ближайший подход — CMC [CMC-2020], который обобщает контрастное обучение на $> 2$ views, но:
- CMC максимизирует MI для всех пар views, без центроидной агрегации;
- CMC работает с визуальными views одного изображения, не с гетерогенными модальностями (LaTeX, Lean, NL, image);
- CMC не имеет механизма управления весами модальностей.

---

### 2. Centroid/prototype-based learning: от few-shot к нашей задаче

#### 2.1 Prototypical Networks и centroid retrieval

Идея "представлять класс через центроид (прототип)" активно развивалась в few-shot learning [PROTO-NETS-2017]. В prototypical networks каждый класс представляется центроидом support set embeddings, а классификация — по расстоянию до центроидов.

**Разница с нашей постановкой:**
- В proto-nets центроид объединяет примеры одного класса (разные объекты).
- У нас центроид объединяет модальности одного объекта (один объект, разные представления).
- Proto-nets работают в few-shot режиме, мы — в full supervision.

Тем не менее, геометрическая интуиция та же: идентификация через центроид лучше, чем через любую отдельную точку, потому что центроид "усредняет" шум отдельных модальностей.

#### 2.2 GradNorm: адаптивная балансировка

GradNorm [GRADNORM-2018] — предшественник нашей задачи управления весами. Алгоритм нормализует градиенты разных задач (task losses), чтобы ни одна задача не доминировала. Формально:

$$\mathcal{L}_{\text{grad}}(t) = \sum_i \bigl| G_i(t) - \bar{G}(t) \cdot r_i(t) \bigr|$$

где $G_i(t)$ — L2 норма градиента задачи $i$, $\bar{G}(t)$ — среднее, $r_i(t)$ — целевое соотношение.

**Отличие от нашего подхода:**
- GradNorm — gradient-based, работает в пространстве градиентов.
- Наш T-S fuzzy controller — signal-based, работает в пространстве наблюдаемых сигналов (loss values, EMA, conflict indicators).
- GradNorm не имеет символьного слоя (нет интерпретируемых правил).
- GradNorm не обеспечивает stability guarantees (нет Lyapunov).

---

### 3. Contrastive loss для $M > 2$ модальностей

#### 3.1 Текущий landscape

CMC [CMC-2020] формализует мультиview контрастное обучение:

$$\mathcal{L}_{\text{CMC}} = \sum_{i < j} \mathcal{L}_{\text{NCE}}(\text{view}_i, \text{view}_j)$$

т.е. сумма попарных InfoNCE по всем парам views. Для $M$ модальностей это $O(M^2)$ пар.

**Проблемы CMC для $M = 5$:**
1. $O(M^2) = 10$ попарных лоссов — неоправданно много, каждая пара получает одинаковый вес.
2. Нет явной агрегации: центроид не участвует в лоссе.
3. Нет механизма обнаружения и подавления collapse/domination одной модальности.

#### 3.2 Наш подход: centroid InfoNCE

Вместо $O(M^2)$ попарных лоссов мы предлагаем:

$$\mathcal{L}_{\text{centroid}}(i) = -\log \frac{\exp(\langle c_i, c_i'\rangle / \tau)}{\sum_j \exp(\langle c_i, c_j'\rangle / \tau)}$$

где $c_i = \frac{1}{M}\sum_m e_m^i$ — центроид полного набора модальностей, $c_i'$ — центроид с modality dropout.

**Преимущество:** $O(N^2)$ (по объектам в батче), а не $O(N^2 \cdot M^2)$.

[**Gap G1**] ImageBind [IMAGEBIND-2023] работает с $M = 6$ модальностями, но через image-anchored binding (все модальности выравниваются к image), без равноправного центроидного подхода. Meta-Transformer [META-TRANSFORMER-2023] использует единый backbone для $M = 12$, но без центроидного retrieval. CMC [CMC-2020] ближе по духу, но использует $O(M^2)$ попарных лоссов без агрегации. **Отсутствует:** centroid InfoNCE для $M \geq 3$ гетерогенных символьных модальностей одного математического объекта с centroid-based retrieval.

---

### 4. Fuzzy control в ML: T-S fuzzy systems

#### 4.1 Основы T-S fuzzy model

Takagi-Sugeno fuzzy system [WANG-1996] представляет нелинейную систему как взвешенную смесь локальных линейных моделей:

$$\text{Rule } r: \quad \text{IF } z_1 \text{ is } M_1^r \;\wedge\; z_2 \text{ is } M_2^r \;\wedge\; \ldots \quad \text{THEN } \dot{x} = A_r x + B_r u$$

Выход системы:

$$\dot{x} = \sum_r h_r(z) \cdot (A_r x + B_r u)$$

где $h_r(z) = \mu_r(z) / \sum_k \mu_k(z)$ — нормализованные активации правил.

Устойчивость T-S системы анализируется через LMI (Linear Matrix Inequalities). Wang et al. показали, что общая квадратичная функция Ляпунова $V(x) = x^\top P x$ ($P > 0$) гарантирует устойчивость, если существует $P$, удовлетворяющая:

$$A_r^\top P + P A_r < 0, \quad \forall r$$

#### 4.2 Развитие: fuzzy Lyapunov functions

Mozelli и Palhares [MOZELLI-2010] ослабили требование общей квадратичной Ляпунова, предложив fuzzy Lyapunov function:

$$V(x) = \sum_r h_r(z) \cdot x^\top P_r x$$

— выпуклая комбинация локальных квадратичных функций с разными $P_r$ для каждого правила. Это менее консервативно (позволяет стабилизировать более широкий класс систем).

Abdelmalek et al. [ABDELMALEK-2007] пошли дальше: неквадратичные функции Ляпунова для T-S. Результат — ещё менее консервативные условия стабильности.

#### 4.3 Применение к нашей задаче

**T-S fuzzy controller в контексте обучения нейросети — это новая задача.** Мы не знаем работ, которые используют T-S fuzzy для управления гиперпараметрами в процессе мультимодального обучения. Существуют работы по:
- Adaptive learning rate (но через heuristics, не через T-S).
- Curriculum learning (но через scheduling, не через fuzzy control).
- Neural architecture search (но NAS ищет архитектуру, не управляет обучением в реальном времени).

[**Gap G2**] T-S fuzzy controller для адаптации гиперпараметров в мультимодальном контрастном обучении — открытая задача.

---

### 5. Lyapunov в DL: stability analysis

#### 5.1 CLF/CBF framework

Li et al. [CLF-CBF-SURVEY] обзорно описывают Control Lyapunov Functions (CLF) и Control Barrier Functions (CBF) для нелинейных систем. CLF гарантирует, что существует управление, делающее систему устойчивой:

$$\exists u: \nabla V \cdot f(x, u) \leq -\alpha(V(x))$$

CBF гарантирует безопасность (система не покидает допустимое множество).

CLF-QP (quadratic program) фильтр — "минимально инвазивный" контроллер: он берёт номинальное управление и минимально корректирует его для обеспечения устойчивости. Это **концептуально аналогично** нашему подходу:
- T-S fuzzy $\to$ номинальное управление (грубое направление корректировки гиперпараметров).
- Lyapunov constraint $\to$ "сглаживающий фильтр" (не даёт контроллеру дестабилизировать обучение).

#### 5.2 Lyapunov в контексте SGD

Прямое применение Ляпунова к доказательству сходимости SGD для нелинейных нейросетей — это **открытая проблема**, и мы не претендуем на её решение. Наша постановка честная:

> We model the hyperparameter adaptation layer as a controlled dynamical subsystem, not the full nonconvex training process.

Функция Ляпунова в нашем контексте:

$$V_t = \alpha \cdot \tilde{L}_t + \beta \cdot \|\Delta\boldsymbol{\lambda}_t\|^2 + \gamma \cdot \text{Var}_m(w_{m,t})$$

Целевое стохастическое условие:

$$\mathbb{E}[V_{t+1} - V_t \mid \mathbf{s}_t] \leq -\eta \cdot \Psi_t + \xi$$

На практике реализуется как soft constraint ($\mathcal{L}_{\text{lyap}}$ штрафует $V_t - V_{t-1}$, т.к. $V_{t+1}$ недоступен на шаге $t$). Мы верифицируем эмпирически: (1) bounded hyperparameter trajectories, (2) absence of oscillatory switching, (3) monotone tendency of surrogate stability objective. LMI-условия [WANG-1996] используются как концептуальная аналогия, не как прямой инструмент доказательства (наша система дискретная и стохастическая).

[**Gap G3**] Lyapunov stability constraints для fuzzy-контролируемой адаптации гиперпараметров в мультимодальном обучении — открытая задача. Работы по Lyapunov в DL существуют для stability analysis обученных сетей (верификация), но не для управления процессом обучения.

---

### 6. Neuro-symbolic подходы к математике

#### 6.1 Нейро-символьное обучение

NS-CL [NS-CL-2019] показал, что символьный слой поверх нейросетевого восприятия улучшает интерпретируемость при сопоставимой производительности. Идея: нейросеть извлекает концепты из raw data, символьный модуль рассуждает над концептами.

Logic Tensor Networks [LTN-2022] формализовали интеграцию логики первого порядка с тензорными вычислениями. "Real Logic" — grounding логических формул в дифференцируемые функции.

Differentiable logics [DIFF-LOGIC-2025] сравнивают различные t-norms (product, Gödel, Łukasiewicz) как основу для дифференцируемых логических операций. Выбор t-norm влияет на свойства градиентов — product t-norm даёт ненулевые градиенты везде, Gödel t-norm даёт разреженные.

#### 6.2 Математические датасеты и ATP

HERALD [HERALD-2025] обеспечивает 50K пар (Lean 4 $\leftrightarrow$ NL) — часть нашего датасета.
ATLAS [ATLAS-2025] предоставляет pipeline автоформализации — аугментация через synthesis.
LeanDojo-v2 [LEANDOJO-V2-2024] — framework для premise retrieval и proof search.
NL2Lean [NL2LEAN-2025] — direct translation NL$\to$Lean через RL. Результаты показывают сложность: прямой перевод хуже, чем embedding-based retrieval.
DeepSeek-Prover-V2 [DEEPSEEK-PROVER-V2-2025] — state-of-the-art в формальном рассуждении, определяет benchmark.
MathLib Semantic Search [MATHLIB-SEARCH-2024] — прямой конкурент для Сценария 1, но $M = 1$ (только текст).
LeanRAG [LEANRAG-2025] — RAG с knowledge graph для Lean, hierarchical retrieval.

#### 6.3 Наша позиция

Мы не строим новый theorem prover и не создаём новый reasoning engine. Мы создаём **embedding space**, в котором все модальности математического объекта коексистируют. Это enables лучший retrieval для downstream ATP (premise selection), лучший RAG (semantic, не syntactic matching), и новый capability (cross-modal search по 5 модальностям).

---

## Часть II: Сжатый обзор (paper-ready Related Work)

### Related Work

**Contrastive representation learning.** The foundation of our approach builds on contrastive learning methods [HADSELL-2006, INFONCE-2018]. InfoNCE [INFONCE-2018] provides a tractable mutual information estimator through noise-contrastive classification. SimCLR [SIMCLR-2020] and MoCo [MOCO-2020] demonstrated that projection heads and large negative pools are critical for learning strong representations. CLIP [CLIP-2021] scaled this paradigm to 400M image-text pairs, achieving remarkable zero-shot transfer. ALIGN [ALIGN-2021] showed that noisy data works at scale. LiT [LIT-2022] introduced locked pre-trained encoders with contrastive tuning. Barlow Twins [BARLOW-2021] offered an alternative via redundancy reduction. All these methods operate with $M = 2$ modalities.

**Multi-view and multimodal learning.** Early work on multimodal deep learning [NGIAM-2011] showed benefits of joint training. CMC [CMC-2020] generalized contrastive learning to multiple views via pairwise InfoNCE ($O(M^2)$ complexity) without centroid aggregation. ImageBind [IMAGEBIND-2023] unified 6 modalities through image-anchored binding. Meta-Transformer [META-TRANSFORMER-2023] applied a unified backbone to 12 modalities. GradNorm [GRADNORM-2018], PCGrad [PCGRAD-2020], and CAGrad [CAGRAD-2021] addressed gradient-based multi-task balancing. None of these works considers centroid-based retrieval for $M \geq 5$ heterogeneous symbolic modalities of mathematical objects.

**Mathematical datasets and theorem proving.** HERALD [HERALD-2025] provides 50K annotated Lean 4 theorems; ATLAS [ATLAS-2025] introduces autoformalization through data augmentation. LeanDojo-v2 [LEANDOJO-V2-2024] offers end-to-end infrastructure for AI-assisted theorem proving. NL2Lean [NL2LEAN-2025] translates natural language to Lean 4 via RL. DeepSeek-Prover-V2 [DEEPSEEK-PROVER-V2-2025] advances formal reasoning with subgoal decomposition. A semantic search engine for Mathlib4 [MATHLIB-SEARCH-2024] provides text-based retrieval ($M = 1$). LeanRAG [LEANRAG-2025] integrates knowledge graphs with RAG for Lean. Our work extends these efforts by embedding mathematical objects across 5 modalities simultaneously, enabling cross-modal retrieval that these systems cannot perform.

**Fuzzy control and Lyapunov stability.** Takagi-Sugeno fuzzy systems [WANG-1996] provide piecewise-affine models with LMI-based stability guarantees. Fuzzy Lyapunov functions [MOZELLI-2010, ABDELMALEK-2007] reduce conservatism of stability conditions. CLF/CBF frameworks [CLF-CBF-SURVEY] formalize minimally invasive control. We adapt the T-S framework to hyperparameter control in neural network training — a novel application where the "plant" is the training dynamics and the "control signal" adjusts loss weights and learning parameters.

**Neuro-symbolic integration.** NS-CL [NS-CL-2019] demonstrated interpretable concept learning. Logic Tensor Networks [LTN-2022] ground first-order logic in differentiable computations. Differentiable logics [DIFF-LOGIC-2025] compare t-norm choices for learning with constraints. Our symbolic layer uses Description Logic / OWL 2 ontologies rather than FOL, aligning with established ontology engineering practices for knowledge representation in control systems.

---

## Часть III: Gap Analysis

### Gap G1: Centroid InfoNCE для $M \geq 3$ гетерогенных символьных модальностей + visual-text alignment для математических формул

**Что есть:** CLIP ($M = 2$), CMC ($M > 2$, $O(M^2)$ попарный InfoNCE без центроидов), ImageBind ($M = 6$, image-anchored binding [IMAGEBIND-2023]), Meta-Transformer ($M = 12$, unified backbone [META-TRANSFORMER-2023]), SigLIP (sigmoid loss, batch-size-independent [SIGLIP-2023]).

**Что отсутствует:**

1. **Centroid-based contrastive learning.** Контрастный лосс, оперирующий через **центроиды** $M$ модальностей одного объекта с centroid-based retrieval. Существующие $M > 2$ системы (ImageBind, Meta-Transformer) не используют центроидную агрегацию. Для $M = 5$ **символьных** модальностей математических объектов (NL en, NL ru, LaTeX, Lean 4, image) — нет работ.

2. **Visual-text alignment для математических формул.** Существующие подходы к визуальному кодированию формул (image-to-LaTeX: Im2LaTeX, HME100K) решают задачу OCR (распознавание), а не семантического выравнивания. В мультимодальном контексте визуальные токены из CNN подаются в текстовый трансформер (как в ViLT, CLIP-ViT), но для математических формул это создаёт **distributional mismatch**: текстовый энкодер (SciRus-tiny) обучен на лингвистических токенах, а визуальные патчи из ResNet имеют другое распределение, семантику и позиционную структуру. Auxiliary contrastive alignment loss ($\mathcal{L}_{\text{visual\_align}}$, привязка visual $\to$ LaTeX) для визуального энкодера математических формул с AlignNet (LayerNorm + Linear) — подход, не встречающийся в литературе для математического домена.

**Мы закрываем:** E2-E5 — centroid InfoNCE с различными вариантами регуляризации и управления весами. AlignNet + $\mathcal{L}_{\text{visual\_align}}$ — semantic alignment визуальных формул с LaTeX-представлениями.

**Paper:** A, B

---

### Gap G2: T-S fuzzy controller для адаптации гиперпараметров нейросетевого обучения

**Что есть:** GradNorm [GRADNORM-2018] (gradient-based балансировка, без символьного слоя). PCGrad [PCGRAD-2020] и CAGrad [CAGRAD-2021] — gradient surgery для конфликтующих задач. PBT [PBT-2017] — популяционная online-адаптация гиперпараметров. T-S fuzzy control для физических систем [WANG-1996, MOZELLI-2010].

**Что отсутствует:** Использование T-S fuzzy system с лингвистическими переменными и DL-онтологией для online адаптации весов модальностей и температуры в мультимодальном контрастном обучении.

**Мы закрываем:** E6 — fuzzy controller для управления матрицей $W$ и гиперпараметрами лосса на основе наблюдаемых сигналов обучения.

**Paper:** C

---

### Gap G3: Lyapunov stability constraints для hyperparameter controller в DL

**Что есть:** CLF/CBF для робототехники [CLF-CBF-SURVEY]. Lyapunov analysis для верификации обученных нейросетей (post-hoc). Stability analysis для RNN (gradient clipping, spectral normalization — но это не Lyapunov в формальном смысле).

**Что отсутствует:** Стохастическая функция Ляпунова как regularizer, гарантирующая bounded trajectories и absence of oscillatory switching для fuzzy-контролируемых гиперпараметров в процессе обучения.

**Мы закрываем:** E7 — Lyapunov smoothing constraint поверх T-S fuzzy controller. Формулировка "controlled dynamical subsystem", не "global convergence proof".

**Paper:** C

---

### Gap G4: 5-модальный датасет математических объектов

**Что есть:** HERALD (Lean $\leftrightarrow$ NL, $M = 2$), ATLAS (informal $\leftrightarrow$ formal, $M = 2$), MathLib4 (Lean only, $M = 1$).

**Что отсутствует:** Датасет, содержащий для каждого математического объекта все 5 модальностей: NL-en, NL-ru, LaTeX, Lean 4, визуальное представление формулы. Масштаб ~1M записей.

**Мы закрываем:** SciLibModal_v2 — расширение датасета v_1 за счёт MathLib4 (Jixia + LLM-аугментация).

**Paper:** A

---

╔══════════════════════════════════════════════════════════════╗
║  META-REASONING LOOP (v1.2)                                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  [MR.1] КАК МЫ ПРИШЛИ К ЭТОМУ ДОКУМЕНТУ                     ║
║                                                              ║
║  Исходные данные: LIT.md (39 записей), MOTIVATION.md,        ║
║  SETUP_DRAFT.md, CLAUDE.md (структура обзора)                ║
║                                                              ║
║  Промежуточные шаги:                                         ║
║  1. Группировка литературы по 6 обязательным блокам           ║
║  2. Для каждого блока: что авторы делали → что не делали →    ║
║     где наш gap                                               ║
║  3. Сжатие в paper-ready Related Work                         ║
║  4. Формализация gaps G1-G4                                   ║
║                                                              ║
║  Изменения v1.1 → v1.2:                                     ║
║  1. Все [NEED_VERIFY] сняты: Snell, ImageBind,               ║
║     Meta-Transformer, SigLIP, PCGrad, CAGrad, PBT —          ║
║     PDF добавлены в literature/addition/                       ║
║  2. Ссылки заменены на формальные ключи из LIT.md v1.2       ║
║  3. Согласование с MATH.md v2.2 по нотации                   ║
║                                                              ║
║  [MR.2] МОЖНО ЛИ УЛУЧШИТЬ                                   ║
║                                                              ║
║  a) Часть II (paper-ready) может быть расширена при           ║
║     полном чтении каждого PDF                                 ║
║  b) VICReg [VICREG-2022] может быть добавлен в Sec. 1        ║
║     как альтернатива anti-collapse                            ║
║                                                              ║
║  [MR.3] ЕСТЬ ЛИ ПРОТИВОРЕЧИЯ                                ║
║                                                              ║
║  Уровень 1: Нет внутренних.                                  ║
║  Уровень 2: Gaps G1-G4 согласованы с MOTIVATION.md v1.2      ║
║    (три публикации + датасет). Нотация согласована с          ║
║    MATH.md v2.2 ($M$, $N$, InfoNCE, centroid, T-S).          ║
║  Уровень 3: Честная постановка Lyapunov ("controlled         ║
║    subsystem") — как требует CLAUDE.md [P.2].                 ║
║                                                              ║
║  [MR.4] ПРОВЕРКА НА ГАЛЛЮЦИНАЦИИ                            ║
║                                                              ║
║  G1: Все ссылки верифицированы через PDF в literature/        ║
║      или literature/addition/. Ни одного [NEED_VERIFY].      ║
║  G2: Формулы: InfoNCE, Hadsell loss, T-S model, LMI,         ║
║      fuzzy Lyapunov, CLF, V_t — все стандартные               ║
║      определения, размерности корректны.                      ║
║  G3: "открытая задача" для Gap G2, G3 — основано на           ║
║      отсутствии в literature/.                                 ║
║  G4: Venues и годы согласованы с LIT.md v1.2.               ║
║                                                              ║
║  [MR.5] ПАТТЕРНЫ                                             ║
║                                                              ║
║  [P.5] NEED_VERIFY CASCADE (v1.1, resolved in v1.2)         ║
║    Все 9 [NEED_VERIFY] записей были разрешены: PDF            ║
║    добавлены в literature/addition/, ссылки формализованы.    ║
║    Паттерн подтверждён: контекстные ссылки допустимы          ║
║    как [NEED_VERIFY], ключевые для gaps — требуют файлы.     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
