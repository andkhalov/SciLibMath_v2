# MATH.md — Полная математическая формализация SciLibMath_v2

> **Версия:** 2.6.2 | **Фаза:** 3 | **Дата:** 2026-03-09
> **Стиль:** PoC (proof-of-concept) — все выводы явные, никаких "очевидно следует"
> **Papers:** A (M.0-M.2), B (M.3-M.4), C (M.5-M.8)

---

## Блок M.0 — Нотация и пространства

### M.0.1 Базовые обозначения

| Символ | Тип | Размерность | Область определения | Смысл | Блок введения |
|---|---|---|---|---|---|
| $d$ | скаляр | $1$ | $\mathbb{N}, d \geq 1$ | размерность эмбеддингового пространства | M.0 |
| $M$ | скаляр | $1$ | $\mathbb{N}, M = 5$ | количество модальностей | M.0 |
| $N$ | скаляр | $1$ | $\mathbb{N}, N \geq 2$ | размер батча | M.0 |
| $o_i$ | кортеж | $M$ элементов | $i \in \{1,\ldots,N\}$ | мультимодальный объект | M.0 |
| $x_i^m$ | вход | varies | $m \in \mathcal{M}$ | вход модальности $m$ объекта $i$ | M.0 |
| $f_m$ | функция | $\mathcal{X}^m \to \mathbb{R}^{d'_m}$ | — | backbone энкодер модальности $m$ | M.1 |
| $g_m$ | функция | $\mathbb{R}^{d'_m} \to \mathbb{R}^d$ | — | projection head модальности $m$ | M.1 |
| $e_i^m$ | вектор | $\mathbb{R}^d$ | — | эмбеддинг: $e_i^m = g_m(f_m(x_i^m))$ | M.1 |
| $c_i$ | вектор | $\mathbb{R}^d$ | — | центроид объекта $i$ | M.0 |
| $\tau$ | скаляр | $\mathbb{R}$ | $\tau > 0$ | температурный параметр | M.3 |
| $\mathbf{s}_t$ | вектор | $\mathbb{R}^p$ | — | состояние системы на шаге $t$ | M.5 |
| $\mathbf{u}_t$ | вектор | $\mathbb{R}^k$ | — | выход fuzzy controller на шаге $t$ ($k = 11$) | M.6 |
| $\boldsymbol{\lambda}_t$ | вектор | $\mathbb{R}^k$ | $\boldsymbol{\lambda}_t \in \Lambda$ | гиперпараметры на шаге $t$ ($k = 11$, вкл. $\lambda_{\text{va}}$) | M.6 |
| $V_t$ | скаляр | $\mathbb{R}$ | $V_t \geq 0$ | функция Ляпунова на шаге $t$ | M.7 |

### M.0.2 Модальности

Множество модальностей:

$$\mathcal{M} = \{\text{en},\; \text{ru},\; \text{lean},\; \text{latex},\; \text{img}\}$$

Для каждой модальности $m \in \mathcal{M}$:
- $\mathcal{X}^m$ — пространство входов модальности $m$
- $f_m: \mathcal{X}^m \to \mathbb{R}^{d'_m}$ — backbone encoder (выход размерности $d'_m$)
- $g_m: \mathbb{R}^{d'_m} \to \mathbb{R}^d$ — projection head (приведение к общей размерности $d$)
- Итоговый эмбеддинг: $e_i^m = g_m(f_m(x_i^m))$

### M.0.3 Центроид

**[Определение M.0.1 — Центроид объекта]**

Для объекта $o_i$ с $M$ модальностями центроид определяется как арифметическое среднее эмбеддингов:

$$c_i = \frac{1}{M} \sum_{m \in \mathcal{M}} e_i^m$$

**[Интуиция]** Центроид — точка в пространстве эмбеддингов, равноудалённая (в среднем) от всех модальных представлений объекта. Он усредняет шум отдельных модальностей и даёт наиболее робастное представление.

**[Assumption A.1]** Все модальности присутствуют для каждого объекта в батче. Если модальность отсутствует — используется modality dropout (см. M.3.2).

### M.0.4 Нормализация

**[Assumption A.2]** Все эмбеддинги L2-нормализованы перед вычислением сходства:

$$\hat{e}_i^m = \frac{e_i^m}{\|e_i^m\|_2 + \varepsilon}$$

Это помещает все эмбеддинги на единичную гиперсферу $\mathbb{S}^{d-1}$. Косинусное сходство $\hat{e}_i^m$ и $\hat{e}_j^{m'}$ равно их скалярному произведению:

$$\cos(\hat{e}_i^m, \hat{e}_j^{m'}) = \langle \hat{e}_i^m, \hat{e}_j^{m'} \rangle$$

Центроид вычисляется из **ненормализованных** эмбеддингов, затем нормализуется отдельно: $\hat{c}_i = c_i / \|c_i\|_2$. Среднее нормализованных векторов $\neq$ нормализованное среднее ненормализованных — это существенно.

**[Литература]** В CLIP [Radford et al., 2021, Sec. 2.3] нормализация применяется перед вычислением similarity matrix. Мы следуем этой конвенции.

---

## Блок M.1 — Архитектура энкодеров [Paper A]

### M.1.1 Family A: 5 отдельных энкодеров

**[Определение M.1.1 — Family A Architecture]**

Для каждой текстовой модальности $m \in \{\text{en}, \text{ru}, \text{lean}, \text{latex}\}$:

$$f_m = \text{Transformer}(\theta_m) \quad \text{(SciRus-tiny 3.5 zh backbone)}$$
$$g_m = \text{Linear}(d'_m, d) \quad \text{(projection head)}$$

где $d'_m = d_{\text{model}}$ — скрытая размерность SciRus-tiny, $d$ — целевая размерность эмбеддинга.

Для визуальной модальности $m = \text{img}$ используется двухэтапный pipeline (см. M.1.2).

**Стратегия обучения:** все слои backbone обучаются end-to-end (full fine-tuning). Discriminative learning rate: backbone получает $\text{lr}_{\text{backbone}} = \text{lr} \cdot \text{lr\_embed\_ratio}$, projection heads получают $\text{lr}$.

**[Связь с v_1]** В [KHALOV-2025] архитектура Family A идентична, но визуальная модальность обрабатывалась через ViT с фиксированным $224 \times 224$. В v_2 мы переходим на ResNet + SciRus-tiny pipeline.

### M.1.2 Визуальный поток

**[Определение M.1.2 — Visual Encoder Pipeline]**

Вход: $x_i^{\text{img}}$ — изображение формулы, высота $h = 64 \text{px}$, ширина $w$ (переменная).

**Шаг 1. Разбиение на перекрывающиеся патчи (Overlapping Patch Embedding):**

Изображение разбивается на патчи размером $p \times p = 64 \times 64$ с горизонтальным шагом (stride) $s < p$:

$$K = \left\lfloor \frac{w - p}{s} \right\rfloor + 1, \quad s = \frac{p}{2} = 32, \quad P_k \in \mathbb{R}^{3 \times p \times p}, \quad k = 1, \ldots, K$$

Патч $P_k$ вырезается из позиции $\bigl[(k-1) \cdot s,\; (k-1) \cdot s + p\bigr]$ по горизонтали (вся высота $h = p = 64$).

**[Обоснование перекрытия]** При $s = p$ (non-overlapping) признаки, расположенные на границе двух соседних патчей, попадают только в один из них и могут быть утрачены или искажены свёрткой вблизи края. При $s < p$ возникает полоса перекрытия шириной $(p - s) = 32$ пикселей, в которой каждый пиксель покрывается двумя соседними патчами. Это обеспечивает:

1. **Сохранение граничных признаков:** символ, расположенный на стыке двух патчей, полностью попадает в рецептивное поле ResNet хотя бы в одном из них;
2. **Робастность к позиции:** каждый пиксель $x$ на расстоянии $\leq p - s$ от границы участвует в $\lceil p/s \rceil = 2$ патчах;
3. **Контролируемая избыточность:** дублирование признаков в полосе перекрытия не накапливается — оно устраняется mean pooling'ом (Шаг 4) и self-attention в SciRus-tiny, которые агрегируют информацию по всей последовательности визуальных токенов.

Количество патчей при overlap: $K_{\text{overlap}} = \lfloor(w-p)/s\rfloor + 1 \approx 2 \cdot \lceil w/p \rceil - 1$ (примерно вдвое больше, чем при $s = p$). Дополнительные вычислительные затраты линейны по $w$ и допустимы для $h = 64$.

**[Литература]** Overlapping Patch Embedding — стандартный приём в Vision Transformers для сохранения локальной непрерывности:
- [Wang et al., 2022, PVTv2, Eq. 1] $\to$ `literature/addition2/2106.13797v7.pdf`: Overlapping Patch Embedding с kernel $= 2s{-}1$, stride $= s$; ablation показывает $+1.3\%$ top-1 на ImageNet по сравнению с non-overlapping;
- [Yuan et al., ICCV 2021, T2T-ViT, Sec. 3.1] $\to$ `literature/addition2/2101.11986v3.pdf`: "Soft Split" с перекрытием патчей, $+2.0\%$ top-1;
- [Wu et al., ICCV 2021, CvT, Sec. 3.1] $\to$ `literature/addition2/Wu_CvT_Introducing_Convolutions_to_Vision_Transformers_ICCV_2021_paper.pdf`: Convolutional Token Embedding с перекрытием, $+0.9\%$.

**Шаг 2. ResNet feature extraction:**

Каждый патч проходит через ResNet (без FC и avg_pool):

$$v_k = \text{ResNet}_{\text{features}}(P_k) \in \mathbb{R}^{d'_{\text{vis}}}$$

где $d'_{\text{vis}} = 512$ для ResNet-18.

**[Интуиция]** Каждый патч — "визуальный токен". ResNet извлекает вектор признаков, аналогичный тому как текстовый токенизатор создаёт embedding для слова.

**Шаг 3. AlignNet и подача в SciRus-tiny:**

Визуальные токены проходят через AlignNet — нормализующую проекцию — и подаются как последовательность в SciRus-tiny:

$$\tilde{v}_k = \text{LayerNorm}\bigl(\text{Linear}(v_k)\bigr) \in \mathbb{R}^{d'_m}$$

$$h_{\text{img}} = \text{SciRus\text{-}tiny}(\tilde{v}_1, \tilde{v}_2, \ldots, \tilde{v}_K)$$

где AlignNet = $\text{LayerNorm}(\text{Linear}(d'_{\text{vis}}, d'_m))$ — проекция визуальных фичей в размерность текстового энкодера с нормализацией распределения.

**[Связь с v2.2]** В предыдущей версии AlignLayer был простым $\text{Linear}(d'_{\text{vis}}, d'_m)$. Добавление LayerNorm стабилизирует распределение визуальных токенов перед подачей в SciRus-tiny, сглаживая distributional mismatch между фичами ResNet и текстовыми embedding'ами. В сочетании с $\mathcal{L}_{\text{visual\_align}}$ (M.1.2*) это обеспечивает семантическую привязку визуальных токенов к текстовому пространству.

**[Связь с SETUP_DRAFT]** "ResNet будет давать вектор визуальных токенов... далее мы делаем секвенцию таких визуальных символов и в нормированном виде (через выравнивающий слой) подаем в SciRus-tiny. Токенизаторы, основная сеть и проекционный слой обучаются совместно."

**Шаг 4. Mean pooling и projection:**

$$\bar{h}_{\text{img}} = \frac{1}{K} \sum_{k=1}^{K} h_{\text{img}}^{(k)}$$

$$e_i^{\text{img}} = g_{\text{img}}(\bar{h}_{\text{img}}) \in \mathbb{R}^d$$

где $g_{\text{img}} = \text{Linear}(d'_m, d)$ — projection head.

**[Assumption A.3]** Mean pooling вместо CLS-токена, потому что нет фиксированного "начала" формулы — информация распределена по всей длине. При overlapping patches (Шаг 1) mean pooling дополнительно выполняет функцию **устранения избыточности**: признаки из полосы перекрытия, дублированные в двух соседних патчах, усредняются при агрегации, а не накапливаются.

### M.1.2* Auxiliary Visual Alignment Loss ($\mathcal{L}_{\text{visual\_align}}$)

**[Определение M.1.2*]**

Для обеспечения семантической связности визуальных эмбеддингов с текстовым пространством вводится вспомогательный контрастивный лосс, привязывающий pooled визуальные токены к pooled LaTeX-токенам того же объекта:

$$\mathcal{L}_{\text{va}}(i) = \bigl\|\text{Pool}(\tilde{v}_1, \ldots, \tilde{v}_K)^{(i)} - \text{Pool}\bigl(\text{Embed}_{\text{latex}}(t_1, \ldots, t_L)\bigr)^{(i)}\bigr\|^2$$

для позитивных пар (один и тот же объект $o_i$), плюс hinge loss для негативных:

$$\mathcal{L}_{\text{va-neg}}(i, j) = \max\bigl(0,\; \delta_{\text{va}} - \bigl\|\text{Pool}(\tilde{v})^{(i)} - \text{Pool}(\text{Embed}_{\text{latex}})^{(j)}\bigr\|\bigr), \quad i \neq j$$

Полный visual alignment loss:

$$\mathcal{L}_{\text{visual\_align}} = \frac{1}{N} \sum_i \mathcal{L}_{\text{va}}(i) + \frac{1}{N(N-1)} \sum_{i \neq j} \mathcal{L}_{\text{va-neg}}(i, j)$$

где:
- $\text{Pool}(\cdot) = \frac{1}{K}\sum_k (\cdot)_k$ — mean pooling по позициям
- $\text{Embed}_{\text{latex}}$ — embedding layer LaTeX-токенизатора (из SciRus-tiny)
- $\delta_{\text{va}} > 0$ — margin для негативных пар

**[Assumption A.11]** Для каждого объекта $o_i$ доступны парные данные: визуальное представление $x_i^{\text{img}}$ и LaTeX-запись $x_i^{\text{latex}}$.

**[Литература]** Конструкция $\mathcal{L}_{\text{visual\_align}}$ основана на contrastive loss [Hadsell, Chopra, LeCun, 2006, CVPR, Eq. 2]: сближение позитивных пар + margin для негативных. Якорь — визуальный embedding, позитив — LaTeX embedding того же объекта.

**[Замечание о $\lambda_{\text{va}}$]** В экспериментах E1-E5 $\lambda_{\text{va}} = \text{const}$ (static auxiliary). В E6-E7 $\lambda_{\text{va}}$ входит в вектор гиперпараметров $\boldsymbol{\lambda}_t \in \mathbb{R}^{11}$ и управляется fuzzy controller (правило R6, M.6.3). Это позволяет контроллеру усиливать визуальное выравнивание когда visual branch отстаёт, и ослаблять когда выравнивание достигнуто.

**[Связь с SETUP_DRAFT]** "ResNet будет давать вектор визуальных токенов... далее мы делаем секвенцию таких визуальных символов и в нормированном виде (через выравнивающий слой) подаем в SciRus-tiny. Токенизаторы, основная сеть и проекционный слой обучаются совместно." AlignNet (LayerNorm + Linear) реализует "выравнивающий слой" из SETUP_DRAFT; $\mathcal{L}_{\text{visual\_align}}$ дополнительно обеспечивает семантическую привязку визуальных токенов к текстовому пространству (LaTeX).

### M.1.3 Family B: 1 symbolic + 1 visual encoder

**[Определение M.1.3 — Family B Architecture]**

Один текстовый энкодер для всех символьных модальностей:

$$f_{\text{sym}} = \text{Transformer}(\theta_{\text{sym}}) \quad \text{(единый backbone для } \{\text{en, ru, lean, latex}\})$$

Визуальный энкодер — как в Family A (M.1.2).

Единый энкодер отображает все текстовые модальности в $\mathbb{R}^d$:

$$e_i^m = g_{\text{sym}}(f_{\text{sym}}(x_i^m)) \in \mathbb{R}^d, \quad m \in \{\text{en, ru, lean, latex}\}$$

Для идентификации модальности — **детерминированная** конкатенация one-hot вектора-индекса:

$$\tilde{e}_i^m = [e_i^m;\; \text{onehot}(m)] \in \mathbb{R}^{d+M}$$

где $\text{onehot}(\text{en}) = [1,0,0,0,0]$, $\text{onehot}(\text{ru}) = [0,1,0,0,0]$, и т.д.

**[Ключевое разделение пространств]**
- Для **всех математических операций** (центроид, лоссы, similarity) используется $e_i^m \in \mathbb{R}^d$ — семантическая часть без one-hot.
- One-hot $\text{onehot}(m) \in \{0,1\}^M$ — детерминированная метка, НЕ обучаемая, НЕ участвует в градиентах. Используется только для идентификации модальности при retrieval.
- Центроид: $c_i = \frac{1}{M}\sum_m e_i^m \in \mathbb{R}^d$ — согласовано с M.0.3.

**[Связь с SETUP_DRAFT]** "Просто искусственно, детерминированно приделываем ещё один компонент вектора — метку, по которой сможем идентифицировать модальность... присвоить метку в виде one-hot вектора, который конкатенируется к полученному вектору модальности."

**Параметры Family B:** $\sim 28$M (vs $\sim 80$M для Family A).

**[Интуиция]** Family B проверяет гипотезу: если один энкодер справляется с 4 текстовыми модальностями, то Family A избыточна. Это ключевая ablation-гипотеза (потенциально болезненный, но научно ценный результат).

**[Замечание: Family B и управление весами]** Family B — максимально простая реализация. Все весовые коэффициенты ($w_g$, $w_m$, $\lambda_{\text{align}}$, $\lambda_{\text{rad}}$, $\lambda_{\text{reg}}$, $\lambda_{\text{va}}$) **фиксированы** перед обучением. Fuzzy controller (M.6) и LossMixer (E5) **НЕ** применяются к Family B. Это чистый ablation baseline: единственная переменная — архитектура энкодеров. Эксперименты E1-E4 проводятся на обоих Family A и B. E5-E7 — только на Family A (требуют отдельных модальных весов).

---

## Блок M.2 — Токенизаторы Lean и LaTeX [Paper A]

### M.2.1 Токенизация как отображение

**[Определение M.2.1]** Для символьной модальности $m$, токенизатор:

$$T_m: \Sigma_m^* \to V_m^*$$

где $\Sigma_m$ — алфавит, $V_m$ — словарь ($|V_m| = \text{vocab\_size}_m$).

Модальности en, ru: предобученные токенизаторы из SciRus-tiny (не модифицируются).
Модальности lean, latex: BPE-токенизаторы обучаются из корпуса SciLibModal_v2.

### M.2.2 Словари

**$V_{\text{lean}}$:** Lean 4 ключевые слова (`theorem`, `lemma`, `by`, `exact`, `apply`, `simp`...), типы (`Nat`, `Real`, `Prop`...), Unicode-операторы ($\to, \forall, \exists, \wedge, \vee, \neg, \leq$...), MathLib-специфичные идентификаторы.

**$V_{\text{latex}}$:** LaTeX-команды (`\frac`, `\sum`, `\int`, `\lim`, `\sqrt`...), окружения, символы ($\alpha, \beta, \infty$...), структурные токены ($\{, \}, \hat{}, \_$...).

**[Интуиция]** Generic текстовый токенизатор разбивает `\frac{a}{b}` на субоптимальные sub-tokens, теряя структуру. Специализированный знает, что `\frac` — один токен.

### M.2.3 Стратегия инициализации эмбеддингов новых токенов [Paper A]

**[Контекст]** Энкодеры en/ru используют предобученный SciRus-tiny с его словарём $V_{\text{base}}$. Для lean/latex мы создаём расширенный словарь $V_m \supset V_{\text{base}}$. Эмбеддинги токенов $V_{\text{new}} = V_m \setminus V_{\text{base}}$ не имеют предобученных весов.

**Подход A — BPE from scratch + random init (ablation baseline):**
- Обучить BPE отдельно на lean/latex корпусе из SciLibModal_v2
- Новые токены получают случайную инициализацию (Xavier)
- Проблема 1: random embeddings далеко от пространства SciRus-tiny — долгая сходимость
- Проблема 2: на ограниченном датасете SciLibModal_v2 ($\sim$10k объектов) может не хватить данных для выхода из случайной инициализации

**Подход B — Vocabulary extension + FVT (Fast Vocabulary Transfer)** [NEED_VERIFY: arXiv:2512.03989]:

**[Определение M.2.3 — FVT-инициализация]** Для нового токена $t_{\text{new}} \in V_{\text{new}}$:

1. Декомпозиция под базовым токенизатором: $D(t_{\text{new}}) = [t_1, t_2, \ldots, t_k] \in V_{\text{base}}^k$
2. Инициализация эмбеддинга:

$$\mathbf{e}_{t_{\text{new}}} = \frac{1}{k} \sum_{i=1}^{k} \mathbf{e}_{t_i}$$

где $\mathbf{e}_{t_i} \in \mathbb{R}^{d_{\text{emb}}}$ — предобученные эмбеддинги из SciRus-tiny.

Для токенов из пересечения $V_{\text{overlap}} = V_m \cap V_{\text{base}}$: эмбеддинги копируются без изменений.

**[Пример]** Токен `\frac` $\notin V_{\text{base}}$. Базовый токенизатор разбивает: $D(\texttt{\textbackslash frac}) = [\texttt{\textbackslash}, \texttt{fr}, \texttt{ac}]$. Инициализация: $\mathbf{e}_{\texttt{\textbackslash frac}} = (\mathbf{e}_{\texttt{\textbackslash}} + \mathbf{e}_{\texttt{fr}} + \mathbf{e}_{\texttt{ac}}) / 3$.

**[Интуиция]** FVT сохраняет новые эмбеддинги в том же регионе пространства, что и предобученные. Это "тёплый старт": новый токен начинает рядом с его субкомпонентами, а не в случайной точке. При fine-tuning он дрейфует к правильному семантическому положению быстрее.

**Выбор:** Подход B (FVT) — основной. Подход A — ablation baseline для оценки вклада инициализации.

**[Assumption A.4*]** Токенизаторы обучаются end-to-end с FVT-инициализацией (заменяет A.4).

### M.2.4 Стратегия обучения [Paper A]

**[Решение]** Все слои backbone обучаются end-to-end с первого шага (full fine-tuning). Трёхфазная стратегия unfreezing (v2.6.2) **DEPRECATED** — избыточна для SciRus-tiny (3 слоя) на датасете ~1M объектов с FVT-инициализацией.

**Обоснование:**
- SciRus-tiny имеет $L = 3$ слоя. Частичная заморозка (top 30% = 1 слой) не даёт значимого эффекта.
- FVT-инициализация (M.2.3) решает проблему "случайных эмбеддингов далеко от пространства модели" — новые токены начинают в корректном регионе.
- Гетерогенная среда (4 текстовых + 1 визуальная модальность) требует адаптации всех слоёв с первого шага: заморозка замедляет кросс-модальное выравнивание.
- WEEP-3 подтвердил: full fine-tuning с $\text{lr\_embed\_ratio} = 0.1$ даёт стабильные результаты.

**Discriminative learning rate:**
- Backbone: $\text{lr}_{\text{backbone}} = \text{lr} \cdot 0.1$
- Projection heads, AlignNet: $\text{lr}_{\text{head}} = \text{lr}$

**[Assumption A.12]** Discriminative LR: $\text{lr\_embed\_ratio} = 0.1$. Все слои обучаемы.

---

## Блок M.3 — Абляционные варианты функции потерь [Paper B]

### M.3.0 Общая структура

Все варианты оперируют на батче $\mathcal{B} = \{o_1, \ldots, o_N\}$ с эмбеддингами $\mathcal{E} = \{e_i^m : i \in [N], m \in \mathcal{M}\}$.

**Auxiliary visual alignment loss.** Во всех экспериментах E1-E7 к основному лоссу добавляется $\mathcal{L}_{\text{visual\_align}}$ (определение в M.1.2*):

$$\mathcal{L}_{E_k}^{\text{total}} = \mathcal{L}_{E_k} + \lambda_{\text{va},t} \cdot \mathcal{L}_{\text{visual\_align}}$$

В E1-E5: $\lambda_{\text{va},t} = \lambda_{\text{va}} = \text{const}$ (статический гиперпараметр).
В E6-E7: $\lambda_{\text{va},t}$ — компонент $\boldsymbol{\lambda}_t$, управляемый fuzzy controller (правило R6, M.6.3).

### M.3.1 E1: Pairwise InfoNCE baseline

**[Мотивация E1]** CLIP [Radford et al., 2021] обучает совместное пространство для **двух** модальностей (изображение, текст) через симметричный InfoNCE. Якорь — эмбеддинг одной модальности, позитив — эмбеддинг другой модальности того же объекта, негативы — эмбеддинги других объектов в батче.

При $M > 2$ модальностей прямое обобщение: рассмотреть **все** $\binom{M}{2} = M(M-1)/2$ пар модальностей и для каждой пары посчитать CLIP-style loss. Для $M = 5$: $\binom{5}{2} = 10$ пар: (en,ru), (en,lean), (en,latex), (en,img), (ru,lean), (ru,latex), (ru,img), (lean,latex), (lean,img), (latex,img).

**[Определение M.3.1]**

Для пары модальностей $(m, m')$, $m \neq m'$:

$$\mathcal{L}_{\text{NCE}}^{m,m'}(i) = -\log \frac{\exp\bigl(\langle \hat{e}_i^m, \hat{e}_i^{m'}\rangle / \tau\bigr)}{\sum_{j=1}^{N} \exp\bigl(\langle \hat{e}_i^m, \hat{e}_j^{m'}\rangle / \tau\bigr)}$$

**[Почему симметричная версия]** В CLIP для пары (image, text) loss считается дважды: $m \to m'$ и $m' \to m$. Роли "якорь" и "позитив" несимметричны в InfoNCE: якорь сравнивается со всеми негативами в знаменателе, позитив — нет. Симметризация устраняет этот перекос и обеспечивает одинаковый градиентный сигнал обоим энкодерам $f_m$ и $f_{m'}$.

$$\mathcal{L}_{\text{pair}}^{m,m'} = \frac{1}{2N} \sum_{i=1}^{N} \Bigl[\mathcal{L}_{\text{NCE}}^{m,m'}(i) + \mathcal{L}_{\text{NCE}}^{m',m}(i)\Bigr]$$

Полный лосс E1:

$$\mathcal{L}_{E1} = \frac{1}{\binom{M}{2}} \sum_{m < m'} \mathcal{L}_{\text{pair}}^{m,m'}$$

где $\binom{M}{2} = \frac{M(M-1)}{2} = 10$ для $M = 5$.

**[Интуиция]** E1 — прямое обобщение CLIP на $M$ модальностей через усреднение по всем $\binom{M}{2}$ парам. Каждая пара получает одинаковый вес — нет механизма приоритизации. Сложность: $O(M^2 \cdot N^2)$.

**[Assumption required]** A.1, A.2.

**[Литература]** CLIP [Radford et al., 2021, Eq. 1] для $M=2$. Обобщение аналогично CMC [Tian et al., 2020, Sec. 3.2].

### M.3.2 E2: Centroid InfoNCE

**[Определение M.3.2]**

**Шаг 1.** Полный центроид: $c_i = \frac{1}{M}\sum_m e_i^m$, $\;\hat{c}_i = c_i / \|c_i\|_2$.

**Шаг 2.** Modality dropout — пертурбированный центроид:

$$\delta_i^m \sim \text{Bernoulli}(1 - p_{\text{drop}}), \quad p_{\text{drop}} = 0.3$$

Гарантия хотя бы одной модальности: если $\sum_m \delta_i^m = 0$, то $\delta_i^{m_{\text{rand}}} := 1$.

$$c_i' = \frac{\sum_m \delta_i^m \cdot e_i^m}{\sum_m \delta_i^m + \varepsilon}, \quad \hat{c}_i' = \frac{c_i'}{\|c_i'\|_2}$$

**Шаг 3.** Контрастный лосс по центроидам:

$$\mathcal{L}_{E2} = \frac{1}{N} \sum_{i=1}^{N} \left[ -\log \frac{\exp\bigl(\langle \hat{c}_i, \hat{c}_i'\rangle / \tau\bigr)}{\sum_{j=1}^{N} \exp\bigl(\langle \hat{c}_i, \hat{c}_j'\rangle / \tau\bigr)} \right]$$

**[Интуиция]** Вместо $O(M^2)$ попарных лоссов — один лосс $O(N^2)$ по объектам. Modality dropout создаёт вариативность: если центроид робастен к удалению модальностей — значит модальности хорошо выровнены.

**[Связь с E1]** E2 vs E1: вместо 10 попарных InfoNCE — 1 центроидный. Вопрос: достаточно ли центроидного сигнала, или нужен alignment term ($\to$ E3)?

### M.3.3 E3: Centroid InfoNCE + alignment + radial regularization

**[Определение M.3.3]**

$$\mathcal{L}_{E3} = \mathcal{L}_{\text{contrast}} + \lambda_{\text{align}} \cdot \mathcal{L}_{\text{align}} + \lambda_{\text{rad}} \cdot \mathcal{L}_{\text{rad}} + \lambda_{\text{reg}} \cdot \mathcal{L}_{\text{reg}}$$

**Контрастная компонента** $\mathcal{L}_{\text{contrast}}$ — из E2 (M.3.2).

**Alignment (выравнивание модальностей вокруг центроида):**

**[Замечание о пространстве]** $\mathcal{L}_{\text{align}}$ и $\mathcal{L}_{\text{rad}}$ оперируют в **ненормализованном** пространстве ($e_i^m$, $c_i$), не на гиперсфере ($\hat{e}_i^m$, $\hat{c}_i$). Это согласовано с M.0.4: нормализация применяется только для вычисления косинусного сходства в $\mathcal{L}_{\text{contrast}}$.

$$\mathcal{L}_{\text{align}} = \frac{1}{NM} \sum_{i=1}^{N} \sum_{m \in \mathcal{M}} \gamma_m \cdot w_i^m \cdot \min\bigl(\|e_i^m - c_i\|^2,\; C_{\text{clip}}\bigr)$$

где:
- $\gamma_m > 0$ — вес модальности (в E3 фиксированный, $\gamma_m = 1\;\forall m$)
- $w_i^m$ — адаптивный вес, фокусирующий обучение на "отстающих" модальностях:

$$w_i^m = \frac{\|e_i^m - c_i\|}{\sum_{m'} \|e_i^{m'} - c_i\| + \varepsilon}$$

- $C_{\text{clip}} > 0$ — порог клиппирования (защита от выбросов)

**[Интуиция]** $\mathcal{L}_{\text{align}}$ тянет каждую модальность к центроиду. Адаптивный вес $w_i^m$ больше для модальностей, которые дальше от центроида — обучение фокусируется на них.

**Radial regularization (гиперсферическое ограничение):**

$$\mathcal{L}_{\text{rad}} = \frac{1}{NM} \sum_{i=1}^{N} \sum_{m \in \mathcal{M}} \bigl(\|e_i^m - c_i\| - \rho\bigr)^2$$

где $\rho > 0$ — целевой радиус гиперсферы вокруг центроида.

**[Интуиция]** $\mathcal{L}_{\text{rad}}$ штрафует модальности, которые находятся слишком далеко **или слишком близко** от поверхности гиперсферы радиуса $\rho$ с центром в $c_i$. При $\rho \to 0$ вырождается в $\mathcal{L}_{\text{align}}$. Ненулевой $\rho$ предотвращает modality collapse (все $e_i^m \to c_i$), сохраняя модальную специфичность.

**[Связь с SETUP_DRAFT]** "Эмбеддинг каждой модальности может находиться на поверхности гиперсферы... радиус сферы задается регуляризационным членом, который штрафует модель если объект сильно далеко или сильно близко от поверхности."

**Anti-collapse регуляризация:**

**[Мотивация]** Representation collapse — известная проблема контрастного обучения [VICReg: Bardes et al., 2022; Barlow Twins: Zbontar et al., 2021]: все эмбеддинги схлопываются в одну точку, минимизируя лосс тривиально ($\mathcal{L} \to 0$) но теряя всю информацию. В VICReg коллапс предотвращается через variance + covariance regularization на матрице эмбеддингов. Barlow Twins требуют диагональность cross-correlation matrix. Мы используем аналогичный принцип, но проще: одно скалярное число $\bar{s}_{\text{neg}}$ вместо полной ковариационной матрицы. Матрица сходства центроидов $S$ при коллапсе стремится к all-ones (все центроиды одинаковы → $\cos \text{sim} = 1$ для всех пар). $\text{ReLU}(\bar{s}_{\text{neg}})$ штрафует только положительное среднее сходство (при $\bar{s}_{\text{neg}} < 0$ негативы достаточно разделены, штрафа нет). $\text{ReLU}(1 - \bar{s}_{\text{pos}})$ — consistency check: self-similarity нормализованных эмбеддингов должна быть 1.

**[Литература]** VICReg [Bardes et al., 2022, Sec. 3.2] — variance + covariance. Barlow Twins [Zbontar et al., 2021, Eq. 1] — cross-correlation.

Матрица сходства центроидов: $S = \hat{c}_i \hat{c}_j^\top \in \mathbb{R}^{N \times N}$

$$\bar{s}_{\text{pos}} = \frac{1}{N} \sum_i S_{ii}, \quad \bar{s}_{\text{neg}} = \frac{1}{N^2 - N} \sum_{i \neq j} S_{ij}$$

$$\mathcal{L}_{\text{reg}} = \text{ReLU}(\bar{s}_{\text{neg}}) + \text{ReLU}(1 - \bar{s}_{\text{pos}})$$

Collapse score: $\text{CS} = \text{clamp}\bigl((\bar{s}_{\text{neg}} - 0.1) / 0.9,\; 0,\; 1\bigr)$

**Adaptive temperature:**

$$\tau_{\text{eff}} = \text{clamp}\bigl(\tau \cdot (1 - \alpha_\tau \cdot (\bar{s}_{\text{neg}} - \tau_{\text{target}})),\; \tau_{\min},\; \tau_{\max}\bigr)$$

**[Связь с v_1]** $\mathcal{L}_{\text{align}}$ и $\mathcal{L}_{\text{reg}}$ идентичны `compute_multimodal_loss` в v_1 [KHALOV-2025, Cell 17].

### M.3.4 E4: Full composite loss, static weights

**[Мотивация E4]** Глобальный лосс $\mathcal{L}_{E3}$ может маскировать проблемы отдельных модальностей. Контрпример: пусть для объекта $o_i$ имеем $e_i^{\text{lean}} = c_i$ (lean-эмбеддинг совпадает с центроидом). Тогда $\|e_i^{\text{lean}} - c_i\|^2 = 0$, $\mathcal{L}_{\text{align}}$ не штрафует lean. При этом $\langle \hat{e}_i^{\text{lean}}, \hat{c}_j \rangle / \tau$ хорошо разделяет объекты (lean "прячется" в центроиде), поэтому $\mathcal{L}_{\text{contrast}}$ тоже мала. Однако lean потеряла собственную семантику: все $e_i^{\text{lean}}$ коллинеарны с $c_i$, cross-modal retrieval lean→ru ломается. Per-modality $\mathcal{L}_{\text{contrast}}^{\text{lean}}$ (E4) обнаруживает это: если $\hat{e}_i^{\text{lean}} \approx \hat{c}_i$ для всех $i$, а центроиды разделены, то retrieval lean→centroid работает, но lean→lean (прямое сравнение) — нет, потому что $\hat{e}_i^{\text{lean}}$ не несёт модальность-специфичной информации.

**[Определение M.3.4]**

$$\mathcal{L}_{E4} = w_g \cdot \mathcal{L}_{\text{global}} + \sum_{m \in \mathcal{M}} w_m \cdot \mathcal{L}_{\text{personal}}^m$$

где $\mathcal{L}_{\text{global}} = \mathcal{L}_{E3}$ и для каждой модальности $m$:

$$\mathcal{L}_{\text{personal}}^m = \mathcal{L}_{\text{align}}^m + \mathcal{L}_{\text{contrast}}^m + \lambda_{\text{reg}} \cdot \mathcal{L}_{\text{reg}}^m$$

**Per-modality alignment** $\mathcal{L}_{\text{align}}^m$ — версия $\mathcal{L}_{\text{align}}$ из M.3.3, ограниченная одной модальностью:

$$\mathcal{L}_{\text{align}}^m = \frac{1}{N} \sum_{i=1}^{N} w_i^m \cdot \min\bigl(\|e_i^m - c_i\|^2,\; C_{\text{clip}}\bigr)$$

где $w_i^m$ — адаптивный вес из M.3.3. Отличие от глобального $\mathcal{L}_{\text{align}}$: суммирование только по объектам $i$ (не по модальностям), нормировка $1/N$ вместо $1/(NM)$, множитель $\gamma_m$ вынесен в модальный вес $w_m$ уровня E4.

Персональная контрастная компонента — InfoNCE от модальности к центроидам:

$$\mathcal{L}_{\text{contrast}}^m(i) = -\log \frac{\exp\bigl(\langle \hat{e}_i^m, \hat{c}_i\rangle / \tau\bigr)}{\sum_{j=1}^{N} \exp\bigl(\langle \hat{e}_i^m, \hat{c}_j\rangle / \tau\bigr)}$$

**Per-modality anti-collapse регуляризация** $\mathcal{L}_{\text{reg}}^m$:

Матрица сходства внутри модальности: $S^m_{ij} = \langle \hat{e}_i^m, \hat{e}_j^m \rangle$

$$\bar{s}_{\text{neg}}^m = \frac{1}{N^2 - N} \sum_{i \neq j} S^m_{ij}$$

$$\mathcal{L}_{\text{reg}}^m = \text{ReLU}(\bar{s}_{\text{neg}}^m)$$

**[Интуиция]** $\mathcal{L}_{\text{reg}}^m$ предотвращает коллапс **внутри** модальности $m$: если все эмбеддинги $\hat{e}_i^m$ сжимаются в одну точку, $\bar{s}_{\text{neg}}^m \to 1$ и штраф растёт. Это дополняет $\mathcal{L}_{\text{reg}}$ из M.3.3, который работает по центроидам (между объектами), а $\mathcal{L}_{\text{reg}}^m$ — по эмбеддингам внутри одной модальности.

**[Связь с M.3.3]** Глобальный $\mathcal{L}_{\text{reg}}$ (M.3.3) считается по $\hat{c}_i$ (центроидам). $\mathcal{L}_{\text{reg}}^m$ считается по $\hat{e}_i^m$ (эмбеддингам одной модальности). Объекты различны: первый предотвращает коллапс центроидов, второй — коллапс модальных представлений.

Веса $\mathbf{w} = (w_g, w_{\text{en}}, w_{\text{ru}}, w_{\text{lean}}, w_{\text{latex}}, w_{\text{img}})$ — **фиксированные**, задаются перед обучением.

**[Интуиция]** $\mathcal{L}_{\text{personal}}^m$ даёт каждой модальности собственный градиентный сигнал. Без него отдельные модальности могут "прятаться" за центроидом.

### M.3.4* Анатомия полного лосса: иерархия весов [Paper B]

**[Определение M.3.4* — Два уровня весов]**

Полный лосс E4+ содержит два уровня весовых коэффициентов:

**Уровень 1** (тип компоненты): $\boldsymbol{\lambda} = \{\lambda_{\text{align}}, \lambda_{\text{rad}}, \lambda_{\text{reg}}, \lambda_{\text{va}}\}$ — относительная важность **типов** потерь (alignment vs regularization vs contrast).

**Уровень 2** (модальность): $\mathbf{w} = \{w_g, w_{\text{en}}, w_{\text{ru}}, w_{\text{lean}}, w_{\text{latex}}, w_{\text{img}}\}$ — относительная важность **модальностей**.

**Полное раскрытие** $\mathcal{L}_{E4}^{\text{total}}$:

$$\mathcal{L} = \underbrace{w_g \cdot \bigl[\mathcal{L}_{\text{contrast}} + \lambda_{\text{align}} \cdot \mathcal{L}_{\text{align}} + \lambda_{\text{rad}} \cdot \mathcal{L}_{\text{rad}} + \lambda_{\text{reg}} \cdot \mathcal{L}_{\text{reg}}\bigr]}_{\text{глобальная компонента (4 терма)}} + \underbrace{\sum_m w_m \cdot \bigl[\mathcal{L}_{\text{align}}^m + \mathcal{L}_{\text{contrast}}^m + \lambda_{\text{reg}} \cdot \mathcal{L}_{\text{reg}}^m\bigr]}_{\text{per-modality (5×3 = 15 термов)}} + \underbrace{\lambda_{\text{va}} \cdot \mathcal{L}_{\text{visual\_align}}}_{\text{visual (1 терм)}}$$

**Атомарные лосс-компоненты и их эффективные веса:**

| # | Компонента | Эффективный вес | Уровень 1 | Уровень 2 |
|---|---|---|---|---|
| 1 | $\mathcal{L}_{\text{contrast}}$ | $w_g$ | (implicit 1) | $w_g$ |
| 2 | $\mathcal{L}_{\text{align}}$ | $w_g \cdot \lambda_{\text{align}}$ | $\lambda_{\text{align}}$ | $w_g$ |
| 3 | $\mathcal{L}_{\text{rad}}$ | $w_g \cdot \lambda_{\text{rad}}$ | $\lambda_{\text{rad}}$ | $w_g$ |
| 4 | $\mathcal{L}_{\text{reg}}$ | $w_g \cdot \lambda_{\text{reg}}$ | $\lambda_{\text{reg}}$ | $w_g$ |
| 5-9 | $\mathcal{L}_{\text{align}}^m$ | $w_m$ | (implicit 1) | $w_m$ |
| 10-14 | $\mathcal{L}_{\text{contrast}}^m$ | $w_m$ | (implicit 1) | $w_m$ |
| 15-19 | $\mathcal{L}_{\text{reg}}^m$ | $w_m \cdot \lambda_{\text{reg}}$ | $\lambda_{\text{reg}}$ (shared) | $w_m$ |
| 20 | $\mathcal{L}_{\text{visual\_align}}$ | $\lambda_{\text{va}}$ | $\lambda_{\text{va}}$ | (independent) |

**Итого:** 20 атомарных термов, 11 свободных параметров ($\tau$ + 4 $\lambda$ + 5 $w_m$ + $w_g$).

**[Интуиция]** Иерархическая структура (11 параметров $\to$ 20 компонентов) — осознанный выбор:
- $\lambda_{\text{reg}}$ shared между global и per-modality → одинаковая сила anti-collapse на обоих уровнях
- $w_m$ масштабирует **все 3** компонента модальности вместе → нет дисбаланса внутри модальности
- Плоская параметризация (20 независимых весов) не имеет этих constraints и склонна к вырождению

**[Таблица: управление весами по экспериментам]**

| Эксперимент | $\boldsymbol{\lambda}$ | $\mathbf{w}$ | $\lambda_{\text{va}}$ | Параметры управления |
|---|---|---|---|---|
| E4 | static | static | static | 0 (всё задано вручную) |
| E5 | implicit (MLP) | implicit (MLP) | static | $\sim$30 (MLP params) |
| E6 | fuzzy-controlled | fuzzy-controlled | fuzzy-controlled | 11 (в $\boldsymbol{\lambda}_t$) |
| E7 | E6 + Lyapunov | E6 + Lyapunov | E6 + Lyapunov | 11 (в $\boldsymbol{\lambda}_t$) |

### M.3.5 E5: Full composite loss, learnable W

**[Мотивация E5]** В E4 веса ($\boldsymbol{\lambda}$, $\mathbf{w}$) фиксированы — нет адаптации к динамике обучения. E5 проверяет гипотезу: может ли нейросеть (MLP) самостоятельно выучить оптимальное распределение весов между компонентами лосса?

LossMixer принимает на вход **текущие значения** всех лосс-компонентов и выдаёт веса. Это data-driven подход (без экспертных правил).

**[Связь с M.3.4*]** LossMixer работает на **плоском** уровне: каждая модальность получает $K = 6$ весов для $K$ компонентов (global align/contrast/reg + personal align/contrast/reg). Иерархия $\boldsymbol{\lambda}$/$\mathbf{w}$ из M.3.4* заменяется одной матрицей $W \in \mathbb{R}^{M \times K}$.

**[Определение M.3.5]**

Веса определяются обучаемым модулем **LossMixer**:

$$\text{comp}_m = \bigl[\mathcal{L}_{\text{align}}^{(g,m)},\; \mathcal{L}_{\text{contrast}}^{(g,m)},\; \mathcal{L}_{\text{reg}}^{(g,m)},\; \mathcal{L}_{\text{align}}^{(p,m)},\; \mathcal{L}_{\text{contrast}}^{(p,m)},\; \mathcal{L}_{\text{reg}}^{(p,m)}\bigr] \in \mathbb{R}^K, \quad K = 6$$

$$W = \text{softmax}\bigl(\text{Linear}_2(\text{GELU}(\text{Linear}_1(\text{comp\_matrix})))\bigr) \in \mathbb{R}^{M \times K}$$

Каждая строка $W_{m,:} \in \Delta^{K-1}$ (симплекс — веса $\geq 0$, сумма $= 1$).

Итоговый лосс модальности: $\mathcal{L}^m = \sum_k W_{m,k} \cdot \text{comp}_{m,k}$

Регуляризация (штраф за вырождение в one-hot):

$$\mathcal{L}_{W\text{-reg}} = \lambda_W \cdot \text{mean}\left(\left(\sum_k W_{m,k}^2 - \frac{1}{K}\right)^2\right)$$

Инициализация: $\text{Linear}_{1,2}.\text{weight} = 0, \;\text{bias} = 0$ $\Rightarrow$ начальные $W = 1/K$ (равномерные).

**[Связь с v_1]** Это точная формализация `LossMixer` из v_1 [KHALOV-2025, Cell 16]. Hidden dim $H = 16$.

### M.3.6 E6: Full composite loss + Fuzzy T-S controller

**[Определение M.3.6]**

$$\mathcal{L}_{E6} = w_{g,t} \cdot \mathcal{L}_{E3}(\boldsymbol{\lambda}_t) + \sum_{m \in \mathcal{M}} w_{m,t} \cdot \mathcal{L}_{\text{personal}}^m(\boldsymbol{\lambda}_t)$$

где $w_{g,t}$ и $w_{m,t}$ — компоненты вектора гиперпараметров $\boldsymbol{\lambda}_t$ (см. ниже).

Гиперпараметры $\boldsymbol{\lambda}_t$ обновляются **не через backprop**, а через T-S fuzzy controller:

$$\boldsymbol{\lambda}_{t+1} = \Pi_\Lambda(\boldsymbol{\lambda}_t + \mathbf{u}_t)$$

$$\mathbf{u}_t = \text{FuzzyTS}(\mathbf{s}_t)$$

где $\Pi_\Lambda$ — проекция на допустимую область (box constraints), $\mathbf{s}_t$ — вектор состояния (M.5), $\mathbf{u}_t$ — выход контроллера (M.6).

Вектор гиперпараметров:

$$\boldsymbol{\lambda}_t = [\underbrace{\tau_t,\; \lambda_{\text{align},t},\; \lambda_{\text{rad},t},\; \lambda_{\text{reg},t},\; \lambda_{\text{va},t}}_{5 \text{ весов типов}},\; \underbrace{w_{\text{en},t},\; w_{\text{ru},t},\; w_{\text{lean},t},\; w_{\text{latex},t},\; w_{\text{img},t},\; w_{g,t}}_{6 \text{ модальных весов}}] \in \mathbb{R}^{11}$$

**Размерность 11** = 5 весов типов ($\tau$, $\lambda_{\text{align}}$, $\lambda_{\text{rad}}$, $\lambda_{\text{reg}}$, $\lambda_{\text{va}}$) + 6 модальных весов ($w_{\text{en}}$, $w_{\text{ru}}$, $w_{\text{lean}}$, $w_{\text{latex}}$, $w_{\text{img}}$, $w_g$).

Допустимая область $\Lambda$ (box constraints): $\tau \in [0.03, 0.3]$, $\lambda_{\text{va}} \in [0.01, 0.5]$, $w_m \in [0.05, 0.5]$, и т.д.

**[Интуиция]** В E5 LossMixer — "чёрный ящик" (MLP через backprop). В E6 fuzzy controller использует символьные правила — можно сказать **почему** вес модальности изменился.

**[Замечание: E5 vs E6]** LossMixer (E5) — MLP с $M \times K + H \times (M \times K + K)$ параметрами ($\approx 30$ params при $K=6$, $H=16$), обучается через backprop. Fuzzy controller (E6) — 7 правил с фиксированной структурой, $A_r \in \mathbb{R}^{11 \times p}$, $\mathbf{b}_r \in \mathbb{R}^{11}$ ($\approx 11(p+1) \cdot R$ params, задаются экспертно). Ключевое различие: LossMixer даёт per-component weights ($W_{m,k}$), fuzzy controller даёт корректировки к глобальным гиперпараметрам ($\Delta\boldsymbol{\lambda}_t$). LossMixer не видит $\mathbf{s}_t$; fuzzy видит тренировочную динамику. Теоретическое обоснование преимущества fuzzy — T.6.

### M.3.7 E7: E6 + Lyapunov smoothing constraint

**[Определение M.3.7]**

$$\mathcal{L}_{E7} = \mathcal{L}_{E6} + \lambda_{\text{lyap}} \cdot \mathcal{L}_{\text{lyap}}$$

где:

$$\mathcal{L}_{\text{lyap}} = \text{ReLU}\bigl(V_t - V_{t-1} + \eta \cdot \Psi_t - \xi\bigr)$$

$V_t$ — функция Ляпунова (M.7), $V_{t-1}$ — её значение на предыдущем шаге, $\Psi_t$ — мера дисбаланса (M.7), $\eta > 0$ — коэффициент затухания, $\xi > 0$ — шумовой потолок.

**[Замечание о каузальности]** Используется $V_t - V_{t-1}$, а **не** $V_{t+1} - V_t$: на шаге $t$ значение $V_{t+1}$ ещё не доступно. Мы штрафуем за *уже наблюдаемый* рост $V_t$ относительно предыдущего шага. Это согласовано с реализацией в TZ.md (`V_current - V_prev`).

**[Интуиция]** $\mathcal{L}_{\text{lyap}}$ — "минимально инвазивный" regularizer: если fuzzy controller сам обеспечивает стабильность, $\mathcal{L}_{\text{lyap}} = 0$. Вмешивается только при нарушении условия убывания $V_t$.

**Цель ablation E6 vs E7:** разделить вклад fuzzy (E6) и Lyapunov (E7). Если $E7 \gg E6$ по метрикам — Lyapunov критически важен. Если $E7 \approx E6$ — Lyapunov декоративен.

### M.3.8 E8: Нелинейные консеквенты (Nonlinear T-S) [Paper C]

**[Определение M.3.8]**

$$\mathcal{L}_{E8} = \mathcal{L}_{E6} \text{ с заменой линейных консеквентов на нелинейные}$$

В E6 каждое правило $r$ имеет линейный консеквент: $\mathbf{u}_t^r = A_r \cdot \mathbf{s}_t + \mathbf{b}_r$. В E8 линейная модель заменяется MLP per rule:

$$\mathbf{u}_t^r = \varphi_r(\mathbf{s}_t)$$

где $\varphi_r: \mathbb{R}^{18} \to \mathbb{R}^{11}$ — двуслойный MLP:

$$\varphi_r(\mathbf{s}_t) = W_2^r \cdot \text{ReLU}(W_1^r \cdot \mathbf{s}_t + \mathbf{b}_1^r) + \mathbf{b}_2^r$$

$$W_1^r \in \mathbb{R}^{h \times 18}, \quad W_2^r \in \mathbb{R}^{11 \times h}, \quad h = 32$$

Fuzzy антецеденты (MF, product t-norm, нормализация) — без изменений относительно E6:

$$\mathbf{u}_t = \sum_r \bar{h}_r(\mathbf{s}_t) \cdot \varphi_r(\mathbf{s}_t)$$

**[Интуиция]** Линейные консеквенты при $\alpha=0.001$ дают слишком малые корректировки — контроллер упирается в границы $\Lambda$ почти сразу. Нелинейные MLP могут адаптивно масштабировать корректировку в зависимости от величины $\mathbf{s}_t$. Параметры $\varphi_r$ обучаются через backprop (градиенты текут через $\mathcal{L}_{\text{lyap}}$).

**[Связь с M.3.6]** Антецеденты (IF-часть) остаются экспертными; изменяются только консеквенты (THEN-часть). Это сохраняет интерпретируемость правил при увеличении выразительности выхода.

**[Assumption required]** [A.1] (нормализация), [A.2] (полный батч). Дополнительно: MLP-консеквенты инициализируются близко к нулю, чтобы в начале обучения поведение совпадало с E6 ($\varphi_r \approx 0$).

### M.3.9 E9: Потенциальные функции (Potential Loss) + learnable W [Paper B]

**[Определение M.3.9]**

Замена $\mathcal{L}_{\text{align}} + \mathcal{L}_{\text{rad}}$ на единый потенциальный лосс, мотивированный аналогией с молекулярной динамикой:

$$\mathcal{L}_{\text{potential}} = U_{\text{attract}} + U_{\text{repel}}$$

**Притяжение к центроиду** (гармонический потенциал):

$$U_{\text{attract}} = \frac{k_a}{NM} \sum_{i=1}^N \sum_{m=1}^M \|e_m^i - c_i\|^2$$

**Отталкивание центроидов** (логарифмический барьер):

$$U_{\text{repel}} = -\frac{k_r}{\binom{N}{2}} \sum_{i < j} \log(\|c_i - c_j\| + \varepsilon)$$

где:
- $k_a > 0$ — жёсткость притяжения (скаляр, рекомендовано: 1.0)
- $k_r > 0$ — сила отталкивания (скаляр, рекомендовано: 0.1)
- $\varepsilon > 0$ — предотвращение $\log(0)$ (рекомендовано: $10^{-6}$)

**[Свойства]**

- **Force-based:** сила $\propto$ расстоянию (не threshold-based как $\mathcal{L}_{\text{rad}}$)
- Притяжение: $F_{\text{attract}} = -\nabla_{e_m^i} U_{\text{attract}} = -2k_a(e_m^i - c_i)$ — сильнее для далёких модальностей
- Отталкивание: $F_{\text{repel}} \propto -1/(\|c_i - c_j\| + \varepsilon)$ — сильнее для близких объектов
- Нет жёсткого порога $\rho$ (проблема: $\rho=0.1$ давал $D_{\text{intra}}=0.007$, 14× ниже цели)

**[Полный лосс E9]**

$$\mathcal{L}_{E9} = \mathcal{L}_{\text{contrast}}^{E2} + \mathcal{L}_{\text{potential}} + \lambda_{\text{reg}} \cdot \mathcal{L}_{\text{ac}} + \text{LossMixer}(\text{components})$$

E9 использует LossMixer (как E5) для обучения весов $W$ через backprop. $k_a$ и $k_r$ заменяют $\lambda_{\text{align}}$ и $\lambda_{\text{rad}}$ в векторе $\boldsymbol{\lambda}_t$.

**[Интуиция]** $\mathcal{L}_{\text{align}} + \mathcal{L}_{\text{rad}}$ имели проблему: $\mathcal{L}_{\text{rad}} = (\|e-c\| - \rho)^2$ с $\rho=0.1$ вызывал over-compression ($D_{\text{intra}} \ll \rho$). Потенциальный лосс не имеет фиксированного целевого радиуса — равновесие определяется балансом сил притяжения и отталкивания.

### M.3.10 E10: Потенциальные функции + Fuzzy controller [Paper C]

**[Определение M.3.10]**

$$\mathcal{L}_{E10} = \mathcal{L}_{\text{contrast}}^{E2} + \mathcal{L}_{\text{potential}} + \lambda_{\text{reg}} \cdot \mathcal{L}_{\text{ac}} + \lambda_{\text{va}} \cdot \mathcal{L}_{\text{va}}$$

Как E6, но с $\mathcal{L}_{\text{potential}}$ вместо $\mathcal{L}_{\text{align}} + \mathcal{L}_{\text{rad}}$.

Контроллер управляет вектором $\boldsymbol{\lambda}_t$, в котором $k_a$ и $k_r$ заменяют $\lambda_{\text{align}}$ и $\lambda_{\text{rad}}$:

$$\boldsymbol{\lambda}_t = [\tau, k_a, k_r, \lambda_{\text{reg}}, \lambda_{\text{va}}, w_{\text{en}}, w_{\text{ru}}, w_{\text{lean}}, w_{\text{latex}}, w_{\text{img}}, w_g] \in \mathbb{R}^{11}$$

**[Цель ablation E9 vs E10]** Если E10 > E9, fuzzy controller улучшает потенциальный подход. Если E10 ≈ E9, learnable W достаточно для potential loss.

---

## Блок M.4 — Матрица весов и её эволюция [Paper B/C]

**[Определение M.4.1 — Эволюция гиперпараметров]**

| Эксперимент | Механизм | Формула |
|---|---|---|
| E4 | Статический | $\boldsymbol{\lambda} = \text{const}$ |
| E5 | Backprop | $W = \text{LossMixer}(\text{comp\_matrix})$ |
| E6 | Fuzzy (linear) | $\boldsymbol{\lambda}_{t+1} = \Pi_\Lambda(\boldsymbol{\lambda}_t + \mathbf{u}_t)$, $\mathbf{u}_t^r = A_r \cdot \mathbf{s}_t + \mathbf{b}_r$ |
| E7 | Fuzzy + Lyapunov | как E6, $+ \mathcal{L}_{\text{lyap}}$ в лоссе |
| E8 | Fuzzy (nonlinear) | как E6, $\mathbf{u}_t^r = \varphi_r(\mathbf{s}_t)$ (MLP per rule) |
| E9 | Potential + Backprop | $\mathcal{L}_{\text{potential}}$ + LossMixer |
| E10 | Potential + Fuzzy | $\mathcal{L}_{\text{potential}}$ + fuzzy controller |

Проекция $\Pi_\Lambda$:

$$[\Pi_\Lambda(\mathbf{x})]_j = \text{clamp}(x_j, \ell_j, u_j)$$

---

## Блок M.5 — Состояние системы и сигналы контроллера [Paper C]

**[Определение M.5.1 — Вектор состояния]**

$$\mathbf{s}_t = \bigl[\underbrace{L_t,\; \Delta L_t,\; \text{EMA}_t,\; \text{Var}_t,\; \text{conflict}_t,\; \text{collapse}_t,\; \|\nabla_\theta L_t\|,\; \Delta R_t}_{\text{Группа 1: агрегированные (8)}},\; \underbrace{L_{\text{en},t},\; L_{\text{ru},t},\; L_{\text{lean},t},\; L_{\text{latex},t},\; L_{\text{img},t}}_{\text{Группа 2: per-modality loss (5)}},\; \underbrace{\text{EMA}_{\text{en},t},\; \text{EMA}_{\text{ru},t},\; \text{EMA}_{\text{lean},t},\; \text{EMA}_{\text{latex},t},\; \text{EMA}_{\text{img},t}}_{\text{Группа 3: per-modality EMA (5)}}\bigr] \in \mathbb{R}^p, \quad p = 18$$

**Группа 1 — Агрегированные сигналы** (8 компонент):

**$L_t$** — aggregate loss на шаге $t$ (скаляр, $L_t \geq 0$).

**$\Delta L_t = L_t - L_{t-1}$** — шаговое изменение ($\Delta L_t < 0$ означает прогресс).

**$\text{EMA}_t$** — экспоненциальное скользящее среднее (aggregate):

$$\text{EMA}_t = \beta \cdot \text{EMA}_{t-1} + (1 - \beta) \cdot \Delta L_t, \quad \beta = 0.9$$

**[Интуиция]** EMA фильтрует шум. $\text{EMA}_t < 0$ — устойчивый прогресс. $\text{EMA}_t > 0$ — стагнация.

**[Связь с SETUP_DRAFT]** "Будем накапливать экспоненциальное скользящее среднее по изменению всех компонентов функции потерь и будем сравнивать то что у нас есть в моменте с этим средним." — Теперь выполнено: EMA считается как для aggregate ($\text{EMA}_t$), так и per-modality ($\text{EMA}_{m,t}$, Группа 3).

**$\text{Var}_t$** — дисперсия модальных потерь:

$$\bar{L}_t = \frac{1}{M} \sum_m L_{m,t}, \quad \text{Var}_t = \frac{1}{M} \sum_m (L_{m,t} - \bar{L}_t)^2$$

$\text{Var}_t \to 0$: сбалансированное обучение. $\text{Var}_t \gg 0$: доминация одной модальности.

**$\text{conflict}_t$** — конфликт градиентов между модальностями:

$$\text{cos\_conflict}^{m,m'} = \frac{\langle \nabla_\theta \mathcal{L}^m,\; \nabla_\theta \mathcal{L}^{m'}\rangle}{\|\nabla_\theta \mathcal{L}^m\| \cdot \|\nabla_\theta \mathcal{L}^{m'}\| + \varepsilon}$$

$$\text{conflict}_t = \frac{1}{\binom{M}{2}} \sum_{m < m'} \max\bigl(0,\; -\text{cos\_conflict}^{m,m'}\bigr) \in [0, 1]$$

$\text{conflict}_t \to 0$: градиенты согласованы. $\text{conflict}_t \to 1$: модальности "тянут" в противоположных направлениях.

**[Assumption A.5]** Вычисление gradient conflict требует $M$ backward passes. Альтернатива: proxy через $\text{Var}_t$ (дешевле, менее точный).

**[Implementation Note — Proxy для gradient conflict]** В текущей реализации $\text{conflict}_t$ аппроксимируется через $\text{Var}_m(L_{m,t})$ вместо прямого вычисления $\cos(\nabla_\theta \mathcal{L}^m, \nabla_\theta \mathcal{L}^{m'})$. Прямое вычисление требует $O(M^2)$ backward passes — непрактично в training loop (при $M=5$ это 10 дополнительных backward passes на каждый шаг). Аппроксимация обоснована: высокая $\text{Var}_m(L_m)$ коррелирует с gradient conflict, т.к. модальности с сильно различающимися loss-значениями генерируют конфликтующие градиентные направления.

**$\text{collapse}_t$** — индикатор коллапса:

$$\text{collapse}_t = \text{clamp}\bigl((\bar{s}_{\text{neg},t} - 0.1) / 0.9,\; 0,\; 1\bigr)$$

**$\|\nabla_\theta L_t\|$** — норма градиента (скаляр):

$$\|\nabla_\theta L_t\| = \sqrt{\sum_j (\partial L_t / \partial \theta_j)^2}$$

**[Интуиция]** Gradient norm характеризует "масштаб" обновления. Резкий рост может сигнализировать о нестабильности, падение до нуля — о стагнации на плато.

**$\Delta R_t$** — drift retrieval метрики:

$$\Delta R_t = R@1_t - R@1_{t-1}$$

вычисляется на валидационном подмножестве каждые $N_{\text{eval}}$ шагов.

**Группа 2 — Per-modality loss** (5 компонент, по одной на каждую модальность):

Для каждой $m \in \mathcal{M}$:

$$L_{m,t} = \mathcal{L}_{\text{personal}}^m \text{ на шаге } t$$

**[Интуиция]** Per-modality loss позволяет контроллеру видеть **какая именно** модальность отстаёт. Без этого сигнала (при $p=8$) контроллер видел только $\text{Var}_t$ — агрегированную дисперсию, но не мог определить, какой $w_m$ корректировать.

**Группа 3 — Per-modality EMA** (5 компонент):

Для каждой $m \in \mathcal{M}$:

$$\Delta L_{m,t} = L_{m,t} - L_{m,t-1}$$

$$\text{EMA}_{m,t} = \beta \cdot \text{EMA}_{m,t-1} + (1 - \beta) \cdot \Delta L_{m,t}, \quad \beta = 0.9$$

**[Интуиция]** $\text{EMA}_{m,t}$ показывает тренд обучения модальности $m$. Если $\text{EMA}_{m,t} > 0$ при $\text{EMA}_t < 0$ — модальность $m$ стагнирует на фоне общего прогресса.

**[Связь с SETUP_DRAFT]** Список сигналов из SETUP_DRAFT: "$\Delta L_t$, EMA loss slope, gradient norm, inter-modality conflict, collapse indicator, retrieval metric drift" — все 6 компонент включены + $L_t$ и $\text{Var}_t$ (Группа 1). Требование "по изменению **всех компонентов** функции потерь" теперь выполнено через per-modality $L_{m,t}$ и $\text{EMA}_{m,t}$ (Группы 2-3).

---

## Блок M.6 — T-S Fuzzy Controller [Paper C]

### M.6.1 Лингвистические переменные

**[Определение M.6.1]**

| Компонент $\mathbf{s}_t$ | Группа | Терм-множество | Сокращения |
|---|---|---|---|
| $\Delta L_t$ | 1 (агр.) | {Negative Large, Negative Small, Zero, Positive Small, Positive Large} | NL, NS, ZE, PS, PL |
| $\text{EMA}_t$ | 1 (агр.) | {Decreasing, Stable, Increasing} | DEC, STB, INC |
| $\text{Var}_t$ | 1 (агр.) | {Low, Medium, High} | LO, MED, HI |
| $\text{conflict}_t$ | 1 (агр.) | {Low, Medium, High} | LO, MED, HI |
| $\text{collapse}_t$ | 1 (агр.) | {Safe, Warning, Critical} | SF, WR, CR |
| $\|\nabla L_t\|$ | 1 (агр.) | {Vanishing, Normal, Exploding} | VAN, NOR, EXP |
| $L_{m,t}$ | 2 (per-m) | {Low, Medium, High} | LO, MED, HI |
| $\text{EMA}_{m,t}$ | 3 (per-m) | {Decreasing, Stable, Increasing} | DEC, STB, INC |

**[Замечание]** Per-modality компоненты ($L_{m,t}$, $\text{EMA}_{m,t}$) используют те же формы функций принадлежности (треугольные, M.6.2), но с **per-modality running statistics** для нормализации (отдельные running mean/std для каждой модальности). Терм-множества одинаковы для всех $m \in \mathcal{M}$ — различия в динамике отражаются через нормализацию.

### M.6.2 Функции принадлежности

**[Определение M.6.2]** Треугольные функции принадлежности:

$$\mu_A(x) = \max\left(0,\; 1 - \frac{|x - c_A|}{w_A}\right)$$

где $c_A$ — центр терма, $w_A$ — ширина.

**[Assumption A.6]** Компоненты $\mathbf{s}_t$ нормализуются через running statistics (running mean/std с momentum) перед применением функций принадлежности.

### M.6.3 Правила T-S

**[Определение M.6.3 — Rule Base]**

Формат правила:

$$\text{Rule } r: \quad \text{IF } [z_1 \text{ is } M_1^r] \;\wedge\; [z_2 \text{ is } M_2^r] \;\wedge\; \ldots \quad \text{THEN } \mathbf{u}_t^r = A_r \mathbf{s}_t + \mathbf{b}_r$$

**T-norm для антецедентов:** product t-norm (даёт ненулевые градиенты везде, в отличие от min t-norm):

$$h_r(\mathbf{s}_t) = \prod_{j \in \text{antecedent}_r} \mu_{M_j^r}(s_t^j)$$

**[Литература]** Выбор product t-norm обоснован в [DIFF-LOGIC-2025, Sec. 4]: product даёт гладкие градиенты, min — разреженные. Для нашей задачи (обучение с backprop через $\mathcal{L}_{\text{lyap}}$) product предпочтителен.

**Нормализация:** $\bar{h}_r(\mathbf{s}_t) = h_r(\mathbf{s}_t) / (\sum_k h_k(\mathbf{s}_t) + \varepsilon)$

**Итоговый выход:**

$$\mathbf{u}_t = \sum_r \bar{h}_r(\mathbf{s}_t) \cdot (A_r \mathbf{s}_t + \mathbf{b}_r)$$

**Нечёткое отрицание:** В антецедентах правил "is not $A$" определяется как дополнение:

$$\mu_{\neg A}(x) = 1 - \mu_A(x)$$

Для комбинированных термов: "$\text{EMA}$ is not DEC" эквивалентно "$\text{EMA}$ is STB or INC", т.е. $\mu_{\text{not DEC}}(x) = \max(\mu_{\text{STB}}(x),\; \mu_{\text{INC}}(x))$.

**Таблица правил (7 правил, покрывающих 6 режимов):**

| # | Режим | Антецедент | Консеквент (направление $\mathbf{u}_t$) |
|---|---|---|---|
| R0 | Default | ни одно правило R1-R6 не активировано сильно ($\max_r \bar{h}_r < \theta_{\text{act}}$) | $\mathbf{u}_t = \mathbf{0}$ (без корректировки) |
| R1 | Modality Imbalance | $\text{Var}$ is HI $\wedge$ $\text{EMA}$ is (STB or INC) | $\uparrow \lambda_{\text{align}}$, $\downarrow w_m$ для $m$ с $L_{m,t}$ is HI |
| R2 | Collapse Risk | $\text{collapse}$ is WR or CR | $\uparrow \lambda_{\text{reg}}$, $\uparrow \tau$ |
| R3 | Gradient Conflict | $\text{conflict}$ is HI $\wedge$ $\Delta L$ is PS/PL | $\downarrow w_m$ для $m$ с $\text{EMA}_{m,t}$ is INC |
| R4 | Stable Convergence | $\text{EMA}$ is DEC $\wedge$ $\text{Var}$ is LO $\wedge$ $\text{collapse}$ is SF | $\mathbf{u}_t \approx \mathbf{0}$ (min intervention) |
| R5 | Stagnation | $\text{EMA}$ is (STB or INC) $\wedge$ $\Delta L$ is ZE | $\uparrow \tau$, $\uparrow \lambda_{\text{align}}$ slightly |
| R6 | Visual Misalignment | $L_{\text{img},t}$ is HI $\wedge$ $\text{EMA}_{\text{img},t}$ is (STB or INC) | $\uparrow \lambda_{\text{va}}$ |

**[Замечание]** R0 обеспечивает безопасный fallback: если входные сигналы не активируют ни одно правило сильно (все $\bar{h}_r$ малы), контроллер не вмешивается. Порог $\theta_{\text{act}}$ — гиперпараметр (рекомендация: $0.05$).

**[Замечание: R6 и визуальная модальность]** Правило R6 реагирует на ситуацию когда визуальный энкодер отстаёт. Антецедент использует $L_{\text{img},t}$ и $\text{EMA}_{\text{img},t}$ из $\mathbf{s}_t$ (Группы 2, 3) как proxy для $\mathcal{L}_{\text{visual\_align}}$: высокий per-modality loss визуальной модальности и отсутствие улучшения (EMA STB/INC) косвенно указывают на слабое визуальное выравнивание. Контроллер увеличивает $\lambda_{\text{va}}$. При снижении $L_{\text{img},t}$ (is LO) R6 деактивируется.

**[Замечание: R1 и per-modality сигналы]** Правило R1 (Modality Imbalance) использует per-modality компоненты $\mathbf{s}_t$ (Группа 2): для каждой модальности $m$, матрица $A_{R1}$ содержит отрицательные веса на позициях $(w_m, L_{m,t})$ — высокий per-modality loss → снижение $w_m$ этой модальности. Это позволяет контроллеру **адресно** перераспределять веса, а не слепо реагировать на $\text{Var}_t$.

**[Определение M.6.3* — Конкретные матрицы $A_r$, $\mathbf{b}_r$ для ключевых правил]**

Напомним индексацию:

$\mathbf{s}_t \in \mathbb{R}^{18}$:
Группа 1 (indices 1-8): $L_t, \Delta L_t, \text{EMA}_t, \text{Var}_t, \text{conflict}_t, \text{collapse}_t, \|\nabla L\|, \Delta R_t$
Группа 2 (indices 9-13): $L_{\text{en}}, L_{\text{ru}}, L_{\text{lean}}, L_{\text{latex}}, L_{\text{img}}$
Группа 3 (indices 14-18): $\text{EMA}_{\text{en}}, \text{EMA}_{\text{ru}}, \text{EMA}_{\text{lean}}, \text{EMA}_{\text{latex}}, \text{EMA}_{\text{img}}$

$\mathbf{u}_t \in \mathbb{R}^{11}$:
indices 1-11: $\tau, \lambda_{\text{align}}, \lambda_{\text{rad}}, \lambda_{\text{reg}}, \lambda_{\text{va}}, w_{\text{en}}, w_{\text{ru}}, w_{\text{lean}}, w_{\text{latex}}, w_{\text{img}}, w_g$

**R1 (Modality Imbalance):** $A_{R1} \in \mathbb{R}^{11 \times 18}$ — разреженная, $\sim$12 ненулевых элементов.

Принцип: высокий per-modality loss $L_{m,t}$ → *снижение* веса $w_m$ этой модальности (дать ей "отдохнуть"),
одновременно *увеличение* $\lambda_{\text{align}}$ (усилить стягивание к центроиду).

Ненулевые элементы:

| Строка ($u$-компонент) | Столбец ($s$-компонент) | Значение | Интуиция |
|---|---|---|---|
| $\lambda_{\text{align}}$ (2) | $\text{Var}_t$ (4) | $+\alpha_1$ | Высокий дисбаланс → усилить alignment |
| $w_{\text{en}}$ (6) | $L_{\text{en}}$ (9) | $-\alpha_2$ | Высокий loss en → снизить $w_{\text{en}}$ |
| $w_{\text{ru}}$ (7) | $L_{\text{ru}}$ (10) | $-\alpha_2$ | Аналогично для ru |
| $w_{\text{lean}}$ (8) | $L_{\text{lean}}$ (11) | $-\alpha_2$ | Аналогично для lean |
| $w_{\text{latex}}$ (9) | $L_{\text{latex}}$ (12) | $-\alpha_2$ | Аналогично для latex |
| $w_{\text{img}}$ (10) | $L_{\text{img}}$ (13) | $-\alpha_2$ | Аналогично для img |
| $w_{\text{en}}$ (6) | $\text{EMA}_{\text{en}}$ (14) | $-\alpha_3$ | Растущий тренд → дополнительно снизить |
| $w_{\text{ru}}$ (7) | $\text{EMA}_{\text{ru}}$ (15) | $-\alpha_3$ | ... |
| $w_{\text{lean}}$ (8) | $\text{EMA}_{\text{lean}}$ (16) | $-\alpha_3$ | ... |
| $w_{\text{latex}}$ (9) | $\text{EMA}_{\text{latex}}$ (17) | $-\alpha_3$ | ... |
| $w_{\text{img}}$ (10) | $\text{EMA}_{\text{img}}$ (18) | $-\alpha_3$ | ... |
| $w_g$ (11) | $\text{Var}_t$ (4) | $+\alpha_4$ | Дисбаланс → усилить global (стягивание к среднему) |

$\mathbf{b}_{R1} = \mathbf{0}$ (нет bias: при нулевом $\mathbf{s}_t$ коррекция нулевая).

Масштабы: $\alpha_1, \alpha_2, \alpha_3, \alpha_4 > 0$ — гиперпараметры (задаются в config, рекомендация: $\alpha_2 \sim 0.1$, $\alpha_3 \sim 0.05$, $\alpha_1 \sim \alpha_4 \sim 0.02$).

**[Интуиция]** $A_{R1}$ кодирует **отрицательную обратную связь**: модальность с высоким loss получает **меньший** вес (в отличие от SGD, который увеличивает градиент для модальности с высоким loss — см. T.6). Два уровня реакции: мгновенный ($L_{m,t}$, коэффициент $\alpha_2$) и трендовый ($\text{EMA}_{m,t}$, коэффициент $\alpha_3 < \alpha_2$). Трендовый — мягче, но устойчивее к шуму.

**R2 (Collapse Risk):** $A_{R2} \in \mathbb{R}^{11 \times 18}$ — ещё более разреженная, 3 ненулевых элемента.

| Строка ($u$-компонент) | Столбец ($s$-компонент) | Значение | Интуиция |
|---|---|---|---|
| $\tau$ (1) | $\text{collapse}_t$ (6) | $+\alpha_5$ | Collapse → увеличить $\tau$ (размягчить softmax) |
| $\lambda_{\text{reg}}$ (4) | $\text{collapse}_t$ (6) | $+\alpha_6$ | Collapse → усилить anti-collapse регуляризацию |
| $\lambda_{\text{align}}$ (2) | $\text{collapse}_t$ (6) | $-\alpha_7$ | Collapse → ослабить alignment (не стягивать ещё сильнее) |

$\mathbf{b}_{R2} = \mathbf{0}$.

**[Интуиция]** При коллапсе все эмбеддинги стягиваются в одну точку. Контроллер одновременно: (1) увеличивает температуру $\tau$, делая contrastive loss менее жёстким; (2) усиливает anti-collapse $\lambda_{\text{reg}}$; (3) ослабляет alignment $\lambda_{\text{align}}$ — потому что alignment *стягивает* модальности к центроиду, что при коллапсе контрпродуктивно.

**[Замечание]** Остальные $A_{R3}$-$A_{R6}$ конструируются по тому же принципу (разреженность, отрицательная обратная связь). Полное определение всех матриц — задача TZ.md/реализации (config YAML). Здесь $A_{R1}$, $A_{R2}$ даны как показательные примеры, достаточные для теоремы T.4.

**[Замечание: R3 и proxy через $L_{m,t}$]** Вычисление per-modality gradient conflict требует $M$ backward passes (A.5), что дорого. Правило R3 использует $\text{EMA}_{m,t}$ (Группа 3) как proxy: модальность с $\text{EMA}_{m,t}$ is INC (растущий тренд потерь) при общем конфликте — вероятный источник конфликта. Это даёт $O(1)$ оценку вместо $O(M)$ backward passes.

**[Связь с SETUP_DRAFT]** Правила R1-R3 соответствуют "regimes A-C": (A) one modality dominates, (B) global loss decreases but one branch diverges, (C) collapse risk. R4 = regime D (stable convergence).

**[Литература]** T-S model: [WANG-1996, Sec. II]. PDC controller design: [WANG-1996, Sec. IV].

**Механизм сравнения с EMA:** В правилах R1, R5 используется сравнение текущего $\Delta L_t$ с $\text{EMA}_t$. Если $\Delta L_t \gg \text{EMA}_t$ (текущий шаг значительно хуже тренда) — усиленная реакция. Если $\Delta L_t \approx \text{EMA}_t$ — штатная динамика.

### M.6.3b Стохастическое расширение T-S контроллера (Variant D)

**[Определение M.6.3b — Stochastic T-S Update with Elastic Reversion]**

Детерминированный T-S контроллер (M.6.3) при взаимодействии с box constraints $\Lambda$ склонен к corner-crashing: линейные консеквенты $A_r \mathbf{s}_t + \mathbf{b}_r$ систематически смещают $\boldsymbol{\lambda}_t$ к границам $\Lambda$, после чего проекция $\Pi_\Lambda$ фиксирует значения на bounds (EXP-001: все $w_m \to \ell_m$, $w_g \to u_{w_g}$, loss не сходится).

**Расширение (Variant D — hybrid):**

$$\mathbf{u}_t^{\det} = \sum_{r=0}^{R-1} \bar{h}_r(\tilde{\mathbf{s}}_t) \cdot (A_r \tilde{\mathbf{s}}_t + \mathbf{b}_r)$$

$$\boldsymbol{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \sigma_t^2 I), \quad \sigma_t = \sigma_0 \cdot \max\bigl(0, 1 - t/T\bigr) \quad \text{(linear annealing)}$$

$$\mathbf{u}_t = \mathbf{u}_t^{\det} + \boldsymbol{\varepsilon}_t$$

$$\boldsymbol{\lambda}_{t+1} = \Pi_\Lambda\bigl(\boldsymbol{\lambda}_t + \alpha \cdot \mathbf{u}_t + \gamma \cdot (\boldsymbol{\lambda}_0 - \boldsymbol{\lambda}_t)\bigr)$$

где:
- $\alpha = 0.001$ — шаг контроллера (reduced from $0.01$)
- $\gamma = 0.01$ — коэффициент elastic mean-reversion к $\boldsymbol{\lambda}_0$
- $\sigma_0 = 0.01$ — начальная дисперсия стохастического шума
- $\boldsymbol{\lambda}_0$ — default значения гиперпараметров (начальная точка из конфигурации)
- $\Lambda$ — сужённые bounds: $w_m \in [0.3, 3.0]$ (was $[0.1, 5.0]$)
- Контроллер применяется каждые $K=10$ шагов (step frequency)
- Warmup: $200$ шагов (raw state without controller intervention)

**[Интуиция]** Три механизма предотвращают corner-crashing:
1. **Elastic reversion** $\gamma(\boldsymbol{\lambda}_0 - \boldsymbol{\lambda}_t)$: создаёт "пружину" к default значениям, сопротивляющуюся drift к boundaries.
2. **Stochastic noise** $\boldsymbol{\varepsilon}_t$: обеспечивает exploration; annealing постепенно убирает шум к финалу обучения.
3. **Narrow bounds** $\Lambda$: уменьшают амплитуду возможного drift.

**[Связь с M.6.3]** M.6.3b — расширение M.6.3, а не замена. При $\sigma_0 = 0$, $\gamma = 0$, $K = 1$ формула M.6.3b вырождается в M.6.3. BIBO-устойчивость (T.4) сохраняется, т.к. $\Pi_\Lambda$ по-прежнему обеспечивает bounded output.

### M.6.4 Связь с онтологией (OWL 2 / SWRL) — future work

**[Замечание]** В текущей версии T-S fuzzy controller реализуется как чисто численный модуль (M.6.1-M.6.3) без интеграции с OWL 2 онтологией. Онтологическая формализация обучающих режимов (T-Box для концептов `TrainingRegime`, `CollapseRisk`, `ModalityImbalance`; SWRL-правила для связки DL-концептов с fuzzy-метками) планируется как расширение для интеграции с SciLib-инфраструктурой:

- **T-Box:** `CollapseRisk ⊑ TrainingRegime ⊓ ∃hasCollapse.HighValue`
- **SWRL:** `CollapseIndicator(?s) ∧ hasValue(?s, ?v) ∧ swrlb:greaterThan(?v, θ_c) → assertFuzzyLabel(?s, "CollapseRisk", μ_collapse(?v))`

Это future work, не входящее в текущий экспериментальный scope (E1-E7). Причина: онтологический слой добавляет инфраструктурную сложность без непосредственного влияния на метрики обучения. Его ценность — в интеграции с SciLib GraphDB для аудита и визуализации обучающих режимов.

**[Литература]** DL/OWL терминология — паттерн [P.4] из CLAUDE.md.

---

## Блок M.7 — Lyapunov мета-теория [Paper C]

### M.7.1 Функция Ляпунова

**[Определение M.7.1]**

$$V_t = \alpha \cdot \tilde{L}_t + \beta \cdot \|\Delta\boldsymbol{\lambda}_t\|^2 + \gamma \cdot \text{Var}_m(w_{m,t})$$

где:
- $\tilde{L}_t = \text{EMA}(L_t)$ — сглаженный aggregate loss
- $\Delta\boldsymbol{\lambda}_t = \boldsymbol{\lambda}_t - \boldsymbol{\lambda}_{t-1}$ — изменение гиперпараметров
- $\text{Var}_m(w_{m,t}) = \frac{1}{M}\sum_m(w_{m,t} - \bar{w}_t)^2$ — дисперсия весов модальностей
- $\alpha, \beta, \gamma > 0$ — весовые коэффициенты

**Свойства:** $V_t \geq 0$ при $\alpha, \beta, \gamma > 0$ и $\tilde{L}_t \geq 0$. $V_t = 0$ только при $\tilde{L}_t = 0$, $\Delta\boldsymbol{\lambda}_t = \mathbf{0}$, $w_{m,t} = \text{const}\;\forall m$.

**[Интуиция]** Три слагаемых кодируют три цели:
1. $\alpha \tilde{L}_t$: хотим чтобы лосс убывал
2. $\beta \|\Delta\boldsymbol{\lambda}_t\|^2$: хотим smooth trajectory (без "прыжков" гиперпараметров)
3. $\gamma \text{Var}_m(w_{m,t})$: хотим balance модальностей

**[Связь с SETUP_DRAFT]** Версия без $\boldsymbol{\lambda}^*$: "λ* ты обычно не знаешь".

**[Литература]** CLF для нелинейных систем: [CLF-CBF-SURVEY, Sec. II]. Fuzzy Lyapunov function (convex blend): [MOZELLI-2010, Theorem 2].

### M.7.2 Стохастическое условие убывания

**[Определение M.7.2 — Descent Condition]**

$$\mathbb{E}[V_{t+1} - V_t \mid \mathbf{s}_t] \leq -\eta \cdot \Psi_t + \xi$$

**[Замечание]** Это *целевое* условие, а не формально доказанное. В M.7.3 мы реализуем его как soft constraint ($\mathcal{L}_{\text{lyap}}$ штрафует $V_t - V_{t-1}$, что является эмпирическим приближением к этому условию). Полное доказательство потребовало бы знания распределения SGD-шума и L-гладкости по $\theta$ (не только по $\boldsymbol{\lambda}$), что выходит за рамки данной работы.

где:
- $\eta > 0$ — коэффициент затухания
- $\xi > 0$ — шумовой потолок
- $\Psi_t$ — мера дисбаланса/коллапса/конфликта:

$$\Psi_t = w_{\Psi_1} \cdot \text{Var}_t + w_{\Psi_2} \cdot \text{conflict}_t + w_{\Psi_3} \cdot \text{collapse}_t$$

**[Интуиция]** Если $\Psi_t > \xi / \eta$ (система далека от "здорового" состояния), правая часть отрицательна $\Rightarrow$ $V_t$ должна убывать в среднем. Если $\Psi_t \approx 0$ (система здорова), допускается рост $V_t$ на $\xi$ (стохастический шум SGD).

**ЧЕСТНАЯ ПОСТАНОВКА:**

> *We do not attempt to prove global convergence of the full nonconvex training process. Instead, we model the hyperparameter adaptation layer as a controlled dynamical subsystem driven by observable training signals.*

Мы показываем (или эмпирически верифицируем) три свойства:
1. **Bounded trajectories:** $\boldsymbol{\lambda}_t \in \Lambda\;\forall t$ (box constraints + $\Pi_\Lambda$)
2. **Absence of oscillatory switching:** $\|\Delta\boldsymbol{\lambda}_t\|$ bounded
3. **Monotone tendency:** $V_t$ убывает в среднем при $\Psi_t > \xi/\eta$

**[Assumption A.7]** Bounded gradient variance — стандартное допущение в анализе SGD [NEED_VERIFY: Bottou et al., 2018].

**[Assumption A.8]** (L-smoothness) Функция потерь $\mathcal{L}(\theta, \boldsymbol{\lambda})$ является $L$-гладкой по $\boldsymbol{\lambda}$ при фиксированном $\theta$:

$$\|\nabla_{\boldsymbol{\lambda}} \mathcal{L}(\theta, \boldsymbol{\lambda}_1) - \nabla_{\boldsymbol{\lambda}} \mathcal{L}(\theta, \boldsymbol{\lambda}_2)\| \leq L_\lambda \|\boldsymbol{\lambda}_1 - \boldsymbol{\lambda}_2\|$$

Это необходимо для того, чтобы стохастическое условие убывания (M.7.2) было осмысленным: без L-гладкости малые изменения $\boldsymbol{\lambda}$ могут вызывать произвольно большие скачки в $V_t$.

**[Следствие T.4(b)]** (Bounded controller output — ранее Assumption A.9) Выход fuzzy controller ограничен:

$$\|\mathbf{u}_t\| \leq U_{\max} \quad \forall t, \; \forall \mathbf{s}_t$$

Это **доказано** в Теореме T.4(b) (M.9): из partition of unity ($\sum_r \bar{h}_r = 1$), bounded state (A.6), и фиксированности $A_r, \mathbf{b}_r$ следует $U_{\max} = \max_r (\|A_r\| \cdot S_{\max} + \|\mathbf{b}_r\|) < \infty$.

### M.7.2b Soft Lyapunov Constraint (реализация)

**[Определение M.7.2b — Soft Lyapunov Constraint]**

$$\mathcal{L}_{\text{lyapunov}} = w_{\text{lyap}} \cdot \max\bigl(0,\; \Delta V_t - \xi\bigr)$$

где $\Delta V_t = V_t - V_{t-1}$, $\xi > 0$ — порог допустимого роста.

**[Отличие от M.7.3]** Ранняя версия (M.7.3) штрафовала $\max(0, V_t - V_{t-1} + \eta\Psi_t - \xi)$ — сложная формула с $\Psi_t$. Упрощённая версия M.7.2b штрафует только **рост** $V_t$ сверх порога $\xi$, без привлечения $\Psi_t$. Это:
- Проще в реализации и отладке
- Не усиливает сигнал broken controller (EXP-001: когда контроллер corner-crashed, сложный Lyapunov penalty усиливал деструктивный сигнал)
- Допускает bounded fluctuations ($\Delta V_t \leq \xi$) без penalty

**[Связь с M.7.2]** Условие M.7.2 ($\mathbb{E}[\Delta V_t] \leq -\eta\Psi_t + \xi$) — теоретическое. M.7.2b — его практическая реализация в виде мягкого ограничения.

### M.7.3 Реализация через loss regularizer

На практике мы **не решаем LMI** и не доказываем устойчивость через матричные неравенства [WANG-1996]. Причина: классические LMI для T-S систем предполагают (i) непрерывное время, (ii) детерминированную динамику, (iii) полностью известную модель. Наша система — дискретная, стохастическая, с неизвестной полной моделью SGD-оптимизации. LMI-литература используется как **концептуальная аналогия** для обоснования архитектуры контроллера, а не как прямой инструмент доказательства.

Вместо LMI мы добавляем soft constraint (loss regularizer):

$$\mathcal{L}_{\text{lyap}} = \lambda_{\text{lyap}} \cdot \max\bigl(0,\; V_t - V_{t-1} + \eta \Psi_t - \xi\bigr)$$

Это концептуально аналогично CLF-QP фильтру [CLF-CBF-SURVEY, Sec. V]: fuzzy controller = "номинальное управление", Lyapunov regularizer = "safety filter".

**[Замечание о two-timescale анализе]** Полная система имеет две шкалы: (1) быстрая — SGD обновляет $\theta$, (2) медленная — fuzzy controller обновляет $\boldsymbol{\lambda}$ каждые $N_{\text{ctrl}}$ шагов. Строгий two-timescale convergence analysis (в духе Borkar, 2008) требует дополнительных условий (Lipschitz-непрерывность оператора усреднения, separation of timescales), которые мы не проверяем формально. Вместо этого мы верифицируем стабильность эмпирически через метрики collapse score и variance across seeds (E6 vs E7).

**[Связь с SETUP_DRAFT]** "T-S даёт грубое направление, Ляпунов сглаживает, для того чтобы не было дисбаланса."

---

## Блок M.8 — Обобщения и теоретические результаты

### M.8.1 Масштабируемость по M

**[Утверждение M.8.1 — Constructive]** Формализация M.3-M.7 параметрическая по $M$: при $M = 2$ сводится к CLIP, при $M = 1$ вырождается в single-modality encoder. Все определения (центроид, modality dropout, T-S controller, Lyapunov) корректны для произвольного $M \geq 2$.

### M.8.2 Centroid InfoNCE vs Pairwise: сложность

**[Утверждение M.8.2 — Observation]** Вычислительная сложность за один батч:

| Метод | Лосс-вычислений | Сложность |
|---|---|---|
| E1 (Pairwise) | $\binom{M}{2}$ InfoNCE | $O(M^2 N^2 d)$ |
| E2 (Centroid) | 1 InfoNCE + centroid | $O(N^2 d + NMd)$ |

Для $M = 5$: E1 требует $10\times$ больше InfoNCE-вычислений, чем E2.

### M.8.3 LossMixer как частный случай T-S

**[Утверждение M.8.3 — Observation]** LossMixer (E5) — нейросетевая аппроксимация T-S controller (E6) с ограничениями:
- LossMixer видит только текущие loss values (без $\Delta L$, EMA, conflict, gradient norm)
- LossMixer — "чёрный ящик", T-S — интерпретируемый
- LossMixer обучается через backprop, T-S — заданные правила
- LossMixer не имеет stability guarantees, T-S + Lyapunov — имеет (bounded trajectories)

### M.8.4 Честная оценка теоретического вклада

| Результат | Тип | Статус | Ограничения |
|---|---|---|---|
| **T.1:** Centroid InfoNCE — нижняя граница $I(C; C')$ | Theorem | Доказано (прямое следствие InfoNCE bound) | Ограничивает $I(C; C')$, НЕ $I(e^m; e^{m'})$ |
| **T.2:** Редукция к CLIP при $M = 2$ | Theorem | Доказано (алгебраическое разложение) | Приближение при хорошем alignment |
| **T.3:** Retrieval guarantee с margin | Theorem | Доказано (triangle inequality) | Требует A.10 (bounded $D_{\text{intra}}$ на eval set) |
| **T.4:** BIBO-устойчивость T-S controller | Theorem | Доказано (конструктивно, через $\Pi_\Lambda$) | BIBO, не асимптотическая устойчивость. Заменяет A.9. Сохраняется для M.6.3b (Variant D) |
| **T.5:** Условное убывание Ляпунова | Theorem | Доказано при выполнении C1-C3 | C1-C3 эмпирически мониторятся, не гарантируются. Soft constraint M.7.2b |
| **T.6:** SGD для модальных весов — квадратичный рост Var | Proposition | Доказано (алгебра + Лемма b') | Стационарность $\mathcal{L}^m$ (two-timescale). T.6(c) — design argument |
| **T.7:** Gradient signal coverage: Centroid vs Pairwise | Theorem | Доказано (chain rule + combinatorics) | (c) предполагает независимость pairwise градиентов. (d) GC > 0 — эмпирическое |
| **T.8:** Sample complexity для centroid retrieval | Theorem | Доказано (Bernstein + union bound) | Bound пессимистичен (union bound). Ценность — качественный результат $N_{\max} \propto \exp(CM)$ |
| **T.9:** Generalization bound: Centroid vs Pairwise | Theorem | Доказано (variance reduction + Bernstein bound) | A.14 (independence) — приближение для Family A. Не работает для Family B |
| $M$-масштабируемость (M.8.1) | Observation | Конструктивное наблюдение | Не тестировалось для $M \gg 5$ |
| Complexity comparison (M.8.2) | Observation | Прямое вычисление | — |
| LossMixer как частный случай T-S (M.8.3) | Observation | Структурное сравнение | — |

**Необходимые базовые сравнения (baselines):**
- E1 (pairwise InfoNCE) vs CMC [CMC-2020] — наш baseline покрывает эту архитектуру
- E5 (LossMixer) vs GradNorm [GRADNORM-2018] — gradient-based vs signal-based балансировка
- ImageBind-style: anchored к одной модальности [IMAGEBIND-2023]
- PCGrad / CAGrad: gradient surgery подходы [PCGRAD-2020; CAGRAD-2021]

Если ни одно обобщение не подтвердится эмпирически — это "engineered system, empirically validated" (а не theoretical contribution). Это честно. Однако теоремы T.1-T.9 обеспечивают **минимальный теоретический фундамент**: MI-связь (T.1), согласованность с литературой (T.2), гарантию retrieval (T.3), ограниченность траекторий (T.4), условную стабильность (T.5), обоснование выбора fuzzy над SGD (T.6), преимущество centroid по gradient coverage (T.7), масштабируемость retrieval по числу модальностей (T.8), и generalization advantage centroid (T.9).

---

## Блок M.9 — Теоретические результаты [Paper B, C]

> Этот блок содержит 9 формальных результатов (T.1-T.9) с полными доказательствами. Каждая теорема предваряется вопросом, на который она отвечает, и объяснением почему этот ответ важен. Все теоремы сопровождаются честными ограничениями.

### T.1 — Centroid InfoNCE как нижняя граница взаимной информации [Paper B]

**[Вопрос]** Является ли Centroid InfoNCE (E2) обоснованным с точки зрения теории информации? Даёт ли он нижнюю границу взаимной информации?

**[Почему важно]** Если да — это теоретическое обоснование перехода от pairwise (E1) к centroid (E2): центроид **не теряет** информацию, а InfoNCE от центроида всё ещё валидная MI lower bound. Без T.1 переход E1→E2 — чисто эвристический.

**[Теорема T.1]**

Пусть $C_i = c_i / \|c_i\|_2$ — нормализованный полный центроид объекта $o_i$, а $C_i'$ — нормализованный центроид после modality dropout (M.3.2). Тогда:

$$I(C; C') \geq \log N - \mathcal{L}_{E2}$$

где $I(C; C')$ — взаимная информация между $C$ и $C'$, $N$ — размер батча.

**[Доказательство]**

Прямое следствие стандартного результата InfoNCE [van den Oord et al., 2018, Proposition 1].

Шаг 1. InfoNCE loss определяется как:

$$\mathcal{L}_{\text{NCE}}(X; Y) = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(f(x_i, y_i))}{\sum_{j=1}^N \exp(f(x_i, y_j))}$$

Шаг 2. По [van den Oord, 2018, Prop. 1], для любого scoring function $f$ и любых случайных величин $X, Y$:

$$I(X; Y) \geq \log N - \mathcal{L}_{\text{NCE}}(X; Y)$$

Шаг 3. Подставляем $X = C$, $Y = C'$, $f(C_i, C_j') = \langle C_i, C_j' \rangle / \tau$:

$$I(C; C') \geq \log N - \mathcal{L}_{E2} \qquad \square$$

**[Честное ограничение]** Теорема T.1 ограничивает взаимную информацию $I(C; C')$ между **полным и dropout-центроидом** одного объекта. Она **НЕ** утверждает, что $\mathcal{L}_{E2}$ оценивает $I(e^m; e^{m'})$ — взаимную информацию между отдельными модальностями. Centroid InfoNCE — дискриминативный objective, не MI estimator между произвольными парами модальностей.

**[Assumption required]** A.1, A.2.

---

### T.2 — Редукция к CLIP при $M = 2$ [Paper B]

**[Вопрос]** Совместим ли наш Centroid InfoNCE с классическим CLIP? Вырождается ли он в CLIP при $M = 2$?

**[Почему важно]** Если да — наш подход является **строгим обобщением** CLIP, а не независимой конструкцией. Это усиливает narrative: CLIP — частный случай при $M = 2$, мы — общий случай при $M \geq 2$. Рецензент не может отклонить подход как "не связанный с CLIP".

**[Теорема T.2]**

При $M = 2$ и $p_{\text{drop}} = 0.5$ ожидаемый Centroid InfoNCE (E2) сводится к смеси двух стандартных InfoNCE (как в CLIP) с поправочным множителем от нормализации центроида.

Формально: пусть $M = 2$, модальности $\{a, b\}$. При $p_{\text{drop}} = 0.5$ и Bernoulli dropout единственные ненулевые варианты dropout-центроида:

$$c_i' = e_i^a \quad (\text{с вероятностью } 1/2), \quad c_i' = e_i^b \quad (\text{с вероятностью } 1/2)$$

(случай обоих выключенных заменяется случайным выбором одной модальности, случай обоих включённых даёт $c_i' = c_i$). Тогда:

$$\mathbb{E}[\mathcal{L}_{E2}] = \frac{1}{2}\mathcal{L}_{\text{NCE}}(\hat{c}, \hat{e}^a) + \frac{1}{2}\mathcal{L}_{\text{NCE}}(\hat{c}, \hat{e}^b) + \text{residual}(\hat{c}, \hat{c})$$

**[Доказательство]**

Шаг 1. При $M = 2$ центроид $c_i = (e_i^a + e_i^b) / 2$.

Шаг 2. Modality dropout с $p_{\text{drop}} = 0.5$: каждая модальность выключается с вероятностью $1/2$. Возможные исходы для $(δ^a, δ^b)$:
- $(1, 1)$: вероятность $1/4$, $c_i' = c_i$
- $(1, 0)$: вероятность $1/4$, $c_i' = e_i^a$
- $(0, 1)$: вероятность $1/4$, $c_i' = e_i^b$
- $(0, 0)$: вероятность $1/4$, $c_i' = e_i^{m_{\text{rand}}}$ (случайная из двух)

Шаг 3. Ожидание по dropout:

$$\mathbb{E}[\mathcal{L}_{E2}] = \frac{1}{4}\mathcal{L}_{\text{NCE}}(\hat{c}, \hat{c}) + \frac{1}{4}\mathcal{L}_{\text{NCE}}(\hat{c}, \hat{e}^a) + \frac{1}{4}\mathcal{L}_{\text{NCE}}(\hat{c}, \hat{e}^b) + \frac{1}{4} \cdot \frac{1}{2}\bigl[\mathcal{L}_{\text{NCE}}(\hat{c}, \hat{e}^a) + \mathcal{L}_{\text{NCE}}(\hat{c}, \hat{e}^b)\bigr]$$

$$= \frac{1}{4}\mathcal{L}_{\text{NCE}}(\hat{c}, \hat{c}) + \frac{3}{8}\mathcal{L}_{\text{NCE}}(\hat{c}, \hat{e}^a) + \frac{3}{8}\mathcal{L}_{\text{NCE}}(\hat{c}, \hat{e}^b)$$

Шаг 4. При хорошем alignment ($e^a \approx e^b$) имеем $c \approx e^a \approx e^b$, и $\mathcal{L}_{\text{NCE}}(\hat{c}, \hat{c}) \approx 0$ (тривиальный случай). Тогда:

$$\mathbb{E}[\mathcal{L}_{E2}] \approx \frac{3}{8}\bigl[\mathcal{L}_{\text{NCE}}(\hat{c}, \hat{e}^a) + \mathcal{L}_{\text{NCE}}(\hat{c}, \hat{e}^b)\bigr]$$

что является взвешенной версией CLIP с якорем в центроиде (а не в одной модальности). $\square$

**[Честное ограничение]** Редукция точная при $e^a \approx e^b$ (хороший alignment). Для плохо обученной модели ($e^a \not\approx e^b$) поправочный терм $\mathcal{L}_{\text{NCE}}(\hat{c}, \hat{c})$ ненулевой.

**[Связь с M.3.1]** CLIP loss $= \frac{1}{2}[\mathcal{L}_{\text{NCE}}(e^a, e^b) + \mathcal{L}_{\text{NCE}}(e^b, e^a)]$. Наш $\mathbb{E}[\mathcal{L}_{E2}]$ при $M = 2$ — аналог с якорем в центроиде.

---

### T.3 — Гарантия retrieval с margin [Paper B]

**[Вопрос]** При каких условиях centroid retrieval (поиск ближайшего центроида) даёт правильный результат? Какой margin нужен между объектами?

**[Почему важно]** Даёт **практический** критерий: если $\delta > 4M\varepsilon$, retrieval гарантирован. Можно мониторить $\delta$ (inter-centroid distance) и $\varepsilon$ (intra-object dispersion) во время обучения и фиксировать момент, когда условие выполняется. Это прямая связь теории с deployment.

**[Теорема T.3]**

Пусть для объектов $o_i, o_j$ ($i \neq j$) выполнено:
- $D_{\text{intra}}(o_i) \leq \varepsilon$ (intra-object dispersion bounded)
- $D_{\text{inter}}(i, j) \geq \delta$ (inter-centroid distance bounded below)

Тогда для любого запроса $q = e_i^m$ (эмбеддинг одной модальности объекта $o_i$) retrieval по ближайшему центроиду даёт корректный результат (ближайший к $q$ центроид — это $c_i$) при условии:

$$\delta > 4M\varepsilon$$

**[Доказательство]**

Шаг 1. Оценка $\|q - c_i\|$ (расстояние от запроса до своего центроида).

По определению $D_{\text{intra}}(o_i) = \frac{1}{M}\sum_m \|e_i^m - c_i\|^2 \leq \varepsilon$. Следовательно для конкретной модальности $m$:

$$\|e_i^m - c_i\|^2 \leq M \cdot D_{\text{intra}}(o_i) \leq M\varepsilon$$

$$\|q - c_i\| = \|e_i^m - c_i\| \leq \sqrt{M\varepsilon}$$

Шаг 2. Оценка $\|q - c_j\|$ снизу (расстояние от запроса до чужого центроида).

По triangle inequality:

$$\|q - c_j\| = \|e_i^m - c_j\| \geq \|c_i - c_j\| - \|e_i^m - c_i\| \geq \sqrt{\delta} - \sqrt{M\varepsilon}$$

Шаг 3. Условие корректного retrieval: $\|q - c_i\| < \|q - c_j\|$ для всех $j \neq i$:

$$\sqrt{M\varepsilon} < \sqrt{\delta} - \sqrt{M\varepsilon}$$

$$2\sqrt{M\varepsilon} < \sqrt{\delta}$$

$$4M\varepsilon < \delta \qquad \square$$

**[Assumption A.10]** (Bounded intra-object dispersion на evaluation set) На этапе inference существует $\varepsilon > 0$ такое, что $D_{\text{intra}}(o_i) \leq \varepsilon$ для всех объектов evaluation set. Это допущение верифицируется эмпирически: мы мониторим $D_{\text{intra}}$ как метрику.

**[Интуиция]** Условие $\delta > 4M\varepsilon$ означает: расстояние между центроидами должно быть в $4M$ раз больше средней дисперсии внутри объекта. Для $M = 5$ это $\delta > 20\varepsilon$ — достаточно мягкое условие для хорошо обученной модели.

**[Связь с M.3.3]** $\mathcal{L}_{\text{align}}$ минимизирует $D_{\text{intra}}$, а $\mathcal{L}_{\text{contrast}}$ максимизирует $D_{\text{inter}}$ — оба работают на выполнение условия T.3.

**[T.3* — Tightness: граница $4M\varepsilon$ достижима]**

Граница $\delta > 4M\varepsilon$ является **tight** (неулучшаемой): существует конфигурация, в которой $\delta = 4M\varepsilon$ и retrieval ломается.

**Конструкция.** Рассмотрим $d \geq 2$, $M$ модальностей, два объекта $o_1, o_2$.

Пусть $c_1 = \mathbf{0}$, $c_2 = \sqrt{\delta} \cdot \mathbf{e}_1$ (где $\mathbf{e}_1$ — единичный вектор по первой оси).

Для $o_1$: расположим $e^m_1$ так, что $\|e^m_1 - c_1\|^2 = M\varepsilon$ для одной "плохой" модальности $m^*$, и $\|e^m_1 - c_1\| = 0$ для остальных. Тогда $D_{\text{intra}}(o_1) = \frac{1}{M} \cdot M\varepsilon = \varepsilon$. Выберем $e^{m^*}_1 = \sqrt{M\varepsilon} \cdot \mathbf{e}_1$ (направлен к $c_2$).

Запрос $q = e^{m^*}_1$. Расстояния:

$$\|q - c_1\| = \sqrt{M\varepsilon}$$

$$\|q - c_2\| = |\sqrt{\delta} - \sqrt{M\varepsilon}|$$

Retrieval ломается (неоднозначность) когда $\|q - c_1\| = \|q - c_2\|$:

$$\sqrt{M\varepsilon} = \sqrt{\delta} - \sqrt{M\varepsilon} \implies \sqrt{\delta} = 2\sqrt{M\varepsilon} \implies \delta = 4M\varepsilon$$

При $\delta = 4M\varepsilon$ запрос $q$ равноудалён от $c_1$ и $c_2$ — retrieval неоднозначен.

При $\delta < 4M\varepsilon$: $\|q - c_2\| < \|q - c_1\|$ — retrieval **ошибочен** (ближайший центроид — $c_2$, а не $c_1$). $\square$

**[Интуиция]** Worst case возникает когда самый "далёкий" эмбеддинг одного объекта направлен ровно к центроиду другого объекта. Граница $4M$ (а не $4$) учитывает, что для $M$ модальностей одна может иметь dispersion до $M\varepsilon$ (budget всего $\varepsilon$ в среднем перераспределён в одну модальность).

---

### T.4 — BIBO-устойчивость T-S контроллера [Paper C]

**[Вопрос]** Является ли T-S fuzzy controller устойчивым? Не может ли он выдать неограниченные корректировки, которые "сломают" обучение?

**[Почему важно]** BIBO-устойчивость гарантирует что $\boldsymbol{\lambda}_t$ всегда остаётся в допустимой области $\Lambda$, а выход контроллера $\mathbf{u}_t$ ограничен. Без этого controller мог бы выдавать экстремальные значения (например $\tau \to 0$ или $w_m \to \infty$), дестабилизируя обучение. T.4 — необходимая предпосылка для T.5 (Ляпунов).

**[Теорема T.4]**

Пусть T-S fuzzy controller (M.6) удовлетворяет:
- (i) Partition of unity: $\sum_r \bar{h}_r(\mathbf{s}_t) = 1$ для всех $\mathbf{s}_t$
- (ii) Bounded state: $\mathbf{s}_t$ нормализован через running statistics (A.6)
- (iii) Проекция $\Pi_\Lambda$ на компактное множество $\Lambda = \prod_j [\ell_j, u_j]$

Тогда:
- **(a)** $\boldsymbol{\lambda}_t \in \Lambda$ для всех $t \geq 0$ (bounded trajectories)
- **(b)** $\|\mathbf{u}_t\| \leq U_{\max}$ для всех $t$ и всех $\mathbf{s}_t$ (bounded control output)
- **(c)** $\|\Delta\boldsymbol{\lambda}_t\| \leq \min(U_{\max},\; \text{diam}(\Lambda))$ для всех $t$

**[Доказательство]**

**(a)** Тривиально по конструкции: $\boldsymbol{\lambda}_{t+1} = \Pi_\Lambda(\boldsymbol{\lambda}_t + \mathbf{u}_t)$. Проекция $\Pi_\Lambda$ на компактный box $\Lambda$ всегда возвращает точку внутри $\Lambda$. Индукцией по $t$: если $\boldsymbol{\lambda}_0 \in \Lambda$, то $\boldsymbol{\lambda}_t \in \Lambda$ для всех $t$. $\checkmark$

**(b)** Выход контроллера:

$$\mathbf{u}_t = \sum_r \bar{h}_r(\mathbf{s}_t) \cdot (A_r \mathbf{s}_t + \mathbf{b}_r)$$

По triangle inequality и partition of unity ($\sum_r \bar{h}_r = 1$, $\bar{h}_r \geq 0$):

$$\|\mathbf{u}_t\| \leq \sum_r \bar{h}_r \cdot \|A_r \mathbf{s}_t + \mathbf{b}_r\| \leq \max_r \|A_r \mathbf{s}_t + \mathbf{b}_r\|$$

По (ii), $\|\mathbf{s}_t\| \leq S_{\max}$ (bounded). Матрицы $A_r$ и смещения $\mathbf{b}_r$ — фиксированные параметры контроллера. Следовательно:

$$U_{\max} = \max_r (\|A_r\| \cdot S_{\max} + \|\mathbf{b}_r\|) < \infty \qquad \checkmark$$

**(c)** Из (a) и non-expansiveness проекции:

$$\|\Delta\boldsymbol{\lambda}_t\| = \|\boldsymbol{\lambda}_{t+1} - \boldsymbol{\lambda}_t\| = \|\Pi_\Lambda(\boldsymbol{\lambda}_t + \mathbf{u}_t) - \boldsymbol{\lambda}_t\| \leq \|\mathbf{u}_t\| \leq U_{\max}$$

и тривиально $\|\Delta\boldsymbol{\lambda}_t\| \leq \text{diam}(\Lambda)$ (оба конца в $\Lambda$). $\square$

**[Важно]** Теорема T.4 **заменяет Assumption A.9**: bounded controller output теперь не постулируется, а **выводится** из конструкции контроллера. A.9 помечается как "Следствие T.4(b)".

**[Честное ограничение]** T.4 доказывает **BIBO-устойчивость** (bounded-input bounded-output): если входы ограничены, выходы ограничены. Это **НЕ** асимптотическая устойчивость и **НЕ** сходимость $\boldsymbol{\lambda}_t \to \boldsymbol{\lambda}^*$. Мы не утверждаем, что контроллер приводит к оптимальным гиперпараметрам — только что траектория остаётся в допустимой области.

---

### T.5 — Условное убывание Ляпунова [Paper C]

**[Вопрос]** При каких условиях функция Ляпунова $V_t$ убывает?

**[Почему важно]** Даёт **проверяемые** условия (C1-C3) для мониторинга: если C1-C3 выполнены — система стабильна. Если нет — сигнал к интервенции (остановка обучения, перезапуск с другими гиперпараметрами). T.5 превращает Ляпунова из абстрактной конструкции (M.7) в практический инструмент диагностики.

**[Теорема T.5]**

Пусть выполнены три верифицируемых условия:

**Условие C1** (EMA loss descent): существуют $\alpha_1 > 0$, $\sigma_1^2 > 0$ такие, что:

$$\mathbb{E}[\tilde{L}_{t+1} - \tilde{L}_t \mid \mathbf{s}_t] \leq -\alpha_1 \cdot \tilde{L}_t + \sigma_1^2$$

(стандартный SGD descent с шумом — $\tilde{L}_t$ убывает в среднем, кроме шумового потолка)

**Условие C2** (Fuzzy controller уменьшает $\text{Var}_m$): при активации правила R1 (Modality Imbalance, M.6.3):

$$\mathbb{E}[\text{Var}_m(w_{m,t+1}) - \text{Var}_m(w_{m,t}) \mid \bar{h}_{R1} > \theta_{\text{act}}] \leq -\alpha_2 \cdot \text{Var}_m(w_{m,t})$$

для некоторого $\alpha_2 > 0$ (контроллер перераспределяет веса в сторону баланса)

**Условие C3** ($\Delta\boldsymbol{\lambda}_t$ контрактивно): существуют $\rho \in (0, 1)$, $\sigma_3 > 0$ такие, что:

$$\|\Delta\boldsymbol{\lambda}_{t+1}\| \leq \rho \cdot \|\Delta\boldsymbol{\lambda}_t\| + \sigma_3$$

(изменения гиперпараметров затухают со временем, кроме шума)

Тогда для функции Ляпунова $V_t = \alpha \tilde{L}_t + \beta \|\Delta\boldsymbol{\lambda}_t\|^2 + \gamma \text{Var}_m(w_{m,t})$ (M.7.1):

$$\mathbb{E}[V_{t+1} - V_t \mid \mathbf{s}_t] \leq -\eta \cdot \Psi_t + \xi$$

где:
- $\eta = \min(\alpha \alpha_1,\; \beta(1 - \rho^2),\; \gamma \alpha_2)$
- $\Psi_t = \tilde{L}_t + \|\Delta\boldsymbol{\lambda}_t\|^2 + \text{Var}_m(w_{m,t})$ (мера "нездоровья" системы)
- $\xi = \alpha \sigma_1^2 + \beta \sigma_3^2 + \text{(higher order terms)}$ (шумовой потолок)

**[Доказательство]**

Разложим $V_{t+1} - V_t$ по трём компонентам:

$$V_{t+1} - V_t = \alpha(\tilde{L}_{t+1} - \tilde{L}_t) + \beta(\|\Delta\boldsymbol{\lambda}_{t+1}\|^2 - \|\Delta\boldsymbol{\lambda}_t\|^2) + \gamma(\text{Var}_{m,t+1} - \text{Var}_{m,t})$$

Берём условное ожидание и подставляем C1-C3:

**Компонента 1:** По C1:
$$\mathbb{E}[\alpha(\tilde{L}_{t+1} - \tilde{L}_t)] \leq -\alpha \alpha_1 \tilde{L}_t + \alpha \sigma_1^2$$

**Компонента 2:** По C3: $\|\Delta\boldsymbol{\lambda}_{t+1}\| \leq \rho\|\Delta\boldsymbol{\lambda}_t\| + \sigma_3$, значит $\|\Delta\boldsymbol{\lambda}_{t+1}\|^2 \leq \rho^2\|\Delta\boldsymbol{\lambda}_t\|^2 + 2\rho\sigma_3\|\Delta\boldsymbol{\lambda}_t\| + \sigma_3^2$. Следовательно:

$$\beta(\|\Delta\boldsymbol{\lambda}_{t+1}\|^2 - \|\Delta\boldsymbol{\lambda}_t\|^2) \leq -\beta(1 - \rho^2)\|\Delta\boldsymbol{\lambda}_t\|^2 + \beta(2\rho\sigma_3\|\Delta\boldsymbol{\lambda}_t\| + \sigma_3^2)$$

При $\|\Delta\boldsymbol{\lambda}_t\| \leq U_{\max}$ (T.4(b)), higher order term ограничен константой.

**Компонента 3:** По C2 (когда R1 активно):
$$\gamma(\text{Var}_{m,t+1} - \text{Var}_{m,t}) \leq -\gamma\alpha_2 \text{Var}_{m,t}$$

Когда R1 не активно, $\text{Var}_m$ мала (иначе R1 активировалось бы), и вклад ограничен.

**Сборка:** Суммируя три компоненты, получаем верхнюю оценку:

$$\mathbb{E}[V_{t+1} - V_t] \leq -\min(\alpha\alpha_1, \beta(1-\rho^2), \gamma\alpha_2) \cdot (\tilde{L}_t + \|\Delta\boldsymbol{\lambda}_t\|^2 + \text{Var}_{m,t}) + \text{(noise terms)}$$

$$= -\eta \cdot \Psi_t + \xi \qquad \square$$

**[Честное ограничение]** Условия C1-C3 **не всегда выполнены:**
- C1 требует, чтобы SGD делал прогресс — нарушается на плато, при плохом learning rate, при catastrophic forgetting.
- C2 требует, чтобы fuzzy controller корректно перераспределял веса — зависит от качества правил и лингвистических переменных.
- C3 требует затухания $\Delta\boldsymbol{\lambda}$ — нарушается при осциллирующем контроллере.

В экспериментах E6/E7 мы **эмпирически мониторим** выполнение C1-C3 как **диагностику**: строим графики $\tilde{L}_t$, $\|\Delta\boldsymbol{\lambda}_t\|$, $\text{Var}_m$, отмечаем интервалы нарушений. T.5 говорит: "если C1-C3 выполнены, система стабильна" — это верифицируемый, но условный результат.

**[Связь с M.7.2]** T.5 формализует целевое условие M.7.2: $\mathbb{E}[V_{t+1} - V_t] \leq -\eta\Psi_t + \xi$, показывая, что оно следует из трёх проверяемых условий, а не постулируется.

---

### T.6 — SGD для модальных весов: квадратичный рост дисперсии [Paper C]

**[Вопрос]** Почему fuzzy controller для модальных весов $w_m$ предпочтительнее прямой оптимизации $w$ через SGD (как в E5)?

**[Почему важно]** Это ключевой аргумент Paper C: почему нужен символьный контроллер, а не просто backprop для весов. T.6 формализует структурный дефект SGD-оптимизации весов и показывает, что fuzzy controller устраняет его by design.

**[Proposition T.6]**

Пусть $\mathcal{L}_{\text{total}} = \sum_{m=1}^M w_m \cdot \mathcal{L}^m_t$, где $\mathcal{L}^m_t > 0$ — per-modality loss. Рассмотрим SGD-обновление весов: $w_{m,t+1} = w_{m,t} - \eta \cdot \partial\mathcal{L}_{\text{total}} / \partial w_m$. Тогда:

**(a)** $\partial\mathcal{L}_{\text{total}} / \partial w_m = \mathcal{L}^m_t$ для всех $m, t$.

**(b)** $\text{Var}_m(w_{m,t+1}) = \text{Var}_m(w_{m,t}) + \eta^2 \cdot \text{Var}_m(\mathcal{L}^m_t) - 2\eta \cdot \text{Cov}_m(w_{m,t}, \mathcal{L}^m_t)$.

**(b')** **[Лемма — антикорреляция после $k$ шагов SGD]** Пусть $\mathcal{L}^m_t$ стационарны во времени (фиксированные параметры $\theta$, two-timescale A.5). Пусть $w_{m,0} = w_0$ для всех $m$ (одинаковая инициализация). Тогда после $k$ шагов SGD:

$$\text{Cov}_m(w_{m,k}, \mathcal{L}^m) = -\eta \cdot k \cdot \text{Var}_m(\mathcal{L}^m)$$

*Доказательство леммы.* При одинаковой инициализации: $w_{m,k} = w_0 - \eta \cdot \sum_{s=0}^{k-1} \mathcal{L}^m_s$. При стационарности $\mathcal{L}^m_s = \mathcal{L}^m$ (не зависит от $s$): $w_{m,k} = w_0 - \eta k \cdot \mathcal{L}^m$. Отклонение: $w_{m,k} - \bar{w}_k = -\eta k (\mathcal{L}^m - \bar{\mathcal{L}})$. Ковариация:

$$\text{Cov}_m(w_{m,k}, \mathcal{L}^m) = \frac{1}{M}\sum_m (w_{m,k} - \bar{w}_k)(\mathcal{L}^m - \bar{\mathcal{L}}) = -\eta k \cdot \frac{1}{M}\sum_m (\mathcal{L}^m - \bar{\mathcal{L}})^2 = -\eta k \cdot \text{Var}_m(\mathcal{L}^m) \leq 0 \qquad \square_{\text{lemma}}$$

Подставляя в (b): после $k$ шагов SGD с одинаковой инициализацией:

$$\text{Var}_m(w_{k+1}) = \text{Var}_m(w_k) + \eta^2 \cdot \text{Var}_m(\mathcal{L}^m) + 2\eta^2 k \cdot \text{Var}_m(\mathcal{L}^m)$$

$$= \text{Var}_m(w_k) + \eta^2(2k + 1) \cdot \text{Var}_m(\mathcal{L}^m)$$

Дисперсия весов растёт **квадратично** по $k$ (ускоряющийся рост).

**(c)** T-S controller с правилом R1: $u_m > 0$ при $\mathcal{L}^m_t$ is HI $\Rightarrow$ $\text{Cov}_m(u_m, \mathcal{L}^m) > 0$ — коррекция в правильную сторону, снижает $\text{Var}_m(w)$.

**(d)** Горизонт планирования: SGD = 1 шаг (только $\partial\mathcal{L}/\partial w$ в момент $t$), Fuzzy через EMA ($\beta = 0.99$) $\approx 100$ шагов.

**[Доказательство]**

**Шаг 1 (a).** $\mathcal{L}_{\text{total}} = \sum_m w_m \cdot \mathcal{L}^m_t$. При фиксированных параметрах сети $\theta$ (two-timescale assumption, A.5*), $\mathcal{L}^m_t$ не зависит от $w_m$:

$$\frac{\partial \mathcal{L}_{\text{total}}}{\partial w_m} = \mathcal{L}^m_t \qquad \checkmark$$

**Шаг 2 (b).** Обновление SGD: $w_{m,t+1} = w_{m,t} - \eta \cdot \mathcal{L}^m_t$.

Среднее по модальностям: $\bar{w}_{t+1} = \bar{w}_t - \eta \cdot \bar{\mathcal{L}}_t$.

Отклонение от среднего:

$$(w_{m,t+1} - \bar{w}_{t+1}) = (w_{m,t} - \bar{w}_t) - \eta \cdot (\mathcal{L}^m_t - \bar{\mathcal{L}}_t)$$

Возводим в квадрат и усредняем по $m$:

$$\text{Var}_m(w_{t+1}) = \frac{1}{M}\sum_m \bigl[(w_{m,t} - \bar{w}_t) - \eta(\mathcal{L}^m_t - \bar{\mathcal{L}}_t)\bigr]^2$$

$$= \text{Var}_m(w_t) - 2\eta \cdot \text{Cov}_m(w_t, \mathcal{L}^m_t) + \eta^2 \cdot \text{Var}_m(\mathcal{L}^m_t) \qquad \checkmark$$

**Шаг 3 (антикорреляция — по Лемме b').** После $k$ шагов SGD с одинаковой инициализацией и при стационарности $\mathcal{L}^m$, Лемма (b') даёт $\text{Cov}_m(w_{m,k}, \mathcal{L}^m) = -\eta k \cdot \text{Var}_m(\mathcal{L}^m) \leq 0$. Подставляя в Шаг 2:

$$\text{Var}_m(w_{k+1}) = \text{Var}_m(w_k) + \eta^2(2k+1) \cdot \text{Var}_m(\mathcal{L}^m) > \text{Var}_m(w_k) \qquad \checkmark$$

Рост **ускоряется** с номером шага ($2k+1$ — линейный коэффициент при $\eta^2 \text{Var}$). $\checkmark$

**Шаг 4 (c).** Правило R1 (Modality Imbalance, M.6.3): при $\text{Var}_m$ is HI, контроллер увеличивает $w_m$ для модальностей с высоким $\mathcal{L}^m_t$ и уменьшает для низких. По конструкции антецедента $A_{R1}$:

$$u_m \propto +(\mathcal{L}^m_t - \bar{\mathcal{L}}_t) \quad \text{когда Var is HI}$$

Следовательно $\text{Cov}_m(u_m, \mathcal{L}^m) > 0$ (положительная корреляция — коррекция **в правильную** сторону).

Обновление T-S: $w_{m,t+1}^{\text{TS}} = \Pi_\Lambda(w_{m,t} + u_m)$. При подходящем масштабе $u_m$:

$$\text{Var}_m(w_{t+1}^{\text{TS}}) \leq \text{Var}_m(w_t) \qquad \checkmark$$

**Шаг 5 (d).** Горизонт планирования.

SGD использует только мгновенный градиент $\partial\mathcal{L}/\partial w_m = \mathcal{L}^m_t$ в момент $t$ $\Rightarrow$ горизонт = 1 шаг.

Fuzzy controller использует EMA-сглаженные сигналы:

$$\text{EMA}_{m,t} = \sum_{k=0}^{\infty} \beta^k (1-\beta) \cdot \Delta\mathcal{L}_{m,t-k}$$

Эффективный горизонт: $\mathbb{E}[\text{horizon}] = \sum_{k=0}^{\infty} k \cdot \beta^k(1-\beta) = \frac{\beta}{1-\beta}$.

При $\beta = 0.99$: горизонт $\approx 100$ шагов. $\square$

**[Честное ограничение]** T.6(a)-(b) — формальный алгебраический результат. Лемма (b') доказана при двух условиях: (1) одинаковая инициализация $w_{m,0}$, (2) стационарность $\mathcal{L}^m$ (что требует two-timescale separation — $\theta$ медленнее $w$). На практике $\mathcal{L}^m_t$ нестационарны, и антикорреляция количественно может отличаться, но знак $\text{Cov} < 0$ устойчив (SGD **структурно** уменьшает $w_m$ у модальностей с высоким лоссом). T.6(c) — design choice: мы показываем что fuzzy controller **может** снижать $\text{Var}_m$, не что он это делает при любых условиях. Окончательный ответ: эксперимент E5 vs E6.

**[Assumption required]** A.1, A.5.

---

### T.7 — Gradient signal coverage: Centroid vs Pairwise InfoNCE [Paper B]

**[Вопрос]** В E1 (pairwise) каждая пара модальностей вносит свой gradient signal.
В E2 (centroid) якорь — центроид $c_i = \frac{1}{M}\sum_m e^m_i$.
Получает ли каждая модальность gradient signal на **каждом** шаге? Насколько
эффективнее centroid по signal coverage?

**[Почему важно]** Если centroid InfoNCE даёт **полный gradient coverage** (все $M$
модальностей обновляются на каждом шаге), это формальный аргумент в пользу E2 vs E1:
не только compute-эффективнее ($O(NM)$ vs $O(NM^2)$), но и **статистически
эффективнее** — каждый example из батча информирует все модальности одновременно.

**[Theorem T.7 — Gradient signal coverage]**

Пусть $\mathcal{L}_{\text{E2}} = -\frac{1}{N}\sum_{i=1}^N \log \frac{\exp(\langle c_i, c_i^+ \rangle / \tau)}{\sum_{j=1}^N \exp(\langle c_i, c_j \rangle / \tau)}$ — centroid InfoNCE (M.3.2).

Пусть $\mathcal{L}_{\text{E1}} = \frac{1}{\binom{M}{2}} \sum_{m < m'} \mathcal{L}^{(m,m')}_{\text{pair}}$ — pairwise InfoNCE (M.3.1).

**(a) Full coverage.** Для E2 каждая модальность $m$ получает ненулевой градиент на каждом шаге:

$$\frac{\partial \mathcal{L}_{\text{E2}}}{\partial e^m_i} = \frac{1}{M} \cdot \frac{\partial \mathcal{L}_{\text{E2}}}{\partial c_i} \neq \mathbf{0} \quad \forall m \in \{1,\ldots,M\}, \; \forall i$$

**(b) Partial coverage для E1.** Для E1 модальность $m$ получает градиент только от $(M-1)$ пар из $\binom{M}{2}$:

$$\frac{\partial \mathcal{L}_{\text{E1}}}{\partial e^m_i} = \frac{1}{\binom{M}{2}} \sum_{m' \neq m} \frac{\partial \mathcal{L}^{(m,m')}_{\text{pair}}}{\partial e^m_i}$$

Доля активных пар для модальности $m$: $\frac{M-1}{\binom{M}{2}} = \frac{2}{M}$.

При $M = 5$: каждая модальность участвует в $\frac{2}{5} = 40\%$ пар.

**(c) Variance reduction.** Пусть $g^m_i = \frac{\partial \mathcal{L}}{\partial e^m_i}$ — стохастический градиент для модальности $m$, объекта $i$. Обозначим $\sigma^2_g = \text{Var}(\nabla_{e^m} \mathcal{L}^{\text{pair}})$ — variance одного pairwise градиента.

Для E2 (centroid): $g^m_i = \frac{1}{M} \cdot \nabla_{c_i} \mathcal{L}_{\text{E2}}$, где $\nabla_{c_i}$ агрегирует информацию от всех $M$ модальностей через $c_i$.

Для E1 (pairwise): $g^m_i = \frac{1}{\binom{M}{2}} \sum_{m' \neq m} \nabla_{e^m_i} \mathcal{L}^{(m,m')}$, среднее $(M-1)$ независимых градиентов.

Variance:

$$\text{Var}(g^m_{\text{E1}}) = \frac{1}{\binom{M}{2}^2} \cdot (M-1) \cdot \sigma^2_g = \frac{4\sigma^2_g}{M^2(M-1)}$$

$$\text{Var}(g^m_{\text{E2}}) = \frac{\sigma^2_c}{M^2}$$

где $\sigma^2_c = \text{Var}(\nabla_{c_i}\mathcal{L}_{\text{E2}})$.

При сопоставимых $\sigma^2_g \sim \sigma^2_c$, ratio:

$$\frac{\text{Var}(g^m_{\text{E2}})}{\text{Var}(g^m_{\text{E1}})} = \frac{M-1}{4} \cdot \frac{\sigma^2_c}{\sigma^2_g}$$

**[Интуиция]** Ratio $\frac{M-1}{4}$ растёт с $M$, что кажется парадоксальным: variance E2 **больше** чем E1? Нет — ключевое отличие: E2 даёт **один** coherent gradient (от одного loss) vs E1 усредняет $(M-1)$ **некоррелированных** пар. Усреднение снижает variance, но ценой потери coherence: E1 градиенты от разных пар могут быть **конфликтующими** (gradient conflict, M.5). E2 gradient по конструкции неконфликтен — все модальности стягиваются к одному центроиду.

**(d) Effective signal.** Определим effective signal-to-noise ratio (SNR):

$$\text{SNR}^m = \frac{\|\mathbb{E}[g^m]\|^2}{\text{Var}(g^m)}$$

Для E2: $\mathbb{E}[g^m_{\text{E2}}] = \frac{1}{M}\mathbb{E}[\nabla_{c_i}\mathcal{L}]$ — **consistent** direction к центроиду.

Для E1: $\mathbb{E}[g^m_{\text{E1}}] = \frac{1}{\binom{M}{2}}\sum_{m'} \mathbb{E}[\nabla_{e^m_i}\mathcal{L}^{(m,m')}]$ — среднее $(M-1)$ **разнонаправленных** ожидаемых градиентов. При gradient conflict: $\|\mathbb{E}[g^m_{\text{E1}}]\| < \frac{1}{M-1}\sum_{m'}\|\mathbb{E}[\nabla \mathcal{L}^{(m,m')}]\|$.

$\Rightarrow$ $\text{SNR}^m_{\text{E2}} > \text{SNR}^m_{\text{E1}}$ при наличии gradient conflict.

**Доказательство.**

**Шаг 1 (a).** Chain rule. $c_i = \frac{1}{M}\sum_{m=1}^M e^m_i$, значит $\frac{\partial c_i}{\partial e^m_i} = \frac{1}{M} \cdot I_d$ для всех $m$.
По chain rule: $\frac{\partial \mathcal{L}_{\text{E2}}}{\partial e^m_i} = \frac{\partial \mathcal{L}_{\text{E2}}}{\partial c_i} \cdot \frac{\partial c_i}{\partial e^m_i} = \frac{1}{M}\frac{\partial \mathcal{L}_{\text{E2}}}{\partial c_i}$.
$\nabla_{c_i}\mathcal{L}_{\text{E2}} \neq \mathbf{0}$ пока $c_i \neq c_i^+$ и $c_i$ не является глобальным минимумом (тривиальный случай). $\square$

**Шаг 2 (b).** В E1: $\mathcal{L}_{\text{E1}} = \frac{1}{\binom{M}{2}}\sum_{m<m'}\mathcal{L}^{(m,m')}$. Модальность $m$ появляется в $\mathcal{L}^{(m,m')}$ только для $m' \neq m$, т.е. ровно в $(M-1)$ из $\binom{M}{2}$ пар. Из $\binom{M}{2} = \frac{M(M-1)}{2}$ доля: $\frac{M-1}{M(M-1)/2} = \frac{2}{M}$. $\square$

**Шаг 3 (c).** Для E1 при **независимости** градиентов разных пар:
$g^m_{\text{E1}} = \frac{1}{\binom{M}{2}} \sum_{m'=1}^{M-1} \nabla_{m'}\mathcal{L}^{(m,m')}$
$\text{Var}(g^m_{\text{E1}}) = \frac{1}{\binom{M}{2}^2}\sum_{m'}\text{Var}(\nabla\mathcal{L}^{(m,m')}) = \frac{(M-1)\sigma^2_g}{\binom{M}{2}^2} = \frac{4\sigma^2_g}{M^2(M-1)}$. $\square$

**Шаг 4 (d).** Определим gradient conflict score для E1:
$\text{GC}^m = 1 - \frac{\|\sum_{m'}\mathbb{E}[\nabla\mathcal{L}^{(m,m')}]\|}{\sum_{m'}\|\mathbb{E}[\nabla\mathcal{L}^{(m,m')}]\|} \in [0, 1]$

$\text{GC}^m = 0$: все градиенты сонаправлены (no conflict). $\text{GC}^m = 1$: полная отмена.
$\|\mathbb{E}[g^m_{\text{E1}}]\| = \frac{1-\text{GC}^m}{\binom{M}{2}}\sum_{m'}\|\mathbb{E}[\nabla\mathcal{L}^{(m,m')}]\|$.

Для E2: $\text{GC}^m_{\text{E2}} = 0$ по конструкции — один gradient $\frac{1}{M}\nabla_{c_i}\mathcal{L}$.

SNR ratio: $\frac{\text{SNR}^m_{\text{E2}}}{\text{SNR}^m_{\text{E1}}} \propto \frac{1}{(1-\text{GC}^m)^2}$.
При $\text{GC}^m > 0$: $\text{SNR}_{\text{E2}} > \text{SNR}_{\text{E1}}$. $\square$

**[Честное ограничение]** (a)-(b) — точные результаты (chain rule + combinatorics). (c) предполагает независимость pairwise градиентов — на практике они коррелированы через shared encoder $f_m$, что **снижает** $\text{Var}(g^m_{\text{E1}})$ relative to our estimate. (d) — качественный аргумент: GC > 0 типичен при гетерогенных модальностях (lean vs img), но его **величина** — эмпирический вопрос. T.7 даёт framework для анализа, не hard bound.

**[Assumption required]** A.1, A.2.

---

### T.8 — Sample complexity для centroid retrieval [Paper B]

**[Вопрос]** Из T.3 известно: retrieval корректен если $\delta > 4M\varepsilon$. Но для скольких объектов $N$ это условие выполняется одновременно? Какой максимальный размер базы данных при заданной точности модели?

**[Почему важно]** Даёт **практическую формулу**: зная $\sigma^2$ (средняя intra-dispersion) и $d$ (размерность), можно вычислить максимальное $N$ для которого retrieval гарантирован. Также показывает что **больше модальностей $\Rightarrow$ больше допустимый $N$** — формальный аргумент в пользу $M = 5$ vs $M = 2$.

**[Theorem T.8 — Retrieval guarantee с вероятностью]**

Пусть модель обучена, $N$ объектов в retrieval базе. Обозначим:
- $\sigma^2 = \mathbb{E}[D_{\text{intra}}(o)]$ — средняя intra-object dispersion
- $D_{\text{intra}}(o_i) = \frac{1}{M}\sum_m \|e^m_i - c_i\|^2$ — сумма $M$ слагаемых

**[Assumption A.13]** (Sub-exponential deviations) Случайные величины $\|e^m_i - c_i\|^2$ являются sub-exponential с параметрами $(\nu^2, b)$: $\mathbb{E}[\exp(\lambda(\|e^m - c\|^2 - \mathbb{E}[\|e^m - c\|^2]))] \leq \exp(\nu^2\lambda^2/2)$ для $|\lambda| < 1/b$. Это выполняется для bounded embeddings (L2-нормализованные на $\mathbb{S}^{d-1}$, A.2): $\|e^m - c\|^2 \leq 4$ всегда, значит sub-exponential с $b = 4$.

**(a) Concentration per object.** $D_{\text{intra}}(o_i)$ — среднее $M$ sub-exponential r.v., по Bernstein:

$$\mathbb{P}\bigl(D_{\text{intra}}(o_i) > \varepsilon\bigr) \leq \exp\biggl(-\frac{M}{2} \cdot \min\Bigl(\frac{(\varepsilon - \sigma^2)^2}{\nu^2},\; \frac{\varepsilon - \sigma^2}{b}\Bigr)\biggr), \quad \varepsilon > \sigma^2$$

**(b) Union bound.** Для retrieval на **всех** $N$ объектах (T.3: $\delta > 4M\varepsilon$, считаем inter-centroid separation достаточной):

$$\mathbb{P}(\exists\; i:\; D_{\text{intra}}(o_i) > \varepsilon) \leq N \cdot \exp\biggl(-\frac{M}{2} \cdot \min\Bigl(\frac{(\varepsilon - \sigma^2)^2}{\nu^2},\; \frac{\varepsilon - \sigma^2}{b}\Bigr)\biggr)$$

Retrieval корректен на всех $N$ объектах с вероятностью $\geq 1 - \delta_{\text{fail}}$ если:

$$N \leq \delta_{\text{fail}} \cdot \exp\biggl(\frac{M}{2} \cdot \min\Bigl(\frac{(\varepsilon - \sigma^2)^2}{\nu^2},\; \frac{\varepsilon - \sigma^2}{b}\Bigr)\biggr)$$

**(c) Масштабирование по $M$.** Максимальное $N$ растёт **экспоненциально** с $M$:

$$N_{\max}(M) \propto \exp\bigl(C \cdot M\bigr)$$

для константы $C = \frac{1}{2}\min\bigl((\varepsilon-\sigma^2)^2/\nu^2,\; (\varepsilon-\sigma^2)/b\bigr) > 0$.

При $M = 5$ vs $M = 2$: $N_{\max}(5) / N_{\max}(2) = \exp(3C)$.

**[Доказательство]**

**Шаг 1 (a).** $D_{\text{intra}}(o_i) = \frac{1}{M}\sum_{m=1}^M X_m$ где $X_m = \|e^m_i - c_i\|^2$ — sub-exponential r.v. с $\mathbb{E}[X_m] = \sigma^2_m$, $\sum_m \sigma^2_m / M = \sigma^2$. По Bernstein inequality для sub-exponential (Vershynin, HDP, Thm. 2.8.1):

$$\mathbb{P}\Bigl(\frac{1}{M}\sum_m X_m - \sigma^2 > t\Bigr) \leq \exp\Bigl(-\frac{M}{2}\min\bigl(t^2/\nu^2, t/b\bigr)\Bigr)$$

Подставляя $t = \varepsilon - \sigma^2$: получаем (a). $\checkmark$

**Шаг 2 (b).** Union bound: $\mathbb{P}(\exists\;i: D_{\text{intra}}(o_i) > \varepsilon) \leq \sum_{i=1}^N \mathbb{P}(D_{\text{intra}}(o_i) > \varepsilon) \leq N \cdot \exp(...)$. $\checkmark$

**Шаг 3 (c).** Из (b): $N \cdot \exp(-CM) \leq \delta_{\text{fail}}$, значит $N \leq \delta_{\text{fail}} \cdot \exp(CM)$. $\square$

**[Числовой пример]** Пусть $\sigma^2 = 0.05$, $\varepsilon = 0.1$, $\nu^2 = 0.01$, $b = 4$, $\delta_{\text{fail}} = 0.01$.

$C = \frac{1}{2}\min(0.05^2/0.01, 0.05/4) = \frac{1}{2}\min(0.25, 0.0125) = 0.00625$.

$M = 5$: $N \leq 0.01 \cdot \exp(5 \cdot 0.00625) = 0.01 \cdot 1.032 \approx 0.01$. Слишком мало!

Это означает: при $\nu^2 = 0.01$ (высокая вариабельность embeddings) bound пессимистичен. Для полезного bound нужна лучше обученная модель ($\sigma^2 \ll \varepsilon$) или большая размерность $d$ (что снижает $\nu^2$).

$\sigma^2 = 0.001$, $\varepsilon = 0.01$, $\nu^2 = 0.001$: $C = \frac{1}{2} \cdot \min(0.081, 0.00225) = 0.001125$. $N \leq 0.01 \cdot \exp(0.005625) \approx 0.01$. Всё ещё мало.

**[Честное ограничение]** Bound (b) пессимистичен по двум причинам: (1) union bound не учитывает корреляции между объектами; (2) sub-exponential tail bound — worst case. На практике $D_{\text{intra}}$ концентрируется значительно лучше. Практическая ценность T.8 — не числовой bound, а **качественный результат (c)**: $N_{\max}$ растёт экспоненциально с $M$. Это формальный аргумент "больше модальностей → лучше retrieval на больших базах".

**[Assumption required]** A.1, A.2, A.10, A.13 (new).

---

### T.9 — Generalization bound: Centroid vs Pairwise [Paper B]

**[Вопрос]** Centroid InfoNCE (E2) использует $\langle c_i, c_j \rangle$ вместо индивидуальных $\langle e^m_i, e^{m'}_j \rangle$. Обобщается ли centroid representation лучше на новые данные?

**[Почему важно]** Если да — это формальный аргумент что centroid не только compute-эффективнее (M.8.2), но и **статистически эффективнее**: требует меньше обучающих примеров для той же generalization gap. Для ограниченного датасета SciLibModal_v2 ($\sim$10k объектов) это критично.

**[Theorem T.9 — Variance reduction и generalization gap]**

**[Assumption A.14]** (Approximate independence of encoders) Для Family A (M.1.1), энкодеры $f_m$ имеют отдельные параметры $\theta_m$. При фиксированном объекте $o_i$ и случайном $o_j$ (из батча), cross-modal similarities $\langle e^m_i, e^{m'}_j \rangle$ для разных пар $(m,m')$ приблизительно некоррелированы:

$$\text{Cov}\bigl(\langle e^m_i, e^{m'}_j \rangle,\; \langle e^l_i, e^{l'}_j \rangle\bigr) \approx 0 \quad \text{для } (m,m') \neq (l,l')$$

Это выполняется точно если encoders independent; приблизительно — при shared backbone с разными projection heads.

**(a) Variance reduction.** Пусть $\sigma^2_s = \text{Var}(\langle e^m_i, e^{m'}_j \rangle)$ — variance одного cross-modal similarity score (усреднённая по парам).

Centroid similarity:

$$\langle c_i, c_j \rangle = \frac{1}{M^2}\sum_{m=1}^M \sum_{m'=1}^M \langle e^m_i, e^{m'}_j \rangle$$

среднее $M^2$ слагаемых. При A.14 (approximate independence):

$$\text{Var}(\langle c_i, c_j \rangle) = \frac{1}{M^4} \sum_{m,m'} \text{Var}(\langle e^m_i, e^{m'}_j \rangle) = \frac{M^2 \cdot \sigma^2_s}{M^4} = \frac{\sigma^2_s}{M^2}$$

Pairwise similarity: $\text{Var}(\langle e^m_i, e^{m'}_j \rangle) = \sigma^2_s$.

**Variance reduction factor:** $\text{Var}(\langle c_i, c_j \rangle) = \sigma^2_s / M^2$. Для $M = 5$: 25-кратное снижение.

**(b) Generalization gap.** Используя variance-sensitive generalization bound (Bernstein-type, Boucheron et al., 2013, Thm. 12.5):

Для loss $\mathcal{L}$ с Lipschitz-constant $\ell$ по similarities и variance $V = \text{Var}(\mathcal{L})$ по объектам в батче:

$$\mathbb{E}[\mathcal{L}] \leq \hat{\mathcal{L}} + \sqrt{\frac{2V \cdot \ln(2/\delta)}{N}} + \frac{7\ell \cdot \ln(2/\delta)}{3(N-1)}$$

InfoNCE — Lipschitz по similarities с constant $\ell \approx 1/\tau$ (temperature). Variance loss определяется variance similarities:

$V_{\text{E2}} \propto \text{Var}(\langle c_i, c_j \rangle) / \tau^2 = \sigma^2_s / (M^2 \tau^2)$

$V_{\text{E1}} \propto \text{Var}(\langle e^m_i, e^{m'}_j \rangle) / \tau^2 = \sigma^2_s / \tau^2$

Variance-dependent term ratio:

$$\frac{\text{gap}_{\text{E2}}}{\text{gap}_{\text{E1}}} = \sqrt{\frac{V_{\text{E2}}}{V_{\text{E1}}}} = \frac{1}{M}$$

**(c) Sample complexity equivalence.** Для достижения одинаковой generalization gap $\varepsilon_{\text{gen}}$:

$$N_{\text{E2}} = \frac{V_{\text{E2}}}{\varepsilon_{\text{gen}}^2} \cdot 2\ln(2/\delta) = \frac{N_{\text{E1}}}{M^2}$$

**Centroid InfoNCE требует в $M^2$ раз меньше обучающих примеров** (в variance-dominated regime).

Для $M = 5$: $N_{\text{E2}} \approx N_{\text{E1}} / 25$.

**[Доказательство]**

**Шаг 1 (a).** $\langle c_i, c_j \rangle = \frac{1}{M^2}\sum_{m,m'}\langle e^m_i, e^{m'}_j \rangle$ — линейная комбинация $M^2$ r.v. с равными коэффициентами $1/M^2$.

$\text{Var}(\langle c_i, c_j \rangle) = \frac{1}{M^4}\sum_{m,m'}\sum_{l,l'}\text{Cov}(\langle e^m_i, e^{m'}_j \rangle, \langle e^l_i, e^{l'}_j \rangle)$

При A.14 (independence): cross-terms $= 0$, остаются только diagonal terms:

$= \frac{1}{M^4} \cdot M^2 \cdot \sigma^2_s = \sigma^2_s / M^2$. $\checkmark$

**Шаг 2 (b).** InfoNCE loss:

$\mathcal{L}(i) = -\langle c_i, c_i^+ \rangle / \tau + \log\sum_j \exp(\langle c_i, c_j \rangle / \tau)$

$\partial\mathcal{L}/\partial\langle c_i, c_j \rangle = (p_j - \mathbf{1}_{j=i^+}) / \tau$ где $p_j = \text{softmax}_j$, $|p_j| \leq 1$. Lipschitz constant $\ell = 1/\tau$. $\checkmark$

Variance loss через delta method: $V \approx \ell^2 \cdot \text{Var}(\text{similarity}) = \sigma^2_s / (M^2\tau^2)$ для E2. $\checkmark$

**Шаг 3 (c).** Variance-dependent term в bound: $\sqrt{2V\ln(2/\delta)/N} = \varepsilon_{\text{gen}}$.

$N = 2V\ln(2/\delta)/\varepsilon_{\text{gen}}^2$. Подставляя $V_{\text{E2}} = V_{\text{E1}}/M^2$: $N_{\text{E2}} = N_{\text{E1}}/M^2$. $\square$

**[Честное ограничение]** (1) A.14 (independence) — приближение. Для Family A с shared backbone SciRus-tiny модальности коррелированы. Correlation снижает variance reduction: вместо $1/M^2$ реальный фактор $\approx 1/(M^2(1-\rho) + M\rho)$ где $\rho$ — средняя корреляция пар. При $\rho = 0.5$, $M = 5$: фактор $\approx 1/15$ вместо $1/25$ (всё ещё существенно). (2) Результат (c) — для variance-dominated regime. При малом $N$ доминирует Rademacher term (complexity-dependent), для которого centroid не даёт преимущества. (3) Для Family B (один encoder) A.14 нарушается сильно: $\rho \to 1$, и advantage centroid почти исчезает.

**[Assumption required]** A.1, A.2, A.14 (new).

---

## Сводка Assumptions

| ID | Формулировка | Блок |
|---|---|---|
| A.1 | Все $M$ модальностей присутствуют для каждого объекта в батче | M.0.3 |
| A.2 | Все эмбеддинги L2-нормализованы перед вычислением сходства | M.0.4 |
| A.3 | Mean pooling для visual encoder (вместо CLS-token) | M.1.2 |
| A.4* | Токенизаторы обучаются end-to-end с FVT-инициализацией (M.2.3) | M.2.2, M.2.3 |
| A.5 | Gradient conflict: $M$ backward passes (или proxy через $\text{Var}_t$) | M.5 |
| A.6 | Компоненты $\mathbf{s}_t$ нормализуются через running statistics | M.6.2 |
| A.7 | Bounded gradient variance (для стохастического Lyapunov) | M.7.2 |
| A.8 | L-smoothness потерь по $\boldsymbol{\lambda}$ | M.7.2 |
| ~~A.9~~ | ~~Bounded controller output~~ → **Следствие T.4(b):** $\|\mathbf{u}_t\| \leq U_{\max}$ выводится из partition of unity, bounded state, компактности $\Lambda$ | M.7.2, M.9 |
| A.10 | Bounded $D_{\text{intra}}(o_i) \leq \varepsilon$ на evaluation set (для T.3) | M.9 (T.3) |
| A.11 | Парные визуальные + LaTeX данные для всех объектов (для $\mathcal{L}_{\text{visual\_align}}$) | M.1.2* |
| A.12 | Full fine-tuning, discriminative LR: `lr_embed_ratio` = 0.1 (backbone lr = lr / 10) | M.2.4 |
| A.13 | Sub-exponential deviations для $\|e^m - c\|^2$ (выполняется при L2-нормализации, $b=4$) | M.9 (T.8) |
| A.14 | Approximate independence encoders (Family A): cross-modal similarities некоррелированы для разных пар $(m,m')$ | M.9 (T.9) |

---

╔══════════════════════════════════════════════════════════════╗
║  META-REASONING LOOP (v2.6.2)                                ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  [MR.1] КАК МЫ ПРИШЛИ К ЭТОМУ ДОКУМЕНТУ                     ║
║                                                              ║
║  Исходные данные: SETUP_DRAFT.md, CLAUDE.md, v_1 ноутбук,    ║
║  LIT.md, LITOBZOR.md + три независимые рецензии:             ║
║  ML-практик, математик-теоретик, рецензент ICML.             ║
║                                                              ║
║  Изменения v2.6.1 → v2.6.2:                                 ║
║  1. T.8: Sample complexity для centroid retrieval (Paper B)   ║
║     N_max ∝ exp(C·M), Bernstein + union bound                ║
║  2. T.9: Generalization bound centroid vs pairwise (Paper B)  ║
║     Var(centroid sim) = σ²/M², N_E2 ≈ N_E1/M²                ║
║  3. A.13 (sub-exponential), A.14 (encoder independence)       ║
║  4. M.8.4: +T.8, +T.9 rows                                   ║
║  5. M.9 header: 7 → 9 результатов                             ║
║                                                              ║
║  Изменения v2.6 → v2.6.1:                                   ║
║  1. T.6: Теорема → Proposition + Лемма (b') антикорреляция    ║
║     Cov(w,L) = -ηk·Var(L^m), квадратичный рост дисперсии     ║
║  2. M.8.4: добавлена строка T.6, summary T.1-T.5→T.1-T.6    ║
║  3. M.3.4: добавлена [Мотивация E4] с контрпримером          ║
║     (lean collapse при L_E3=0)                                ║
║  4. T.7: NEW — Gradient signal coverage (Paper B)             ║
║     Full coverage E2 vs 2/M partial E1, SNR analysis          ║
║  5. M.6.3*: конкретные матрицы A_{R1}, A_{R2} с ~15 элементами║
║  6. T.3*: tightness proof — граница 4Mε достижима             ║
║  7. M.8.4: добавлена строка T.7                               ║
║  8. M.9 header: 6 теорем → 7 результатов                      ║
║                                                              ║
║  Изменения v2.5 → v2.6:                                     ║
║  1. λ_va → fuzzy controller в E6-E7, λ_t R^10 → R^11          ║
║  2. s_t R^18 (без изменений, R6 использует L_{img,t} как proxy)║
║  3. M.2: M.2.3 FVT initialization, M.2.4 unfreezing strategy  ║
║  4. M.3.1: expanded E1 motivation (CLIP→M>2, symmetry)         ║
║  5. M.3.3: anti-collapse motivation (VICReg/Barlow Twins link) ║
║  6. M.3.4*: full loss anatomy (hierarchy λ/w, 20 components)   ║
║  7. M.3.5: expanded E5 motivation (flat vs hierarchical)       ║
║  8. M.6.3: R6 (Visual Misalignment), 6→7 правил               ║
║  9. M.9: intro/outro T.1-T.5, T.6 (SGD vs Fuzzy), T.7 (grad)  ║
║  10. Family B: clarified (no fuzzy controller, pure ablation)   ║
║  11. Assumptions: A.4→A.4* (FVT), добавлен A.12 (unfreezing)   ║
║                                                              ║
║  Предыдущие изменения (v2.4 → v2.5):                         ║
║  1. M.3.4: определён L_reg^m (per-modality anti-collapse)     ║
║     S^m_{ij}, s̄_neg^m, L_reg^m = ReLU(s̄_neg^m)               ║
║  2. M.3.6: L_E6 = w_{g,t}·L_E3 + Σ w_{m,t}·L_personal^m     ║
║     Добавлено замечание E5 vs E6 (LossMixer vs fuzzy)          ║
║  3. M.5.1: s_t R^8 → R^18 (+per-modality L_m, EMA_m)         ║
║  4. M.6.1: таблица += L_{m,t}, EMA_{m,t}                      ║
║  5. M.6.3: R1,R3 адресно используют per-modality сигналы       ║
║                                                              ║
║  Предыдущие изменения (v2.3 → v2.4):                        ║
║  1. M.1.2 Шаг 1: non-overlapping → overlapping patches        ║
║     stride s = p/2 = 32, K = ⌊(w-p)/s⌋ + 1                   ║
║     Обоснование: сохранение граничных признаков               ║
║     Литература: PVTv2, T2T-ViT, CvT [NEED_VERIFY]            ║
║  2. A.3: mean pooling устраняет избыточность от overlap        ║
║                                                              ║
║  Предыдущие изменения (v2.2 → v2.3):                        ║
║  1. M.1.2: AlignNet = LayerNorm(Linear)                       ║
║     SciRus-tiny остаётся, L_visual_align добавлен             ║
║  2. M.1.2*: L_visual_align — контрастивная привязка           ║
║     pooled visual → pooled LaTeX (по Hadsell 2006).           ║
║  3. M.3.0: L_visual_align добавлен как static auxiliary       ║
║     ко всем E1-E7, λ_va не в fuzzy controller.               ║
║  4. M.9: 5 теорем T.1-T.5 с полными доказательствами:        ║
║     T.1: InfoNCE → MI lower bound (Paper B)                   ║
║     T.2: Редукция к CLIP при M=2 (Paper B)                    ║
║     T.3: Retrieval guarantee δ > 4Mε (Paper B)                ║
║     T.4: BIBO-устойчивость T-S controller (Paper C)           ║
║     T.5: Условное убывание Ляпунова при C1-C3 (Paper C)       ║
║  5. M.8.4: таблица расширена теоремами + ограничениями        ║
║  6. A.9 → следствие T.4(b) (не assumption)                    ║
║  7. Добавлены A.10 (bounded D_intra), A.11 (paired data)     ║
║                                                              ║
║  Предыдущие изменения (v2.0 → v2.2):                        ║
║  - Family B: детерминированный one-hot тег                    ║
║  - V_{t+1}→V_t: каузальный парадокс устранён                 ║
║  - L_lyap согласована (V_t-V_{t-1})                           ║
║  - LMI как концептуальная аналогия + two-timescale disclaimer ║
║                                                              ║
║  [MR.2] МОЖНО ЛИ УЛУЧШИТЬ                                   ║
║                                                              ║
║  a) A_{R1}, A_{R2} конкретизированы (M.6.3*). A_{R3}-A_{R6}   ║
║     остаются задачей TZ.md/реализации.                        ║
║  b) ρ в L_rad: задать эмпирически (hyperparameter search).   ║
║  c) T.1 ограничивает I(C;C'), не I(e^m;e^{m'}) — честно.    ║
║  d) T.5 условная: C1-C3 мониторятся, не гарантируются.       ║
║  e) δ_va (margin в L_visual_align): подбирать эмпирически.   ║
║  f) T.6: формализовано — Лемма (b') Cov = -ηk·Var(L^m).     ║
║  g) T.7(c) variance: independence assumption (approximate).   ║
║  h) T.3*: tight bound добавлен, граница неулучшаема.          ║
║  i) T.8: sample complexity — bound пессимистичен, но          ║
║     качественный результат (exp scaling with M) ценен.        ║
║  j) T.9: generalization — A.14 (independence) approximate     ║
║     для shared backbone. Для Family B не работает.            ║
║                                                              ║
║  [MR.3] ЕСТЬ ЛИ ПРОТИВОРЕЧИЯ                                ║
║                                                              ║
║  Уровень 1: Согласованность в файле:                          ║
║  - λ_t ∈ R^11 (v2.6: +λ_va в E6-E7), u_t ∈ R^11             ║
║  - s_t ∈ R^18 (без изменений в v2.6)                         ║
║  - A_r ∈ R^{11×18}, b_r ∈ R^11                               ║
║  - A.9 → T.4(b): непротиворечиво                              ║
║  - L_visual_align: static в E1-E5, dynamic в E6-E7 (R6)      ║
║  - T.6: SGD vs Fuzzy — согласовано с E5 vs E6 замечанием      ║
║  - 7 правил R0-R6 (v2.6: +R6 Visual Misalignment)             ║
║  Уровень 2: Согласованность с другими файлами:                ║
║  - TZ.md Sec 5: обновить (AlignNet, L_visual_align)           ║
║  - MOTIVATION.md Sec 3.3: обновить visual pipeline            ║
║  - LITOBZOR.md Gap G1: усилить                                ║
║  Уровень 3: Литература:                                       ║
║  - T.1 корректно ссылается на [INFONCE-2018, Prop. 1]         ║
║  - T.3: triangle inequality — стандартный результат            ║
║  - T.4: BIBO — стандартное определение                         ║
║  - Hadsell 2006 — в literature/ (верифицировано)              ║
║  - Bottou 2018 — [NEED_VERIFY] (не в literature/)             ║
║                                                              ║
║  [MR.4] ПРОВЕРКА НА ГАЛЛЮЦИНАЦИИ                            ║
║                                                              ║
║  G1: Все ссылки в теоремах:                                   ║
║    [INFONCE-2018, Prop.1] — верифицировано ✓                  ║
║    [Hadsell 2006, Eq. 2] — верифицировано ✓                   ║
║    Bottou 2018 — [NEED_VERIFY] (сохранён)                    ║
║  G2: Размерности в теоремах:                                  ║
║    T.1: C, C' ∈ S^{d-1}, I(C;C') — скаляр ✓                 ║
║    T.3: D_intra ∈ R (скаляр), δ, ε ∈ R+ ✓                    ║
║    T.4: s_t ∈ R^18, u_t ∈ R^11, λ_t ∈ R^11, Λ ⊂ R^11 ✓      ║
║    A_r ∈ R^{11×18}, b_r ∈ R^11 ✓                             ║
║    T.6: w_m ∈ R (скаляр per modality), Var_m ∈ R+ ✓          ║
║    T.7: g^m ∈ R^d (gradient per modality), GC ∈ [0,1] ✓     ║
║    T.3*: tight example: d≥2, δ=4Mε → retrieval ambiguous ✓  ║
║    T.8: N ∈ N, σ² ∈ R+, ν²,b ∈ R+, exp(CM) — скаляр ✓      ║
║    T.9: Var ∈ R+, ratio 1/M² — скаляр, N_E2/N_E1 ✓          ║
║    T.5: V_t ∈ R+ (скаляр), η,ξ ∈ R+ ✓                       ║
║    L_visual_align: Pool(ṽ) ∈ R^{d'_m}, Pool(Embed) ∈ R^{d'_m}║
║    Размерности совпадают (обе d'_m после AlignNet) ✓          ║
║  G3: Обобщения:                                               ║
║    T.3 "для всех объектов" — при условии A.10, корректно      ║
║    T.5 "система стабильна" — при C1-C3, условный результат    ║
║    Нет ложных "всегда" или "в общем случае" ✓                 ║
║  G4: Атрибуции:                                               ║
║    InfoNCE bound → van den Oord 2018 ✓                        ║
║    Contrastive loss → Hadsell 2006 ✓                          ║
║                                                              ║
║  [MR.5] ПАТТЕРНЫ                                             ║
║                                                              ║
║  [P.8] CAUSAL DIRECTION IN REGULARIZERS                       ║
║    (сохранён из v2.2)                                         ║
║                                                              ║
║  [P.9] ALIGNMENT LAYER FOR CROSS-DOMAIN TOKEN INJECTION        ║
║    Обнаружен: при анализе visual pipeline M.1.2               ║
║    Тип: паттерн (архитектурный)                               ║
║    Описание: подача визуальных токенов из CNN в текстовый      ║
║    трансформер требует нормализации распределения.              ║
║    Правило: если подаём токены из домена B в encoder            ║
║    обученный на домене A — использовать AlignNet               ║
║    (LayerNorm + Linear) + auxiliary contrastive loss            ║
║    для семантической привязки.                                  ║
║    Применяется к: визуальные токены в текстовый encoder.        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
