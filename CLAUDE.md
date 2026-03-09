# CLAUDE.md — SciLibMath_v2 Experiment Codebase

> **Версия:** 1.0 | **Проект:** SciLibMath_v2 | **Автор:** Andrey Khalov (MIPT)
> **Назначение:** Реализация абляционных экспериментов E1-E7 по формализации из writing/MATH.md v2.6.2

---

## 0. КОНТЕКСТ ПРОЕКТА

**Что это:** Код мультимодального контрастного обучения для математических объектов (5 модальностей: en, ru, lean, latex, img). Центроидная геометрия, ablation E1-E7, fuzzy T-S controller, Lyapunov стабилизация.

**Формализация:** `writing/MATH.md` v2.6.2 — ЗАФИКСИРОВАНА, не менять. Код ОБЯЗАН соответствовать постановке. Любое расхождение с MATH.md — баг.

**Три публикации:**
- Paper A: датасет + архитектура энкодеров (E1-E4, Family A vs B)
- Paper B: геометрия + centroid retrieval (E1-E5, теоремы T.1-T.3, T.7-T.9)
- Paper C: fuzzy controller + Lyapunov (E6-E7, теоремы T.4-T.6)

**Агент живёт на уровень выше:** `/home/opt/scilib/scilib.ai/laboratory/SciLibMath_v2/`
**Корень проекта (git repo):** `/home/opt/scilib/scilib.ai/laboratory/SciLibMath_v2/v_2/`

---

## 1. GIT SANITY RULES (ОБЯЗАТЕЛЬНО)

### 1.1 Запрещённые действия
- **НИКОГДА** не работать в detached HEAD. Проверять: `git branch --show-current` не пусто.
- **НИКОГДА** не делать `git reset --hard` без явной команды пользователя
- **НИКОГДА** не делать `git rebase` / `git cherry-pick` без явной команды
- **НИКОГДА** не делать `git push --force`
- **НИКОГДА** не ставить себя в co-authors коммитов
- **НИКОГДА** не делать `git add -A` / `git add .` — только конкретные файлы

### 1.2 Git Snapshot (обязательный протокол)
Перед КАЖДОЙ итерацией агент выводит:
```bash
git status -sb
git branch --show-current
git rev-parse --short HEAD
git log -1 --oneline --decorate
```
Если `git branch --show-current` пусто — СТОП, восстановить ветку.

### 1.3 Коммиты
- Осмысленные commit messages на английском
- Формат: `<type>: <description>` (feat, fix, refactor, test, docs, chore)
- НЕ коммитить без запроса пользователя
- НЕ коммитить секреты, данные, чекпоинты, логи

### 1.4 Ветки
- `main` — стабильная версия
- `dev` — текущая разработка
- `experiment/<name>` — экспериментальные ветки
- Мёрж только через `--no-ff`

---

## 2. СТРУКТУРА ПРОЕКТА

```
v_2/                              ← корень git repo
├── CLAUDE.md                     ← этот файл
├── README.md                     ← описание проекта
├── init.sh                       ← автоматическая настройка окружения (ОБНОВЛЯТЬ!)
├── requirements.txt              ← зависимости
├── .gitignore                    ← исключения
│
├── writing/                      ← ЗАФИКСИРОВАНО, НЕ МЕНЯТЬ
│   ├── MATH.md                   ← формализация v2.6.2 (источник истины для кода)
│   ├── TZ.md                     ← техзадание v1.6
│   ├── MOTIVATION.md             ← мотивация
│   ├── LIT.md                    ← литература
│   └── LITOBZOR.md               ← обзор литературы
│
├── configs/                      ← YAML конфиги экспериментов
│   ├── base.yaml                 ← общие параметры
│   ├── e1_pairwise.yaml          ← E1: Pairwise InfoNCE
│   ├── e2_centroid.yaml          ← E2: Centroid InfoNCE
│   ├── e3_centroid_reg.yaml      ← E3: + alignment + radial
│   ├── e4_composite_static.yaml  ← E4: full loss, static weights
│   ├── e5_composite_learnable.yaml ← E5: LossMixer
│   ├── e6_fuzzy.yaml             ← E6: fuzzy T-S controller
│   └── e7_lyapunov.yaml          ← E7: + Lyapunov constraint
│
├── code/                         ← ВЕСЬ КОД ЗДЕСЬ
│   ├── train.py                  ← ЕДИНАЯ ТОЧКА ЗАПУСКА
│   ├── evaluate.py               ← оценка модели (retrieval, metrics)
│   │
│   ├── data/                     ← загрузка и обработка данных
│   │   ├── __init__.py
│   │   ├── dataset.py            ← SciLibModal_v2 Dataset class
│   │   ├── dataloader.py         ← DataLoader, batching, sampling
│   │   ├── tokenizers.py         ← Lean/LaTeX токенизаторы
│   │   └── transforms.py         ← трансформации для визуальной модальности
│   │
│   ├── models/                   ← архитектура
│   │   ├── __init__.py
│   │   ├── encoders.py           ← 5 энкодеров (en, ru, lean, latex, img)
│   │   ├── projections.py        ← проекционные головы
│   │   ├── family_a.py           ← Family A (5 отдельных энкодеров)
│   │   ├── family_b.py           ← Family B (2 энкодера: symbolic + visual)
│   │   └── align_net.py          ← AlignNet для visual alignment
│   │
│   ├── losses/                   ← функции потерь (СТРОГО по MATH.md)
│   │   ├── __init__.py
│   │   ├── infonce.py            ← InfoNCE (pairwise E1, centroid E2)
│   │   ├── alignment.py          ← alignment + radial regularization (E3)
│   │   ├── composite.py          ← полный составной лосс (E4)
│   │   ├── loss_mixer.py         ← LossMixer MLP (E5)
│   │   ├── anti_collapse.py      ← anti-collapse регуляризация
│   │   └── visual_align.py       ← L_visual_align
│   │
│   ├── controller/               ← fuzzy T-S controller (E6-E7)
│   │   ├── __init__.py
│   │   ├── state_vector.py       ← s_t ∈ R^18 — вычисление состояния
│   │   ├── membership.py         ← функции принадлежности μ
│   │   ├── rules.py              ← правила R0-R6, матрицы A_r
│   │   ├── ts_controller.py      ← T-S контроллер (выход u_t ∈ R^11)
│   │   └── lyapunov.py           ← V_t регуляризация (E7)
│   │
│   ├── metrics/                  ← метрики
│   │   ├── __init__.py
│   │   ├── retrieval.py          ← R@1, R@3, R@10 (centroid + cross-modal)
│   │   ├── geometry.py           ← D_intra, D_inter, collapse score
│   │   ├── modality_balance.py   ← balance score, variance
│   │   └── scalability.py        ← деградация на 10k-50k объектах
│   │
│   ├── logging/                  ← логирование
│   │   ├── __init__.py
│   │   ├── tb_logger.py          ← TensorBoard writer
│   │   └── s3_backup.py          ← rsync backup в S3/MinIO
│   │
│   └── utils/                    ← утилиты
│       ├── __init__.py
│       ├── config.py             ← загрузка YAML конфигов
│       ├── seed.py               ← фиксация random seed
│       ├── checkpoint.py         ← save/load checkpoint
│       └── device.py             ← GPU setup, mixed precision
│
├── tests/                        ← тесты
│   ├── test_losses.py            ← unit tests для всех лоссов
│   ├── test_controller.py        ← unit tests для fuzzy controller
│   ├── test_metrics.py           ← unit tests для метрик
│   └── test_data.py              ← тесты загрузки данных
│
├── data/                         ← данные (в .gitignore)
│   └── scilibrumodal-v2/         ← клон датасетного репо
│
├── checkpoints/                  ← чекпоинты (в .gitignore)
├── runs/                         ← TensorBoard логи (в .gitignore)
└── logs/                         ← текстовые логи (в .gitignore)
```

---

## 3. CODING STANDARDS

### 3.1 Модульность
- **НЕ** писать монолитные классы. Один файл — одна ответственность.
- Каждый модуль ≤ 300 строк. Если больше — разбивать.
- Всё запускается через `code/train.py` с конфигом: `python code/train.py --config configs/e4_composite_static.yaml`

### 3.2 Код = формализация
- Каждая функция потерь ОБЯЗАНА иметь комментарий со ссылкой на MATH.md: `# MATH.md M.3.2, Eq. L_E2`
- Размерности в комментариях: `# shape: (N, M, d) -> (N, d)`
- Assertions для проверки размерностей в debug mode

### 3.3 Конфиги (YAML)
- Все гиперпараметры в YAML, НЕ в коде
- base.yaml наследуется всеми экспериментами
- Каждый эксперимент переопределяет только то, что отличается
- Пример структуры:
```yaml
experiment: e4_composite_static
seed: 42
device: cuda

data:
  dataset_path: data/scilibrumodal-v2
  batch_size: 64
  test_fraction: 0.05
  num_workers: 4

model:
  family: A
  embedding_dim: 256
  encoders: ...

loss:
  type: composite_static
  tau: 0.07
  lambda_align: 0.3
  lambda_rad: 0.1
  lambda_reg: 0.05
  lambda_va: 0.1
  weights: {en: 1.0, ru: 1.0, lean: 1.5, latex: 1.5, img: 1.0, global: 1.0}

training:
  epochs: 10
  lr: 1e-4
  optimizer: AdamW
  scheduler: cosine
  warmup_steps: 500
  mixed_precision: true
  gradient_clip: 1.0

eval:
  eval_every_steps: 100
  retrieval_k: [1, 3, 10]
  scalability_sizes: [10000, 25000, 50000]

logging:
  tensorboard_dir: runs/
  s3_backup: true
  s3_bucket: scilibmath-v2-logs
  backup_every_minutes: 30
  log_embeddings: true
  log_loss_components: true
  log_controller_state: true  # E6-E7 only

checkpoint:
  save_every_epochs: 1
  keep_best: 3
  metric: centroid_recall_at_1
```

### 3.4 Reproducibility
- `seed` фиксируется через `torch.manual_seed`, `np.random.seed`, `random.seed`
- `torch.backends.cudnn.deterministic = True` в debug mode
- Конфиг эксперимента сохраняется в TensorBoard и в чекпоинте
- Каждый чекпоинт содержит: model state, optimizer state, epoch, step, config, metrics

### 3.5 Mixed precision
- По умолчанию fp16 через `torch.cuda.amp`
- RTX 3090 поддерживает fp16 и tf32
- Для fuzzy controller (E6-E7): fp32 (чувствителен к точности)

---

## 4. DATASET MANAGEMENT

### 4.1 Датасет
- Репозиторий: https://github.com/andkhalov/scilibrumodal-v2
- Клонируется в `data/scilibrumodal-v2/` (в .gitignore)
- Загрузка данных через `init.sh`

### 4.2 Data split
- 95% train, 5% test (фиксированный random split по seed)
- Test set — НИКОГДА не виден модели при обучении
- Test set используется для: R@1, R@3, R@10 каждые N шагов
- Отдельно: scalability test на 10k-50k объектах

### 4.3 DataLoader
- Каждый batch содержит ПОЛНЫЕ мультимодальные объекты (все 5 модальностей)
- Если у объекта нет какой-то модальности — маскировать (mask tensor)
- Hard negatives: по возможности, негативы из той же математической области

---

## 5. TENSORBOARD И S3 BACKUP

### 5.1 Что логировать (ОБЯЗАТЕЛЬНО для всех экспериментов)
```
Scalars:
  - loss/total, loss/contrast, loss/align, loss/rad, loss/reg, loss/va
  - loss/per_modality/{en,ru,lean,latex,img}
  - metrics/centroid_R@1, R@3, R@10
  - metrics/crossmodal_R@1 (для каждой пары модальностей)
  - metrics/D_intra, D_inter, collapse_score, modality_balance
  - training/lr, training/grad_norm, training/epoch
  - training/batch_time_ms

Для E6-E7 дополнительно:
  - controller/s_t_components (все 18)
  - controller/u_t_components (все 11)
  - controller/rule_activations (h_r для R0-R6)
  - controller/lambda_t (все 11 компонентов)
  - lyapunov/V_t, lyapunov/delta_V (E7)

Histograms (каждые K шагов):
  - embeddings/per_modality (распределение норм)
  - embeddings/centroid_distances
  - weights/encoder_params (для мониторинга unfreezing)

Images (каждые K шагов):
  - similarity_matrix (N×N centroid similarities)
  - modality_tsne (2D проекция эмбеддингов, цветом по модальности)
```

### 5.2 S3 backup
- rsync в MinIO bucket `scilibmath-v2-logs` каждые 30 минут
- После завершения каждого эксперимента — полный sync
- Чекпоинты лучших моделей — тоже в S3
- Проверка доступа: `init.sh` верифицирует подключение к S3

---

## 6. ЭКСПЕРИМЕНТЫ: ДВА РЕЖИМА

### 6.1 Режим 1: Ablation sweep (быстрый)
- Все E1-E7 по 2-3 эпохи
- Цель: определить какой эксперимент показывает лучшие результаты
- Сравнение по R@1, collapse score, training stability
- Результат: выбор лучшего режима для полного прогона

### 6.2 Режим 2: Full training
- Лучший эксперимент(ы) на 10 эпох
- Полный набор метрик
- Scalability test на 10k-50k объектах
- Cross-modal retrieval analysis
- Результат: данные для Paper B (и Paper C если E6/E7 победят)

---

## 7. EVALUATION PROTOCOL

### 7.1 Retrieval metrics (на test set, 5%)
- **Centroid R@k**: query = centroid объекта без одной модальности, target = полный центроид
- **Cross-modal R@k**: query = одна модальность, target = другая модальность того же объекта
- **Partial retrieval**: query = подмножество модальностей, target = полный центроид
- k ∈ {1, 3, 10}

### 7.2 Geometric metrics
- **D_intra**: средний радиус объекта (MATH.md M.3.3)
- **D_inter**: среднее расстояние между центроидами
- **Collapse score**: s̄_neg из MATH.md M.3.3
- **Modality balance**: Var_m(w_m) — дисперсия модальных весов

### 7.3 Scalability test
- Подготовить тестовые наборы: 10k, 25k, 50k объектов
- Если датасет меньше — использовать augmentation или synthetic negatives
- Замерить: R@k vs N_objects, latency vs N_objects

---

## 8. init.sh — АВТОМАТИЧЕСКАЯ НАСТРОЙКА

**ПРАВИЛО:** `init.sh` ОБЯЗАТЕЛЬНО обновлять при любом изменении зависимостей, структуры данных, или конфигурации. Это единственная точка входа для нового разработчика.

init.sh должен:
1. Проверить Python >= 3.11
2. Создать venv если не существует
3. Установить зависимости (requirements.txt)
4. Клонировать/обновить датасет (scilibrumodal-v2)
5. Скачать данные (если нужна отдельная загрузка)
6. Проверить GPU (nvidia-smi)
7. Создать директории (checkpoints/, runs/, logs/)
8. Проверить S3 доступ (rsync test)
9. Запустить smoke test (import всех модулей, загрузка одного batch)
10. Вывести сводку: OK / FAIL

---

## 9. DIARY.md PROTOCOL

**DIARY.md** находится на уровне `SciLibMath_v2/DIARY.md` (не в v_2/).

### Правила:
- **Append-only**: новые записи В НАЧАЛО файла
- **Никогда** не редактировать/удалять старые записи
- Запись после КАЖДОЙ итерации (commit, эксперимент, значимое изменение)

### Формат записи:
```
---
## YYYY-MM-DD HH:MM (ветка)

**Цель:** что делали
**Изменения:** список файлов/модулей
**Результат:** что получилось (метрики, если есть)
**Проверка:** как проверить
**Откат:** как вернуть (1-3 команды)
```

---

## 10. ТЕСТЫ

### Минимальный набор:
- `tests/test_losses.py` — все 7 вариантов лосса, проверка размерностей, градиентов
- `tests/test_controller.py` — fuzzy controller: s_t вычисление, правила, u_t выход
- `tests/test_metrics.py` — retrieval, geometry, balance
- `tests/test_data.py` — загрузка данных, batching, маски

### Запуск:
```bash
python -m pytest tests/ -v
```

### Когда запускать:
- Перед каждым коммитом
- После изменения лосс-функций
- После изменения controller логики

---

## 11. ЧТО НЕ ДЕЛАТЬ

- **НЕ менять** файлы в `writing/` — формализация зафиксирована
- **НЕ хардкодить** гиперпараметры — всё в YAML конфигах
- **НЕ коммитить** данные, чекпоинты, логи, venv
- **НЕ писать** монолитные файлы > 300 строк
- **НЕ запускать** эксперимент без проверки что тесты проходят
- **НЕ игнорировать** расхождения между кодом и MATH.md
- **НЕ забывать** обновлять init.sh при изменении окружения
- **НЕ забывать** записывать в DIARY.md

---

## 12. HARDWARE

- **GPU:** NVIDIA RTX 3090 (24GB VRAM)
- **CPU:** достаточно для data loading
- **RAM:** проверить при загрузке полного датасета
- **Disk:** checkpoints + tensorboard могут занять ~10-50GB

### Batch size estimation:
- d=256, M=5, fp16: ~2-4GB для batch_size=64
- При OOM: уменьшить batch_size, включить gradient accumulation
- Gradient accumulation steps в конфиге

---

*CLAUDE.md v1.0 | SciLibMath_v2 Code | 2026-03-09*
