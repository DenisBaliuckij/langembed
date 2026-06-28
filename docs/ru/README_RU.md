# langembed — Русская документация

> **English documentation:** [README.md](../../README.md)

Пайплайн для обучения sentence embeddings с нуля для языков с ограниченными ресурсами — с циклом активного обучения от носителей языка. Три ветки моделей обучаются на одних данных и оцениваются одной метрикой, обеспечивая чистое A/B/C-сравнение архитектур.

---

## Содержание

1. [Что это такое](#что-это-такое)
2. [Обзор архитектуры](#обзор-архитектуры)
3. [Структура репозитория](#структура-репозитория)
4. [Установка](#установка)
5. [Быстрый старт — smoke-пайплайн](#быстрый-старт--smoke-пайплайн)
6. [Продакшн-пайплайн (полные данные)](#продакшн-пайплайн-полные-данные)
7. [DVC: подробное руководство](#dvc-подробное-руководство)
8. [Справочник по конфигурации](#справочник-по-конфигурации)
9. [Фазы пайплайна в деталях](#фазы-пайплайна-в-деталях)
10. [Создание и использование эмбеддингов](#создание-и-использование-эмбеддингов)
11. [Сервинг — эндпоинт /embed](#сервинг--эндпоинт-embed)
12. [Сервис разметки и active learning](#сервис-разметки-и-active-learning)
13. [Оценка качества](#оценка-качества)
14. [Трекинг экспериментов в MLflow](#трекинг-экспериментов-в-mlflow)
15. [Тестирование](#тестирование)
16. [Docker и docker-compose](#docker-и-docker-compose)
17. [Адаптация под другой язык](#адаптация-под-другой-язык)
18. [Справочник по Makefile](#справочник-по-makefile)
19. [Решение проблем](#решение-проблем)

---

## Что это такое

Цель исследования — обучить высококачественные sentence embeddings для языка с ограниченными ресурсами (в качестве примера используется гуджарати) и измерить, насколько каждое архитектурное решение влияет на качество.

**Три ветки, обучаемые на одинаковых данных от носителей языка:**

| Ветка | Подход | Ключевые компоненты |
|-------|--------|----------------------|
| **A** | С нуля | Собственный BPE-токенизатор → RoBERTa-подобный энкодер (MLM-претрейн) → SimCSE контрастивное дообучение |
| **B** | Многоязычный трансфер | mBERT / XLM-R, дообученные на тех же данных от носителей языка |
| **C** | LLM как эмбеддер | Декодерная LLM (стиль LLaMA) + LoRA-адаптеры, mean-pooling по последним скрытым состояниям (подход llm2vec) |

**Носители языка занимают центральное место:** они обеспечивают сигнал разметки через цикл активного обучения. Система подаёт аннотаторам наиболее «неопределённые» пары предложений, максимизируя информационную ценность каждой размеченной пары.

**Метрика качества:** коэффициент ранговой корреляции Спирмена на изолированном тестовом наборе STS (Semantic Textual Similarity). Тестовый набор никогда не используется при обучении — это архитектурный инвариант.

---

## Обзор архитектуры

Пайплайн состоит из фаз 0–7 плюс дополнительной фазы 4C:

| Фаза | Название | Описание |
|------|----------|----------|
| 0 | Нормализация | NFKC + IndicNLP (гуджарати) + схлопывание пробелов через `preprocess.normalize` |
| 1 | Корпус | Агрегация `data/raw/*.txt`, MinHash-дедупликация, запись `data/corpus.txt` |
| 2 | Токенизатор | BPE с Whitespace-претокенизатором, NFKC-нормализатором, vocab_size=8000 |
| 3 | MLM-претрейн | RoBERTa-подобный энкодер, HuggingFace Trainer, логирование в MLflow |
| 4 | SimCSE | Контрастивное дообучение (dropout-аугментированные позитивные пары) через SentenceTransformers |
| 4C | LLM LoRA | LoRA-адаптация декодерной LLM для получения эмбеддингов; базовые веса заморожены |
| 5 | Разметка | FastAPI-сервис: очередь active learning, сбор меток, экспорт триплетов |
| 6 | Оценка | Spearman, Recall@k, MRR@k на изолированном STS-тесте |
| 7 | Сервинг | FastAPI-эндпоинт `/embed`; L2-нормализованные векторы по HTTP |

**Три инварианта — нарушать нельзя:**

1. **Единая нормализация** — весь текст (при обучении и при сервинге) проходит через `langembed.preprocess.normalize`. Логика нормализации не дублируется.
2. **Запрет утечки теста** — файлы `data/sts_test_*` никогда не попадают в обучающие данные. Защита встроена в `build_corpus.py` и `evaluate.py`.
3. **Гиперпараметры из конфигов** — все гиперпараметры берутся из `configs/*.yaml`. Магические числа в коде обучения запрещены.

**Поток данных:**

```
сырые текстовые файлы
    │
    ▼
build_corpus  ──(MinHash-дедупликация)──► data/corpus.txt
    │
    ▼
train_tokenizer ───────────────────────► artifacts/tokenizer/
    │
    ▼
train_mlm  ──────(MLM-претрейн)────────► artifacts/encoder/
    │
    ▼
train_simcse  ──(SimCSE fine-tune)─────► artifacts/simcse/   ← Ветка A
    │
    ├── сервис разметки (цикл active learning)
    │       │
    │       ▼
    │   native_triplets.jsonl
    │       │
    │       ▼
    ├── train_supervised ────────────────► artifacts/supervised/
    │
    ▼
evaluate ──────────────────────────────► metrics/eval.json → MLflow
    │
    ▼
serve ─────────────────────────────────► POST /embed
```

---

## Структура репозитория

```
src/langembed/
  config.py              load_config() — единая точка входа для YAML-конфигов
  preprocess.py          normalize() — единственная функция нормализации текста
  data/
    build_corpus.py      агрегация + MinHash-дедупликация сырых текстов
    dedup.py             утилиты MinHash/LSH дедупликации
  tokenizer/
    train_tokenizer.py   BPE-токенизатор (библиотека HuggingFace tokenizers)
  pretrain/
    train_mlm.py         RoBERTa-подобный MLM-претрейн (HuggingFace Trainer)
  contrastive/
    train_simcse.py      SimCSE через SentenceTransformers
    train_supervised.py  supervised fine-tune на экспортированных триплетах
  llm_embed/
    train_lora.py        LoRA fine-tune декодерной LLM для эмбеддингов
    mntp.py              MNTP-претрейн (Masked Next-Token Prediction)
    model.py             обёртка LLM-эмбеддера
  annotation/
    api.py               FastAPI: /queue, /annotate, /export
    models.py            SQLAlchemy ORM: Annotator, Item, Annotation
    db.py                сессия БД / зависимость get_db
    active_learning.py   scoring неопределённости, выбор очереди
    quality.py           взвешенная каппа Коэна, агрегация с весами надёжности
  eval/
    evaluate.py          Spearman + Recall@k + MRR@k
  serving/
    serve.py             FastAPI-эндпоинт /embed

configs/
  tokenizer.yaml         Фазы 1+2 (корпус + токенизатор)
  pretrain.yaml          Фаза 3 (MLM-претрейн)
  contrastive.yaml       Фаза 4 (SimCSE / supervised)
  eval.yaml              Фаза 6 (оценка, пути к веткам)
  llm_embed.yaml         Фаза 4C (LLM LoRA)
  smoke/                 Конфиги smoke-пайплайна (английские фикстуры, только CPU)
    tokenizer.yaml
    pretrain.yaml
    contrastive.yaml
    eval.yaml

smoke/
  dvc.yaml               DVC smoke-пайплайн (5 стадий, английские фикстурные данные)
  dvc.lock               DVC lock-файл (закоммичен)

dvc.yaml                 Продакшн DVC-пайплайн (полные языковые данные)
Makefile                 make lint | test | test-e2e | corpus | pretrain | …
Dockerfile               Многостадийный: base (serve extras) / ml (+ torch)
docker-compose.yml       postgres, redis, annotation :8001, serve :8000, train

tests/
  conftest.py            Общие фикстуры (SQLite in-memory, TestClient)
  e2e/                   Полный пайплайн smoke-теста на английском языке
  fixtures/
    en_corpus.txt        Английские фикстурные предложения
    en_sts_test.jsonl    Английские STS-пары для E2E оценки

data/                    Отслеживается DVC (не в git)
artifacts/               Артефакты моделей под DVC (не в git)
metrics/
  smoke_eval.json        Метрики smoke-пайплайна (закоммичены, cache: false)
  eval.json              Метрики продакшн-оценки (под DVC)

docs/
  ru/README_RU.md        Русская документация (этот файл)
  IMPLEMENTATION_PLAN.md Пофазовый план реализации
```

---

## Установка

### Локально (CPU или GPU)

```bash
git clone https://github.com/DenisBaliuckij/langembed
cd langembed
pip install -e ".[ml,serve,dev]"
cp .env.example .env        # заполнить DATABASE_URL и REDIS_URL
```

**Зависимости по экстрам:**

| Экстра | Устанавливает | Когда нужен |
|--------|---------------|-------------|
| `ml` | torch, transformers, sentence-transformers, datasets | Фазы обучения |
| `serve` | fastapi, uvicorn, sqlalchemy, psycopg2 | Сервис разметки + сервинг |
| `dev` | ruff, mypy, pytest | Разработка |

Для локальной разработки установить все три:

```bash
pip install -e ".[ml,serve,dev]"
```

### Docker (полный стек)

```bash
cp .env.example .env
# Отредактировать .env: задать POSTGRES_PASSWORD, REDIS_URL, DATABASE_URL

docker compose up -d postgres redis

# Base-образ (~400 МБ): сервис разметки + сервинг, без torch
docker build --target base -t langembed-annotation .

# ML-образ (~4 ГБ): включает torch и все зависимости для обучения
docker build --target ml -t langembed-ml .

docker compose up -d
```

Сервисы после `docker compose up`:

| Сервис | Порт | Назначение |
|--------|------|------------|
| `annotation` | 8001 | API сервиса разметки с active learning |
| `serve` | 8000 | Эндпоинт инференса `/embed` |
| `train` | — | Одноразовый контейнер для обучения |
| `postgres` | 5432 | Хранилище разметки |
| `redis` | 6379 | Очередь задач |

---

## Быстрый старт — smoke-пайплайн

Smoke-пайплайн обучает маленькую английскую модель полностью за ~30 секунд на CPU, используя фикстурные данные из `tests/fixtures/`. Он проверяет весь кодовый путь без GPU и реальных языковых данных.

```bash
# Запустить полный smoke DVC-пайплайн (5 стадий)
make smoke-dvc

# Эквивалент:
python -m dvc repro smoke/dvc.yaml
```

**Что происходит:**

```
Стадия 1: corpus    — дедупликация tests/fixtures/en_corpus.txt → data/smoke/corpus_en.txt
Стадия 2: tokenizer — BPE vocab_size=500 → artifacts/smoke/tokenizer_en/
Стадия 3: pretrain  — RoBERTa hidden=128, 50 шагов → artifacts/smoke/encoder_en/
Стадия 4: simcse    — SimCSE 1 эпоха → artifacts/smoke/simcse_en/
Стадия 5: evaluate  — Spearman на en_sts_test.jsonl → metrics/smoke_eval.json
```

**Ожидаемый вывод после первого запуска:**

```
Running stage 'corpus':   ...
Running stage 'tokenizer': ...
Running stage 'pretrain':  ...
Running stage 'simcse':    ...
Running stage 'evaluate':  ...
branch en_smoke: Spearman = 0.6499
branch en_smoke: Recall@5=0.5000, MRR@5=0.3308
metrics written to metrics/smoke_eval.json
```

**Идемпотентность (второй запуск — всё из кэша):**

```bash
make smoke-dvc
# Стадия 'corpus': cached
# Стадия 'tokenizer': cached
# Стадия 'pretrain': cached
# Стадия 'simcse': cached
# Стадия 'evaluate': cached
# All stages are up to date.
```

DVC отслеживает хэши входных данных и перезапускает стадию только при изменении зависимостей.

**Просмотр smoke-метрик:**

```bash
dvc metrics show metrics/smoke_eval.json
# или:
cat metrics/smoke_eval.json
```

**Примечание по smoke-метрикам:** Spearman ≈ 0.65 для модели, обученной 50 шагов — ожидаемо ниже, чем у полностью обученной модели. Smoke-пайплайн проверяет корректность кода, а не качество модели.

---

## Продакшн-пайплайн (полные данные)

### 1. Подготовка сырых данных

```bash
# Поместить сырые текстовые файлы (по одному предложению в строке) в data/raw/
cp your_language_wiki.txt data/raw/wiki_gu.txt
cp your_other_corpus.txt  data/raw/news_gu.txt

# Подготовить тестовый набор STS (JSONL: sentence_a, sentence_b, score)
cp your_sts_test.jsonl data/sts_test_gu.jsonl
```

### 2. Запуск инфраструктуры

```bash
docker compose up -d postgres redis
```

### 3. Запуск полного DVC-пайплайна

```bash
dvc repro
```

Продакшн `dvc.yaml` выполняет стадии в порядке:

```
corpus → tokenizer → pretrain → simcse → supervised → evaluate
```

### 4. Запуск кампании разметки (Фаза 5)

```bash
make serve-annotation
# Документация API: http://localhost:8001/docs
```

См. [Сервис разметки и active learning](#сервис-разметки-и-active-learning).

### 5. Оценка всех веток

```bash
make eval
# Читает configs/eval.yaml (ветки A, B, C)
# Записывает metrics/eval.json + логирует в MLflow
```

### 6. Запуск сервинга эмбеддингов

```bash
LANGEMBED_MODEL_DIR=artifacts/embed_gu_v1 make serve
# Документация API: http://localhost:8000/docs
```

### 7. Просмотр результатов

```bash
mlflow ui          # http://localhost:5000
dvc metrics show   # вывести metrics/eval.json
dvc dag            # визуализировать граф пайплайна
```

---

## DVC: подробное руководство

### Основные команды

```bash
# Запустить все устаревшие стадии (продакшн)
dvc repro

# Запустить только smoke-пайплайн
dvc repro smoke/dvc.yaml
# или:
make smoke-dvc

# Проверить, какие стадии устарели (без запуска)
dvc status

# Вывести текущие метрики
dvc metrics show

# Сравнить метрики с предыдущим коммитом
dvc metrics diff HEAD~1

# Визуализировать продакшн-граф
dvc dag

# Визуализировать smoke-граф
dvc dag smoke/dvc.yaml
```

### Удалённое хранилище (командная работа)

```bash
# Настроить S3-remote (один раз на репозиторий)
dvc remote add -d myremote s3://my-bucket/langembed
dvc remote modify myremote region eu-central-1

# Загрузить все отслеживаемые данные и артефакты моделей
dvc push

# Воспроизвести эксперимент на другой машине
git clone https://github.com/DenisBaliuckij/langembed
cd langembed
dvc pull       # скачать артефакты с remote
dvc repro      # перезапустить устаревшие стадии
```

Любой член команды сможет воспроизвести точный эксперимент командами `dvc pull && dvc repro`.

### Как DVC отслеживает файлы

- Все артефакты, указанные в `outs:` в `dvc.yaml` / `smoke/dvc.yaml`, отслеживаются DVC по хэшу содержимого.
- Хэши хранятся в `.dvc/cache` и в lock-файлах (`dvc.lock`, `smoke/dvc.lock`).
- Большие файлы (веса моделей, датасеты) исключены из git через `.gitignore`.
- `metrics/smoke_eval.json` закоммичен в git напрямую (`cache: false`), поэтому он виден в `git log` и `dvc metrics show` без DVC-remote.

### Архитектурное примечание по smoke-пайплайну

Smoke-пайплайн находится в `smoke/dvc.yaml` (поддиректория репозитория). В каждой стадии задан `wdir: ..`, чтобы все пути в конфигах (например, `data/smoke/corpus_en.txt`, `artifacts/smoke/tokenizer_en`) разрешались относительно корня репозитория, а не поддиректории `smoke/`. Это соответствует соглашениям `.gitignore` и конфигурационных файлов.

---

## Справочник по конфигурации

Все гиперпараметры находятся в `configs/*.yaml`. Код обучения читает их через `langembed.config.load_config(path)`.

### `configs/tokenizer.yaml` (Фазы 1–2)

```yaml
language: en               # код языка для preprocess.normalize
data:
  raw_paths:               # список сырых текстовых файлов для агрегации
    - data/raw/wiki_gu.txt
  out_path: data/corpus_gu.txt
  test_path: data/sts_test_gu.jsonl  # защита от утечки: эти предложения исключаются из корпуса
tokenizer:
  vocab_size: 8000
  min_frequency: 2
  unk_rate_max: 0.05       # ошибка, если доля неизвестных токенов превышает порог
  out_dir: artifacts/tokenizer_gu
```

### `configs/pretrain.yaml` (Фаза 3)

```yaml
seed: 42
tokenizer_dir: artifacts/tokenizer_gu
corpus_path: data/corpus_gu.txt
report_to: [mlflow]        # бэкенды для трекинга экспериментов
out_dir: artifacts/encoder_gu
model:
  hidden_size: 512
  num_hidden_layers: 6
  num_attention_heads: 8
  intermediate_size: 2048
  max_position_embeddings: 514
  max_seq_length: 512
training:
  per_device_train_batch_size: 64
  gradient_accumulation_steps: 4
  learning_rate: 5.0e-4
  weight_decay: 0.01
  warmup_steps: 10000
  max_steps: 200000
  fp16: true               # установить false на CPU
  save_steps: 10000
  logging_steps: 500
  mlm_probability: 0.15
smoke:
  max_steps: 50            # используется только при флаге --smoke
```

### `configs/contrastive.yaml` (Фаза 4)

```yaml
seed: 42
encoder_dir: artifacts/encoder_gu
simcse:
  sentences_path: data/corpus_gu.txt
  out_dir: artifacts/simcse_gu
  batch_size: 128
  epochs: 3
  warmup_steps: 100
  max_seq_length: 512
```

### `configs/eval.yaml` (Фаза 6)

```yaml
test_path: data/sts_test_gu.jsonl
score_scale: 5.0           # делит сырые оценки, нормализуя к [0, 1]
retrieval_k: 10            # k для Recall@k и MRR@k
branches:                  # имя ветки → директория модели
  A: artifacts/embed_gu_v1
  B: artifacts/embed_gu_mling
  C: artifacts/embed_gu_llm
train_paths:               # защита от утечки: не должны содержать тестовые предложения
  - data/corpus_gu.txt
  - data/native_triplets.jsonl
metrics_path: metrics/eval.json
```

### Smoke-конфиги (`configs/smoke/`)

Smoke-конфиги повторяют продакшн с уменьшенными размерами:
- `vocab_size: 500` (против 8000 в продакшне)
- `hidden_size: 128`, `num_hidden_layers: 2` (против 512 / 6)
- `max_steps: 50` через флаг `--smoke`
- `batch_size: 8` (против 128)
- `test_path: data/smoke/sts_test_placeholder.jsonl` — файл намеренно отсутствует; защита от утечки возвращает пустое множество, потому что фикстурные предложения корпуса намеренно пересекаются с тестовыми

---

## Фазы пайплайна в деталях

### Фаза 0 — Нормализация

`langembed.preprocess.normalize(text, lang="gu")` — **единственная функция нормализации**, используемая повсеместно: при сборке корпуса, при обучении и при сервинге.

Применяемые шаги:
1. NFKC-нормализация Unicode
2. Нормализация скрипта через IndicNLP (гуджарати по умолчанию; пропускается для неиндийских языков)
3. Схлопывание пробелов

Для неиндийских языков никаких изменений кода не требуется — IndicNLP пропускается автоматически.

### Фаза 1 — Сбор корпуса

```bash
make corpus
# или: python -m langembed.data.build_corpus --config configs/tokenizer.yaml
```

Читает все файлы из `raw_paths`, нормализует каждое предложение, удаляет дубликаты через MinHash LSH (настраиваемый порог Жаккара) и записывает `out_path`. Защита от утечки исключает предложения из STS-тестового набора (`test_path`).

### Фаза 2 — BPE-токенизатор

```bash
make tokenizer
# или: python -m langembed.tokenizer.train_tokenizer --config configs/tokenizer.yaml
```

Обучает Byte-Pair Encoding токенизатор (библиотека HuggingFace `tokenizers`) с Whitespace-претокенизатором и NFKC-нормализатором. Артефакт: `artifacts/tokenizer_gu/` (словарь + merges + специальные токены).

### Фаза 3 — MLM-претрейн

```bash
# Полное обучение (200к шагов, рекомендуется GPU)
make pretrain

# CPU smoke (50 шагов, hidden=128)
make pretrain-smoke
```

Обучает RoBERTa-подобный энкодер с нуля через HuggingFace Trainer. Задача маскированного языкового моделирования с вероятностью маскировки 15%. Loss и perplexity логируются в MLflow. Артефакт: `artifacts/encoder_gu/`.

**Продакшн-параметры:** 200 000 шагов, batch_size=64, gradient_accumulation=4 (эффективный batch=256), hidden_size=512, 6 слоёв, 8 голов, fp16=true.

### Фаза 4 — SimCSE: контрастивное дообучение

```bash
# Полное обучение
make simcse

# CPU smoke (1 эпоха, 256 предложений)
make simcse-smoke
```

Несупервизированный SimCSE: каждое предложение проходит через энкодер дважды с разными dropout-масками, создавая два «вида» одного предложения как позитивную пару. `MultipleNegativesRankingLoss` использует все остальные предложения в батче как батч-негативы. Артефакт: `artifacts/simcse_gu/` (полная директория SentenceTransformer).

### Фаза 4C — LLM LoRA (ветка C)

```bash
make llm-mntp   # Masked Next-Token Prediction претрейн
make llm-lora   # LoRA fine-tuning
```

Адаптирует декодерную LLM (стиль LLaMA) для получения sentence embeddings. Все базовые веса модели заморожены; обучаются только LoRA-адаптеры. Эмбеддинги получаются через mean-pooling по последним скрытым состояниям. Конфиг: `configs/llm_embed.yaml`. Артефакт: `artifacts/llm_lora/`.

### Фаза 5 — Сервис разметки

См. [Сервис разметки и active learning](#сервис-разметки-и-active-learning).

### Фаза 6 — Оценка качества

```bash
make eval
# или: python -m langembed.eval.evaluate --config configs/eval.yaml

# Только smoke-модель:
python -m langembed.eval.evaluate --config configs/smoke/eval.yaml
```

Для каждой ветки из `configs/eval.yaml` вычисляется:
- **Spearman-корреляция** между предсказанными косинусными сходствами и человеческими оценками STS
- **Recall@k** — доля запросов, где правильный документ попал в топ-k
- **MRR@k** — Mean Reciprocal Rank первого правильного результата в топ-k

Результаты записываются в `metrics/eval.json` и логируются в MLflow. Перед загрузкой модели запускается защита от утечки теста.

### Фаза 7 — Сервинг

См. [Сервинг — эндпоинт /embed](#сервинг--эндпоинт-embed).

---

## Создание и использование эмбеддингов

### После запуска пайплайна

```python
from sentence_transformers import SentenceTransformer

# Загрузить любую обученную модель
model = SentenceTransformer("artifacts/smoke/simcse_en")  # smoke-модель (128 измерений)
# model = SentenceTransformer("artifacts/embed_gu_v1")    # продакшн-ветка A (512 измерений)

sentences = [
    "The cat sat on the mat.",
    "A cat was resting on a rug.",
    "The weather is sunny today.",
]

# Кодировать с L2-нормализацией
embeddings = model.encode(sentences, normalize_embeddings=True)
print(embeddings.shape)  # (3, 128) для smoke, (3, 512) для продакшна

# Косинусное сходство (L2-нормализованные → скалярное произведение = косинус)
import numpy as np
sims = embeddings @ embeddings.T
print(sims)
```

### Соблюдение инварианта нормализации

Весь текст должен проходить через `normalize` перед кодированием, как при обучении:

```python
from langembed.preprocess import normalize
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("artifacts/embed_gu_v1")

texts = ["Hello world", "Привет мир"]
embeddings = model.encode([normalize(t) for t in texts], normalize_embeddings=True)
```

Эндпоинт `/embed` применяет `normalize` автоматически. При прямом использовании Python — применяйте вручную.

### Батчевое кодирование больших датасетов

```python
from sentence_transformers import SentenceTransformer
from langembed.preprocess import normalize

model = SentenceTransformer("artifacts/embed_gu_v1")

with open("data/corpus_gu.txt", encoding="utf-8") as f:
    sentences = [normalize(line.strip()) for line in f if line.strip()]

embeddings = model.encode(
    sentences,
    batch_size=256,
    normalize_embeddings=True,
    show_progress_bar=True,
)
print(f"Закодировано {len(sentences)} предложений → форма {embeddings.shape}")
```

### Ручная оценка модели

```python
from langembed.config import load_config
from langembed.eval.evaluate import evaluate

cfg = load_config("configs/eval.yaml")
results = evaluate(cfg)
# {'spearman_A': 0.82, 'retrieval_recall@10_A': 0.75, 'retrieval_mrr@10_A': 0.61, ...}
```

---

## Сервинг — эндпоинт /embed

### Запуск сервера

```bash
# По умолчанию: загружает из artifacts/embed_gu_v1
make serve

# Кастомная модель:
LANGEMBED_MODEL_DIR=artifacts/smoke/simcse_en make serve

# Продакшн с несколькими воркерами:
LANGEMBED_MODEL_DIR=artifacts/embed_gu_v1 \
  uvicorn langembed.serving.serve:app \
  --host 0.0.0.0 --port 8000 --workers 4
```

Сервер запускается на `http://localhost:8000`. Интерактивная документация API: `http://localhost:8000/docs`.

### `POST /embed`

**Запрос:**

```json
{
  "texts": ["Привет мир", "Ещё одно предложение"]
}
```

**Ответ:**

```json
{
  "embeddings": [[0.12, -0.34, 0.56, ...], [0.78, 0.23, -0.11, ...]],
  "dim": 512
}
```

**curl:**

```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Привет мир", "Тестовое предложение"]}'
```

**Python:**

```python
import requests

response = requests.post(
    "http://localhost:8000/embed",
    json={"texts": ["Привет мир", "Ещё одно предложение"]}
)
data = response.json()
embeddings = data["embeddings"]  # list[list[float]]
dim = data["dim"]                # int
```

**Примечания:**
- Входные тексты автоматически нормализуются через `preprocess.normalize` на стороне сервера.
- Все возвращаемые векторы L2-нормализованы (‖v‖₂ = 1).
- Модель загружается один раз при старте сервера и используется для всех запросов.
- Директория модели задаётся через переменную окружения `LANGEMBED_MODEL_DIR`.

---

## Сервис разметки и active learning

Сервис разметки работает на порту 8001 и управляет циклом активного обучения для Фазы 5.

### Запуск сервиса

```bash
# Локально (нужны запущенные postgres + redis)
docker compose up -d postgres redis
make serve-annotation
# Интерактивная документация: http://localhost:8001/docs

# Docker
docker compose up -d annotation
```

### Рабочий процесс active learning

```
1. После SimCSE-обучения вычислить uncertainty для всех пар предложений:
       uncertainty(a, b) = 1 - |cos(embed(a), embed(b)) - 0.5| / 0.5
   Максимальная неопределённость при cos=0.5 (модель «неуверена»).
   Нулевая неопределённость при cos=0 или cos=1 (пара очевидна).

2. Загрузить пары с высокой неопределённостью в таблицу Item.

3. Аннотаторы получают пары:
       GET /queue?annotator_id=1&n=20
   → 20 неопределённых пар + 2 скрытых калибровочных «золотых» вопроса

4. Каждая пара получает оценку 0–5 (0=несвязные, 5=идентичный смысл):
       POST /annotate  {"item_id": 42, "annotator_id": 1, "label": 4.0}

5. Экспорт агрегированных меток как обучающих триплетов:
       POST /export  → записывает data/native_triplets.jsonl

6. Обучить supervised контрастивную модель на триплетах:
       make supervised

7. Пересчитать uncertainty корпуса новой моделью → пополнить очередь → повторить.
```

### Эндпоинты

#### `GET /queue?annotator_id={id}&n={n}`

```bash
curl "http://localhost:8001/queue?annotator_id=1&n=10"
```

```json
{
  "items": [
    {
      "id": 123,
      "sentence_a": "Кошка сидела на коврике.",
      "sentence_b": "Котёнок лежал на подстилке.",
      "uncertainty": 0.98,
      "status": "pending"
    }
  ]
}
```

#### `POST /annotate`

```bash
curl -X POST http://localhost:8001/annotate \
  -H "Content-Type: application/json" \
  -d '{"item_id": 123, "annotator_id": 1, "label": 4.0}'
```

```json
{"ok": true}
```

#### `POST /export`

```bash
curl -X POST "http://localhost:8001/export?out_path=data/native_triplets.jsonl"
```

```json
{"written": 847, "path": "data/native_triplets.jsonl"}
```

Формат триплетов в `data/native_triplets.jsonl`:

```json
{"anchor": "предложение A", "positive": "похожее предложение", "negative": "несвязанное предложение"}
```

Триплеты строятся из пар:
- с агрегированной оценкой ≥ 4.0 (позитивные)
- с агрегированной оценкой ≤ 1.0 (негативные)

### Качество аннотаторов

- Надёжность каждого аннотатора отслеживается через **взвешенную каппу Коэна**.
- Метки агрегируются с **весами, пропорциональными надёжности** — аннотаторы, систематически расходящиеся с консенсусом, получают меньший вес.
- «Золотые» калибровочные вопросы (status `"gold"`) имеют известные правильные ответы; они незаметно добавляются в очередь для выявления недобросовестных аннотаторов.

### Наполнение золотых вопросов

```bash
python scripts/seed_gold.py
```

Запустить один раз перед началом кампании разметки для наполнения таблицы золотых калибровочных вопросов.

---

## Оценка качества

### Запуск оценки

```bash
make eval
# или: python -m langembed.eval.evaluate --config configs/eval.yaml

# Только smoke-модель:
python -m langembed.eval.evaluate --config configs/smoke/eval.yaml
```

### Метрики

Для каждой ветки из `configs/eval.yaml`:

| Метрика | Значение |
|---------|----------|
| `spearman_{ветка}` | Корреляция Спирмена между предсказанными косинусами и оценками человека |
| `retrieval_recall@k_{ветка}` | Доля запросов, где правильный документ в топ-k |
| `retrieval_mrr@k_{ветка}` | Mean Reciprocal Rank первого правильного результата |

### Сравнение веток

```bash
# Вывести текущие метрики
dvc metrics show

# Сравнить с предыдущим коммитом
dvc metrics diff HEAD~1

# Визуальное сравнение в MLflow
mlflow ui   # Experiments → выбрать → отметить прогоны → Compare
```

### Защита от утечки теста

Перед загрузкой любой модели `evaluate.py` хэширует все предложения из тестового файла, затем проверяет каждый путь в `train_paths`. При совпадении оценка прерывается:

```
RuntimeError: Test leakage detected via data/corpus_gu.txt
```

---

## Трекинг экспериментов в MLflow

MLM-претрейн логирует loss и perplexity в MLflow на каждом шаге оценки. Фаза оценки записывает Spearman, Recall@k и MRR@k в `metrics/eval.json` и MLflow.

```bash
mlflow ui    # http://localhost:5000
```

Для сравнения веток: **Experiments → выбрать эксперимент → отметить прогоны → Compare**.

График Parallel Coordinates в MLflow удобен для визуализации связи гиперпараметров с корреляцией Спирмена на нескольких прогонах.

---

## Тестирование

### Unit-тесты (быстрые, без внешних зависимостей)

```bash
make test
# или: pytest tests/
```

Покрывают: `normalize`, `dedup`, `uncertainty_from_cosine`, `weighted_kappa`, `aggregate`.

### API contract-тесты (FastAPI TestClient + SQLite in-memory)

Включены в `make test`. Postgres и Redis не нужны.

Покрывают: `/queue`, `/annotate`, `/export`, `/embed`.

### E2E smoke-тест (полный CPU-пайплайн)

```bash
make test-e2e
# или: pytest -m e2e tests/e2e/ -v
```

Запускает полный пайплайн на английских фикстурных данных: `build_corpus` → tokenizer → MLM (50 шагов) → SimCSE (1 эпоха) → evaluate. Занимает 2–5 минут на CPU.

### Линтинг и проверка типов

```bash
make lint
# запускает:
#   ruff check src tests
#   ruff format --check src tests
#   mypy src
```

Все три проверки обязаны пройти перед коммитом. CI проверяет `make lint && make test`.

---

## Docker и docker-compose

### Сборка образов

```bash
# Разметка + сервинг (~400 МБ, без torch)
docker build --target base -t langembed-annotation .

# Полный ML-образ (~4 ГБ, включает torch)
docker build --target ml -t langembed-ml .
```

### Запуск сервисов

```bash
# Только инфраструктура
docker compose up -d postgres redis

# Полный стек
docker compose up -d

# Запуск обучения внутри ML-контейнера
docker compose run --rm train make corpus
docker compose run --rm train make pretrain

# Просмотр логов
docker compose logs -f annotation
docker compose logs -f serve
```

### Переменные окружения (`.env`)

```bash
POSTGRES_USER=langembed
POSTGRES_PASSWORD=secret
POSTGRES_DB=langembed
DATABASE_URL=postgresql://langembed:secret@postgres:5432/langembed
REDIS_URL=redis://redis:6379/0
LANGEMBED_MODEL_DIR=artifacts/embed_gu_v1
```

---

## Адаптация под другой язык

1. **`configs/tokenizer.yaml`** — установить `language: <lang_code>` и обновить `raw_paths`.
2. **`configs/eval.yaml`** — обновить `test_path` на путь к STS-тесту нового языка.
3. **`preprocess.normalize`** — для неиндийских языков IndicNLP пропускается автоматически; применяются только NFKC + схлопывание пробелов.
4. Запустить `dvc repro` — все стадии перезапустятся автоматически при изменении конфигов.

Изменений кода для неиндийских языков не требуется.

**Формат тестового STS-файла** (JSONL, по одной паре в строке):

```json
{"sentence_a": "Первое предложение", "sentence_b": "Второе предложение", "score": 3.5}
```

`score` должен быть в диапазоне `[0, score_scale]`, где `score_scale` задан в `configs/eval.yaml` (по умолчанию: 5.0).

---

## Справочник по Makefile

| Цель | Описание |
|------|----------|
| `make setup` | `pip install -e ".[ml,serve,dev]"` |
| `make lint` | `ruff check` + `ruff format --check` + `mypy` |
| `make test` | Unit-тесты + API contract-тесты |
| `make test-e2e` | Полный E2E smoke-тест на английском языке |
| `make corpus` | Фаза 1: сборка очищенного корпуса |
| `make tokenizer` | Фаза 2: обучение BPE-токенизатора |
| `make pretrain` | Фаза 3: MLM-претрейн (полный, 200к шагов) |
| `make pretrain-smoke` | Фаза 3: MLM-претрейн (50 шагов, CPU) |
| `make simcse` | Фаза 4: SimCSE контрастивное дообучение |
| `make simcse-smoke` | Фаза 4: SimCSE (1 эпоха, 256 предложений, CPU) |
| `make supervised` | Фаза 4: supervised обучение на триплетах |
| `make llm-mntp` | Фаза 4C: LLM Masked Next-Token Prediction |
| `make llm-lora` | Фаза 4C: LLM LoRA fine-tuning |
| `make serve-annotation` | Фаза 5: сервис разметки на порту 8001 |
| `make eval` | Фаза 6: оценка всех веток |
| `make serve` | Фаза 7: сервис `/embed` на порту 8000 |
| `make smoke-dvc` | Запустить полный smoke DVC-пайплайн |

---

## Решение проблем

### Стадия перезапускается неожиданно после `dvc repro`

DVC перезапускает стадию при изменении хэша любой зависимости. Выполнить `dvc status`, чтобы увидеть, какие зависимости изменились. Если изменён исходный файл, отслеживаемый как зависимость, все зависящие от него стадии перезапустятся.

### `RuntimeError: Test leakage detected`

Защита от утечки сработала: предложение из обучающего корпуса обнаружено в STS-тестовом наборе. Проверить, что `test_path` в `configs/eval.yaml` указывает на правильный файл. Для smoke-пайплайна это ожидаемо: фикстурные данные намеренно пересекаются — smoke-конфиг использует `data/smoke/sts_test_placeholder.jsonl` (отсутствующий файл) с `train_paths: []` для намеренного отключения защиты.

### Нехватка памяти при претрейне

- Уменьшить `per_device_train_batch_size` в `configs/pretrain.yaml`.
- Увеличить `gradient_accumulation_steps` для сохранения эффективного размера батча.
- Уменьшить `hidden_size` для получения меньшей модели.
- Включить `fp16: true` на GPU для уменьшения потребления памяти вдвое.

### MLflow не логирует метрики

- Убедиться, что MLflow UI запущен: `mlflow ui`.
- Проверить `report_to: [mlflow]` в `configs/pretrain.yaml`.
- Задать `MLFLOW_TRACKING_URI`, если используется удалённый MLflow-сервер.

### `make test-e2e` выполняется слишком долго

Ожидаемое время: 2–5 минут на CPU. Если выполнение занимает более 15 минут, уменьшить `smoke.max_steps` в `configs/pretrain.yaml` (по умолчанию: 50).

### Windows + Python 3.14: краш subprocess `sentence_transformers`

Симптом: subprocess завершается с кодом `3221225477` (нарушение доступа к памяти). Причина: конфликт порядка инициализации DLL расширения C для `pyarrow`. Исправление уже применено в `train_simcse.py` и `evaluate.py`: `datasets`, `pandas`, `pyarrow` и `torch` импортируются внутри тела функции перед `sentence_transformers`. Применять тот же паттерн в любом новом коде, использующем `sentence_transformers` в subprocess.
