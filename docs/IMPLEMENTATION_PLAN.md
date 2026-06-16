# План реализации для Claude Code: пайплайн эмбеддингов «с нуля»

> **Назначение файла.** Это исполняемая build-спецификация для Claude Code. Работай **строго по фазам**, по порядку. После каждой фазы прогоняй её *критерии приёмки* (раздел «Acceptance») и **делай git-commit** перед переходом к следующей. Не начинай фазу N+1, пока не зелёная фаза N. Если критерий не выполняется — чини в рамках текущей фазы.
>
> Рекомендация по размещению: сохрани этот файл как `docs/IMPLEMENTATION_PLAN.md`, а раздел **«Конвенции»** скопируй в корневой `CLAUDE.md`, чтобы он попадал в контекст каждой сессии.

---

## 0. Цель проекта

Построить пайплайн, который для произвольного естественного языка (пример: **гуджарати**) обучает семантические эмбеддинги предложений **с нуля**: собственный токенизатор → MLM-предобучение энкодера → contrastive-обучение. Носители языка встроены как источник обучающего сигнала (через сервис разметки с active learning) и контур контроля качества.

Дизайн — **две ветки на одних данных и одной метрике**:
- **Ветка A** (исследовательская): токенизатор + энкодер + эмбеддинги, всё с нуля.
- **Ветка B** (научный контроль): мультиязычная модель, дообученная на тех же данных носителей.

Итоговый артефакт исследования — сравнение A vs B по корреляции Спирмена на изолированном STS-тесте.

---

## 1. Стек и предпосылки

- **Python** 3.11, **PyTorch** (CUDA), `transformers`, `sentence-transformers`, `tokenizers`, `datasets`
- **Сервис носителей**: `fastapi`, `uvicorn`, `sqlalchemy`, `pydantic-settings`, `psycopg[binary]`; очереди — `celery` + `redis`
- **Хранилище**: PostgreSQL + расширение **pgvector**
- **Данные/качество**: `indic-nlp-library`, `datasketch`, `scikit-learn`, `numpy`
- **Тулинг**: `ruff` (lint+format), `mypy`, `pytest`, `dvc` (версии данных/моделей), `mlflow`
- **Разметка**: Label Studio (внешний сервис, конфиги храним в репозитории)

GPU нужен для фаз 3–4. Фазы 0–2, 5–7 проходят на CPU (кроме инференса модели в 6–7).

---

## 2. Структура репозитория (создать в фазе 0)

```
langembed/
├── CLAUDE.md                      # конвенции (см. §3)
├── README.md
├── pyproject.toml                 # зависимости + конфиг ruff/mypy/pytest
├── Makefile                       # единые команды (см. §11)
├── .env.example
├── docker-compose.yml             # postgres+pgvector, redis
├── configs/
│   ├── tokenizer.yaml
│   ├── pretrain.yaml
│   ├── contrastive.yaml
│   └── eval.yaml
├── data/                          # gitignore
├── artifacts/                     # gitignore (модели, чекпойнты)
├── src/langembed/
│   ├── __init__.py
│   ├── config.py                  # pydantic-settings, загрузка YAML
│   ├── preprocess.py              # normalize() — ЕДИНЫЙ для train и serve
│   ├── data/
│   │   ├── build_corpus.py
│   │   └── dedup.py
│   ├── tokenizer/train_tokenizer.py
│   ├── pretrain/train_mlm.py
│   ├── contrastive/
│   │   ├── train_simcse.py
│   │   └── train_supervised.py
│   ├── annotation/
│   │   ├── db.py                  # engine, session, Base
│   │   ├── models.py              # Item, Annotator, Annotation
│   │   ├── active_learning.py     # uncertainty()
│   │   ├── quality.py             # kappa, aggregate, reliability
│   │   ├── api.py                 # FastAPI: /queue, /annotate, /export
│   │   └── label_studio/
│   │       ├── sts_pairwise.xml
│   │       └── triplet_choice.xml
│   ├── eval/evaluate.py
│   └── serving/serve.py           # FastAPI: /embed
├── scripts/seed_gold.py
└── tests/
    ├── test_preprocess.py
    ├── test_dedup.py
    ├── test_tokenizer.py
    ├── test_quality.py
    └── test_active_learning.py
```

---

## 3. Конвенции (скопировать в `CLAUDE.md`)

- **Инвариант train/serve.** Любой текст и на обучении, и на инференсе проходит через **одну и ту же** функцию `langembed.preprocess.normalize`. Нигде не дублировать логику нормализации. Тест `test_preprocess.py` фиксирует это поведение.
- **Запрет утечки теста.** Файлы `data/sts_test_*.jsonl` **никогда** не попадают ни в обучающие выборки, ни в SimCSE, ни в supervised, ни в active-learning-пул. Добавить guard в `build_corpus.py`, который падает, если пересечение по хэшам непустое.
- **Конфиги — только через `config.py`.** Никаких «магических чисел» в коде обучения; все гиперпараметры читаются из `configs/*.yaml` через pydantic-модели.
- **Стиль.** `ruff format` + `ruff check`; типизация обязательна, `mypy` без ошибок. Длина строки 100.
- **Воспроизводимость.** Фиксировать seed во всех скриптах обучения; путь к артефакту = `artifacts/<branch>/<stage>/`.
- **Коммиты.** Один commit на завершённую фазу, сообщение вида `feat(phaseN): <итог>`. После коммита — короткий отчёт что сделано и какие acceptance прошли.
- **Не выходить за рамки фазы.** Не реализовывать заранее то, что относится к будущим фазам.

---

## 4. Фаза 0 — Скелет проекта

**Цель:** рабочий каркас, окружение, CI-команды.

Задачи:
- [ ] Инициализировать репозиторий и структуру каталогов из §2.
- [ ] `pyproject.toml`: зависимости (§1), конфиг `ruff`, `mypy`, `pytest`.
- [ ] `config.py`: `Settings` (pydantic-settings) + функция `load_config(path) -> dict`.
- [ ] `docker-compose.yml`: сервисы `postgres` (образ с pgvector) и `redis`; `.env.example` с переменными подключения.
- [ ] `Makefile` с целями из §11 (заглушки, где код ещё не готов).
- [ ] Пустые модули с сигнатурами (см. соответствующие фазы) и `TODO`.

**Acceptance:**
```bash
make setup        # окружение ставится без ошибок
make lint         # ruff + mypy зелёные
docker compose up -d postgres redis && docker compose ps   # оба healthy
```

---

## 5. Фаза 1 — Данные: нормализация, корпус, дедуп

**Цель:** из сырых источников получить чистый одноязычный корпус.

Реализовать:
```python
# src/langembed/preprocess.py
def normalize(text: str) -> str: ...
    # NFC + indic-нормализация (язык из конфига) + сжатие пробелов

# src/langembed/data/dedup.py
def dedup(docs: list[str], threshold: float = 0.8) -> list[str]: ...
    # MinHash + LSH, near-дубликаты

# src/langembed/data/build_corpus.py
def build_corpus(raw_paths: list[str], out_path: str, test_hashes: set[str]) -> int: ...
    # normalize -> фильтр языка -> dedup -> guard на утечку теста -> запись JSONL
    # возвращает число строк; падает при пересечении с test_hashes
```

Задачи:
- [ ] `normalize()` + юнит-тест (идемпотентность: `normalize(normalize(x)) == normalize(x)`).
- [ ] `dedup()` + тест (искусственные дубликаты схлопываются).
- [ ] `build_corpus()` с guard против утечки теста (см. §3) + тест, что guard падает при пересечении.
- [ ] Цель `make corpus`.

**Acceptance:**
```bash
make test -- tests/test_preprocess.py tests/test_dedup.py   # зелёные
make corpus                                                 # создаёт data/corpus_gu.txt
wc -l data/corpus_gu.txt                                    # > 0; залогировать размер
```

---

## 6. Фаза 2 — Токенизатор с нуля

**Цель:** субтокенизатор, обученный только на целевом корпусе.

Реализовать `train_tokenizer.py` (HF `tokenizers`, BPE, `NFKC`, whitespace-pretokenizer, спецтокены `<s><pad></s><unk><mask>`), сохранение в `artifacts/tokenizer_gu/` и обёртку `PreTrainedTokenizerFast`. Параметры (`vocab_size`, `min_frequency`) — из `configs/tokenizer.yaml`.

Задачи:
- [ ] Обучение и сохранение токенизатора.
- [ ] Диагностика: посчитать долю `<unk>` и среднее число субтокенов на слово на отложенной выборке; залогировать.
- [ ] Тест: round-trip `decode(encode(x))` сохраняет нормализованный текст; доля `<unk>` ниже порога из конфига.
- [ ] Цель `make tokenizer`.

**Acceptance:**
```bash
make tokenizer
make test -- tests/test_tokenizer.py          # зелёный
# в логе: unk_rate < 0.01, mean_subtokens_per_word в разумных пределах
```

---

## 7. Фаза 3 — MLM-предобучение энкодера с нуля

**Цель:** энкодер, обученный masked LM на корпусе (случайная инициализация).

Реализовать `train_mlm.py`: `RobertaConfig` (размер из `configs/pretrain.yaml` — по умолчанию hidden 512 / 6 слоёв / 8 голов), `RobertaForMaskedLM` (без предобученных весов), `DataCollatorForLanguageModeling(mlm_probability=0.15)`, HF `Trainer`. Логирование в MLflow. Сохранение в `artifacts/encoder_gu/`.

Задачи:
- [ ] Конфиг модели и обучения из YAML; фикс seed.
- [ ] Запуск обучения с чекпойнтами и early-логикой по eval-перплексии на отложенном сете.
- [ ] Smoke-режим (`--max-steps 50`) для быстрой проверки пайплайна без полного прогона.
- [ ] Цель `make pretrain` (+ `make pretrain-smoke`).

**Acceptance:**
```bash
make pretrain-smoke           # 50 шагов проходят без ошибок, loss конечен
# полный прогон: make pretrain -> в MLflow перплексия выходит на плато
ls artifacts/encoder_gu/      # есть config.json + веса
```

---

## 8. Фаза 4 — Эмбеддинги: contrastive

**Цель:** из энкодера получить эмбеддинги предложений (unsupervised → supervised).

Реализовать:
```python
# train_simcse.py — unsupervised SimCSE: пары (s, s), MultipleNegativesRankingLoss,
#   mean-pooling; вход artifacts/encoder_gu -> выход artifacts/simcse_gu

# train_supervised.py — триплеты (anchor, positive, negative) из data/native_triplets.jsonl,
#   MultipleNegativesRankingLoss; вход artifacts/simcse_gu -> выход artifacts/embed_gu_v1
```
Параметры (batch, epochs, warmup) — из `configs/contrastive.yaml`. Большой batch важен для in-batch негативов.

Задачи:
- [ ] `train_simcse.py` + smoke-режим.
- [ ] `train_supervised.py`; если `native_triplets.jsonl` пуст — корректно завершиться с понятным сообщением (данные появятся после фазы 5).
- [ ] Цели `make simcse`, `make supervised`.

**Acceptance:**
```bash
make simcse-smoke             # обучение идёт, loss падает
ls artifacts/simcse_gu/       # модель сохранена
# supervised прогоняется после фазы 5 (когда есть триплеты)
```

---

## 9. Фаза 5 — Сервис носителей + active learning

**Цель:** сервис, который выдаёт носителям информативные задания, собирает разметку, контролирует качество и экспортирует триплеты для фазы 4.

Реализовать:
```python
# annotation/models.py  — Item(sentence_a, sentence_b, uncertainty, status, gold_label),
#                          Annotator(name, reliability), Annotation(item_id, annotator_id, label)
# annotation/active_learning.py
def uncertainty(pairs: list[tuple[str, str]]) -> np.ndarray: ...
    # 1 - |cos - 0.5| / 0.5, модель artifacts/embed_gu_v1 (или simcse_gu на старте)
# annotation/quality.py
def weighted_kappa(a, b) -> float: ...
def aggregate(labels, reliabilities) -> float: ...
def update_reliability(correct_on_gold, total_gold, prior=2.0) -> float: ...
# annotation/api.py — FastAPI:
#   GET  /queue?annotator_id&n   -> n самых неуверенных + 2 gold-вопроса
#   POST /annotate               -> сохранить Annotation
#   POST /export                 -> собрать триплеты -> data/native_triplets.jsonl
```
Конфиги Label Studio (`sts_pairwise.xml`, `triplet_choice.xml`) положить в `annotation/label_studio/`. `scripts/seed_gold.py` — наполнение gold-набора.

Задачи:
- [ ] Модели + `db.py` (engine/session), миграция таблиц.
- [ ] `uncertainty()` + тест (граничные пары получают высший скор).
- [ ] `quality.py` + тесты (kappa на согласных метках = 1.0; агрегация уважает веса).
- [ ] FastAPI-эндпойнты `/queue`, `/annotate`, `/export`.
- [ ] Экспорт триплетов: из оценок близости строить (anchor, positive, hard-negative).
- [ ] Цель `make serve-annotation`.

**Acceptance:**
```bash
make test -- tests/test_quality.py tests/test_active_learning.py   # зелёные
make serve-annotation &                                            # FastAPI поднимается
curl "localhost:8001/queue?annotator_id=1&n=5"                     # возвращает items + gold
# после раунда разметки: POST /export создаёт непустой data/native_triplets.jsonl
```

---

## 10. Фаза 6 — Оценка (A vs B)

**Цель:** честное сравнение веток на изолированном тесте.

Реализовать `evaluate.py`: `EmbeddingSimilarityEvaluator` (Спирмен) на `data/sts_test_gu.jsonl`; прогон для обеих веток (`artifacts/embed_gu_v1` и `artifacts/embed_gu_mling`); вывод таблицы и запись метрик в MLflow.

Задачи:
- [ ] Подготовить ветку B: дообучить выбранную мультиязычную модель на тех же `native_triplets.jsonl`, сохранить в `artifacts/embed_gu_mling`.
- [ ] `evaluate.py` считает Спирмена по обеим веткам + retrieval@k.
- [ ] Проверка guard: упасть, если тест пересекается с любыми обучающими файлами.
- [ ] Цель `make eval`.

**Acceptance:**
```bash
make eval        # печатает Spearman для A и B + retrieval@k; пишет в MLflow
# guard на утечку теста срабатывает на подложенном пересечении (тест)
```

---

## 11. Фаза 7 — Сервинг

**Цель:** инференс-сервис эмбеддингов поверх той же нормализации + векторное хранилище.

Реализовать `serve.py` (FastAPI `POST /embed`): применяет `preprocess.normalize`, кодирует моделью `artifacts/embed_gu_v1`, возвращает векторы. Миграция pgvector:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE documents (id BIGSERIAL PRIMARY KEY, text TEXT NOT NULL, embedding VECTOR(512));
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);
```

Задачи:
- [ ] `POST /embed` с нормализацией и батчингом.
- [ ] Миграция pgvector + функция upsert документа с эмбеддингом.
- [ ] Тест: один и тот же текст через `/embed` и напрямую через `normalize`+модель даёт идентичный вектор (доказательство отсутствия skew).
- [ ] Цель `make serve`.

**Acceptance:**
```bash
make serve &
curl -X POST localhost:8000/embed -H "Content-Type: application/json" \
     -d '{"texts":["..."]}'      # возвращает embeddings + dim
```

---

## 12. Makefile — единые команды

```
make setup              # установка зависимостей (uv/pip)
make lint               # ruff check + ruff format --check + mypy
make test               # pytest (можно: make test -- <paths>)
make corpus             # фаза 1
make tokenizer          # фаза 2
make pretrain[-smoke]   # фаза 3
make simcse[-smoke]     # фаза 4 (unsupervised)
make supervised         # фаза 4 (supervised, нужны триплеты)
make serve-annotation   # фаза 5 (FastAPI, порт 8001)
make eval               # фаза 6
make serve              # фаза 7 (FastAPI, порт 8000)
```

---

## 13. Стратегия тестирования

- **Юнит-тесты — на каждую чистую функцию**: `normalize`, `dedup`, `uncertainty`, `weighted_kappa`, `aggregate`, `update_reliability`.
- **Smoke-тесты обучения** (`*-smoke`, единицы шагов) — что пайплайн обучения не падает; не путать с полным прогоном.
- **Контрактные тесты API** — `/queue`, `/annotate`, `/embed` (через FastAPI `TestClient`, БД — sqlite/инмемори или контейнерный postgres).
- **Guard-тесты** — утечка теста и инвариант train/serve обязаны иметь явные падающие сценарии.

---

## 14. Definition of Done (весь проект)

- [ ] `make lint && make test` зелёные.
- [ ] Воспроизводимый прогон фаз 1→7 одной последовательностью команд.
- [ ] Метрики A и B на изолированном тесте записаны в MLflow; таблица сравнения в `README.md`.
- [ ] Сервис `/embed` отдаёт векторы; доказан нулевой train/serve skew (тест из фазы 7).
- [ ] Данные и модели версионируются через DVC; артефакты не в git.

---

## 15. Guardrails для агента

- Не коммить содержимое `data/` и `artifacts/` (только `.gitignore`-записи и DVC-указатели).
- Не хардкодить гиперпараметры — только через `configs/*.yaml`.
- Не дублировать нормализацию — единственный источник истины `preprocess.normalize`.
- Не трогать `data/sts_test_*` в обучающих путях — это нарушение валидности эксперимента.
- При нехватке данных (пустые триплеты на старте фазы 4) — корректно завершаться с подсказкой запустить фазу 5, а не падать с трейсбеком.
- Полные прогоны обучения (фазы 3–4) запускать только после успешных smoke-режимов.

---

## Фаза 4C — Ветка C: LLM как эмбеддер (LLM2Vec / instruction-embeddings)

**Цель:** третья ветка эксперимента — декодерная LLM в роли эмбеддера, на тех же
данных носителей и той же метрике, что A и B. Современный рецепт (e5-mistral,
gte-Qwen, NV-Embed, LLM2Vec): взять decoder-LLM, заменить голову next-token на
пулинг (last-token/mean), добавить инструкцию-префикс и contrastive-дообучение,
параметр-эффективно через LoRA.

Два режима (`configs/llm_embed.yaml`, поле `mode`):
- **ready_embedder** (по умолчанию, быстрый путь): база — уже instruction-эмбеддер,
  мультиязычный и instruction-aware (например `Qwen/Qwen3-Embedding-0.6B`).
  LoRA-дообучение на триплетах носителей.
- **llm2vec** (research-путь): база — «сырая» decoder-LLM. Сначала MNTP-адаптация
  (`mntp.py` через библиотеку `llm2vec`) для перевода в двунаправленное внимание,
  затем LoRA-contrastive. Сильные под гуджарати базы: `sarvamai/sarvam-1` (2B,
  10 индийских языков, эффективная работа с индийским письмом), `krutrim-ai/Krutrim-2`
  (12B), `Qwen/Qwen2.5-1.5B`.

Задачи:
- [ ] `train_lora.py`: строит SentenceTransformer (Transformer+Pooling) с LoRA-адаптером
      и инструкцией как default-промптом → артефакт совместим с harness'ом фазы 6.
- [ ] (только llm2vec) `mntp.py`: MNTP-адаптация через `llm2vec`.
- [ ] Ветка C добавлена в `configs/eval.yaml` → фаза 6 печатает Spearman для A/B/C.
- [ ] DVC-стадия `llm_lora` (→ `artifacts/embed_gu_llm`) связана как зависимость `evaluate`.
- [ ] Тесты `test_llm_embed.py` (форматирование инструкции, индексы last-token).

**Acceptance:**
```bash
make test -- tests/test_llm_embed.py        # зелёные
make llm-lora                               # LoRA-дообучение -> artifacts/embed_gu_llm
make eval                                    # Spearman для A, B и C в одной таблице
```

**Заметки по ресурсам:** для decoder-LLM включена 4-bit квантизация
(bitsandbytes) + LoRA — это позволяет дообучать 0.6–2B модели на одной GPU;
last-token пулинг — стандарт для decoder-эмбеддеров; инструкция «зашита» как
default-промпт, чтобы оценка A/B/C шла единообразно на «сырых» предложениях.
