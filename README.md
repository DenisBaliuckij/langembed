# langembed

Пайплайн обучения семантических эмбеддингов **с нуля** для произвольного
естественного языка (пример: гуджарати), с контуром носителей языка и active
learning. Дизайн — две ветки на одних данных и одной метрике: A (всё с нуля) и
B (мультиязычный контроль) и C (LLM как эмбеддер: LLM2Vec / instruction-embeddings).

## Быстрый старт

```bash
make setup                         # зависимости (ml + serve + dev)
docker compose up -d postgres redis
make lint && make test             # проверка каркаса
```

## Фазы (детали — docs/IMPLEMENTATION_PLAN.md)

```
0 каркас        -> make lint, docker compose ps
1 корпус        -> make corpus
2 токенизатор   -> make tokenizer
3 MLM (с нуля)  -> make pretrain-smoke, make pretrain
4 contrastive   -> make simcse, make supervised
4C ветка C (LLM) -> make llm-lora   (LLM2Vec / instruction-embeddings)
5 носители      -> make serve-annotation  (FastAPI :8001)
6 оценка        -> make eval              (A vs B)
7 сервинг       -> make serve             (FastAPI :8000)
```

## Воспроизводимый прогон (DVC)

Стадии 1→6 связаны в DAG в `dvc.yaml` (сервинг — рантайм, в пайплайн не входит):

```bash
dvc repro            # corpus -> tokenizer -> pretrain -> simcse -> supervised -> evaluate
dvc metrics show     # Spearman по веткам A/B из metrics/eval.json
```

Внешние входы (положить до `dvc repro`): `data/raw/*` (сырой текст),
`data/sts_test_gu.jsonl` (изолированный тест), `data/native_triplets.jsonl`
(экспорт из сервиса носителей, Phase 5).

## CI

`.github/workflows/ci.yml` на каждый push/PR ставит только лёгкие зависимости
(`.[dev]`, без GPU-стека) и гоняет `ruff check`, `ruff format --check`, `mypy src`,
`pytest`. Тесты покрывают чистые функции и не требуют torch/fastapi.

## Инварианты (нарушать нельзя)

- Единый `langembed.preprocess.normalize` на train и serve.
- STS-тест (`data/sts_test_*`) не попадает ни в одну обучающую выборку.
- Гиперпараметры — только через `configs/*.yaml`.
