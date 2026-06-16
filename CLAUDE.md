# CLAUDE.md — конвенции проекта langembed

Работай по фазам из `docs/IMPLEMENTATION_PLAN.md` строго по порядку. После
каждой фазы прогоняй её Acceptance и делай commit. Не начинай фазу N+1, пока
фаза N не зелёная.

## Инварианты

- **Единая нормализация.** Любой текст и на train, и на serve проходит через
  `langembed.preprocess.normalize`. Не дублировать логику нормализации.
- **Запрет утечки теста.** Файлы `data/sts_test_*` не попадают в обучающие
  выборки, SimCSE, supervised и active-learning-пул. Guard есть в
  `build_corpus.py` и `eval/evaluate.py`.
- **Конфиги.** Гиперпараметры только из `configs/*.yaml` через `config.load_config`.
  Никаких магических чисел в коде обучения.
- **Стиль.** `ruff format` + `ruff check`; типизация; `mypy` без ошибок; строка 100.
- **Воспроизводимость.** Фиксировать seed; артефакты в `artifacts/<stage>/`.

## Тяжёлые зависимости

ML-импорты (torch, transformers, sentence-transformers, tokenizers, datasets)
делаются **внутри функций**, чтобы модули оставались импортируемыми и
линтуемыми без полного ML-стека. Не выноси их на уровень модуля.

## Команды

`make setup | lint | test | corpus | tokenizer | pretrain[-smoke] | simcse[-smoke] | supervised | serve-annotation | eval | serve`

## Definition of Done

`make lint && make test` зелёные; прогон фаз 1→7 одной последовательностью;
метрики A/B в MLflow; `/embed` отдаёт векторы; данные/модели под DVC, не в git.
