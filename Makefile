.PHONY: setup lint test test-e2e corpus tokenizer pretrain pretrain-smoke train-llm simcse simcse-smoke supervised serve-annotation eval serve llm-mntp llm-lora smoke-dvc extract-pdf-ru corpus-ru tokenizer-ru pretrain-ru pretrain-ru-smoke simcse-ru simcse-ru-smoke seed-sts-ru eval-ru embed-ru

PY ?= python

setup:
	pip install -e ".[ml,serve,dev,translate]"

lint:
	ruff check src tests
	ruff format --check src tests
	mypy src

test:
	pytest $(filter-out $@,$(MAKECMDGOALS))

test-e2e:
	pytest -m e2e tests/e2e/ -v

corpus:
	$(PY) -m langembed.data.build_corpus --config configs/tokenizer.yaml

tokenizer:
	$(PY) -m langembed.tokenizer.train_tokenizer --config configs/tokenizer.yaml

pretrain:
	$(PY) -m langembed.pretrain.train_mlm --config configs/pretrain.yaml

pretrain-smoke:
	$(PY) -m langembed.pretrain.train_mlm --config configs/pretrain.yaml --smoke

train-llm:
	$(PY) -m langembed.llm.train_gpt --config configs/ru/llm.yaml

simcse:
	$(PY) -m langembed.contrastive.train_simcse --config configs/contrastive.yaml

simcse-smoke:
	$(PY) -m langembed.contrastive.train_simcse --config configs/contrastive.yaml --smoke

supervised:
	$(PY) -m langembed.contrastive.train_supervised --config configs/contrastive.yaml

serve-annotation:
	uvicorn langembed.annotation.api:app --port 8001 --reload

llm-mntp:
	$(PY) -m langembed.llm_embed.mntp --config configs/llm_embed.yaml

llm-lora:
	$(PY) -m langembed.llm_embed.train_lora --config configs/llm_embed.yaml

eval:
	$(PY) -m langembed.eval.evaluate --config configs/eval.yaml

serve:
	uvicorn langembed.serving.serve:app --port 8000 --reload

smoke-dvc:
	python -m dvc repro smoke/dvc.yaml

extract-pdf-ru:
	$(PY) -m langembed.data.extract_text --pdf data/raw/voina-i-mir.pdf --out data/raw/voina_i_mir_ru.txt

corpus-ru:
	$(PY) -m langembed.data.build_corpus --config configs/ru/tokenizer.yaml

tokenizer-ru:
	$(PY) -m langembed.tokenizer.train_tokenizer --config configs/ru/tokenizer.yaml

pretrain-ru:
	$(PY) -m langembed.pretrain.train_mlm --config configs/ru/pretrain.yaml

pretrain-ru-smoke:
	$(PY) -m langembed.pretrain.train_mlm --config configs/ru/pretrain.yaml --smoke

simcse-ru:
	$(PY) -m langembed.contrastive.train_simcse --config configs/ru/contrastive.yaml

simcse-ru-smoke:
	$(PY) -m langembed.contrastive.train_simcse --config configs/ru/contrastive.yaml --smoke

seed-sts-ru:
	$(PY) scripts/seed_sts_pairs.py --config configs/ru/contrastive.yaml --n 60

eval-ru:
	$(PY) -m langembed.eval.evaluate --config configs/ru/eval.yaml

embed-ru:
	$(PY) scripts/embed_corpus.py --config configs/ru/contrastive.yaml --out artifacts/embeddings_ru/embeddings.jsonl

%:
	@:
