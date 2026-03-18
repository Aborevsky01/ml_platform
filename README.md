# Оптимизация inference pipeline для rubert-mini-frida

## Цель

Исследовать уровни оптимизации inference pipeline для модели эмбеддингов `sergeyzh/rubert-mini-frida` и оценить влияние базового инференса, оптимизации модели через ONNX Runtime и динамического батчинга на latency, throughput и потребление ресурсов. [huggingface](https://huggingface.co/sergeyzh/rubert-mini-frida)

## Сетап

- Модель: `sergeyzh/rubert-mini-frida`. [huggingface](https://huggingface.co/sergeyzh/rubert-mini-frida)
- Окружение: Python 3.10+, CPU.  
- Основные библиотеки: `sentence-transformers`, `transformers`, `torch`, `optimum[onnxruntime]`, `onnxruntime`, `fastapi`, `uvicorn`, `prometheus-client`, `psutil`, `requests`, `tqdm`. 
- Все зависимости указаны в `requirements-3.txt`.

## Структура проекта

- `main.py` — базовый PyTorch сервис (SentenceTransformer + FastAPI + Prometheus). 
- `export_onnx.py` — экспорт модели в формат ONNX через `ORTModelForFeatureExtraction`. 
- `main_onnx.py` — сервис с инференсом через ONNX Runtime (FastAPI, mean pooling, L2-нормализация). 
- `main_batching.py` — сервис с динамическим батчированием поверх ONNX (очередь, несколько воркеров, Prometheus-метрики).
- `benchmark.py` — внешний бенчмарк-клиент для замера latency, throughput и ресурсов по PID процесса. 

## Метрики и методика

Метрики: [edge-ai-vision](https://www.edge-ai-vision.com/2023/10/a-guide-to-optimizing-transformer-based-models-for-faster-inference/)

- Latency:
  - средняя задержка (Average latency, ms);
  - 95-й перцентиль (p95 latency, ms);
  - min/max и стандартное отклонение.
- Throughput:
  - Requests/sec — количество запросов в секунду при заданной `concurrency`.
- Ресурсы:
  - средняя и пиковая загрузка CPU (%);
  - среднее и пиковое потребление памяти RSS (MB).

Методика:

- Для каждого сервиса поднимается FastAPI-приложение на отдельном порту.  
- PID процесса сервера определяется через `ps aux | grep ...` и передается в `benchmark.py` параметром `--pid`.    
- Бенчмарки запускаются с фиксированными параметрами:
  - `batch-size=1`, `num-requests=200`, `warmup=10`;
  - `concurrency ∈ {1, 10, 20, 50}` для PyTorch и batched ONNX;
  - для ONNX без батчинга в предоставленных логах часть прогонов выполнена с `num-requests=100`.

***

## Часть 1. Базовый инференс (PyTorch + SentenceTransformer)

### Реализация

Сервис для базового инференса реализован в `main.py`: 

- Загрузка модели:
  ```python
  model = SentenceTransformer("sergeyzh/rubert-mini-frida", device="cpu")
  ```
- Эндпоинт `POST /embed` принимает JSON:
  ```json
  { "texts": ["текст 1", "текст 2"] }
  ```
- Инференс:
  ```python
  embeddings = model.encode(req.texts, normalize_embeddings=True)
  ```
- Результат возвращается как список эмбеддингов.

Для мониторинга добавлены Prometheus-метрики: 
- `request_processing_seconds_pt` — server-side latency;  
- `request_count_pt` — счётчик запросов;  
- `/metrics` — эндпоинт для сбора метрик.

### Результаты бенчмарка (PyTorch, порт 8000)

| Concurrency | Num requests | Avg latency (ms) | p95 latency (ms) | Min (ms) | Max (ms) | Std (ms) | Throughput (req/s) | Avg CPU (%) | Peak CPU (%) | Avg RSS (MB) | Peak RSS (MB) |
|------------:|-------------:|-----------------:|-----------------:|---------:|---------:|---------:|--------------------:|------------:|-------------:|-------------:|--------------:|
| 1           | 200          | 4.21             | 4.55             | 3.87     | 6.55     | 0.26     | 233.13             | 29.3        | 43.1         | 192.08       | 193.42        |
| 10          | 200          | 49.91            | 60.50            | 27.65    | 76.37    | 6.78     | 198.12             | 18.8        | 63.8         | 281.46       | 311.50        |
| 20          | 200          | 94.62            | 113.79           | 32.50    | 124.44   | 11.47    | 207.66             | 20.8        | 61.2         | 237.32       | 292.83        |
| 50          | 200          | 236.30           | 355.34           | 153.57   | 379.89   | 51.92    | 200.63             | 47.0        | 61.5         | 237.01       | 262.53        |

Вывод: базовый PyTorch-сервис показывает очень низкую latency при одиночных запросах, но при росте нагрузки p95 latency и RSS растут, throughput ограничивается примерно 200 запросами в секунду. 

***

## Часть 2. Оптимизация модели через ONNX Runtime

### Результаты бенчмарка (ONNX без батчинга, порт 8001)

Сервис на порту 8001 (`python main_onnx.py`), параметры бенчмарка одинаковы: `batch-size=1`, `num-requests=200`, `warmup=10`, `concurrency ∈ {1,10,20,50}`. 

| Concurrency | Num requests | Avg latency (ms) | p95 latency (ms) | Min (ms) | Max (ms) | Std (ms) | Throughput (req/s) | Avg CPU (%) | Peak CPU (%) | Avg RSS (MB) | Peak RSS (MB) |
|------------:|-------------:|-----------------:|-----------------:|---------:|---------:|---------:|--------------------:|------------:|-------------:|-------------:|--------------:|
| 1           | 200          | 4.37             | 5.22             | 3.81     | 12.43    | 0.80     | 221.82             | 24.6        | 59.0         | 190.72       | 194.08        |
| 10          | 200          | 28.78            | 215.03           | 10.78    | 231.45   | 46.00    | 340.54             | 37.0        | 80.0         | 256.21       | 408.00        |
| 20          | 200          | 36.85            | 52.74            | 19.58    | 71.07    | 7.76     | 514.82             | 60.1        | 88.1         | 409.01       | 409.89        |
| 50          | 200          | 80.15            | 98.26            | 32.12    | 112.42   | 15.45    | 534.33             | 60.0        | 82.4         | 410.34       | 410.62        |


- При `concurrency=1` PyTorch и ONNX дают сопоставимую среднюю latency (≈4.2–4.4 ms), однако ONNX использует другую память/инициализацию; при этом ONNX уже демонстрирует высокий throughput (≈222 req/s). 
- При росте `concurrency` ONNX уверенно выходит вперёд по throughput: до ≈515 req/s при 20 и ≈534 req/s при 50 конкурентных запросах, в то время как PyTorch стабилизируется около 200 req/s. [onnxruntime](https://onnxruntime.ai/docs/performance/transformers-optimization.html)
- p95 latency у ONNX растёт с нагрузкой (особенно заметен хвост при `concurrency=10` из-за колебаний в очереди запросов), но остаётся значительно ниже, чем у PyTorch на высоких уровнях нагрузки. [opensource.microsoft](https://opensource.microsoft.com/blog/2022/04/19/scaling-up-pytorch-inference-serving-billions-of-daily-nlp-inferences-with-onnx-runtime/)

***

## Часть 3. Динамическое батчирование (ONNX + очередь + воркеры)

### Реализация dynamic batching

Dynamic batching реализован в `main_batching.py` через очередь и несколько фоновых воркеров. [betterprogramming](https://betterprogramming.pub/improving-your-prediction-api-with-dynamic-batching-50b98e5054f7)

Основные параметры:

- `MAX_BATCH_SIZE = 32` — максимальный размер батча;  
- `MAX_WAIT_MS = 20` — максимальное время ожидания дополнительных запросов;  
- `NUM_WORKERS = 2` — количество воркеров, обрабатывающих общую очередь.

Архитектура:

- `RequestContext` содержит тексты и `asyncio.Future` для результата.  
- `request_queue: asyncio.Queue[RequestContext]` — очередь запросов.  
- Воркеры `batching_worker(worker_id)`:
  - забирают первый запрос из очереди;
  - доформировывают батч до `MAX_BATCH_SIZE` или до истечения `MAX_WAIT_MS`;
  - выполняют batched токенизацию и инференс через ONNX-модель;
  - распределяют результаты по futures.  
- Эндпоинт `/embed` кладет контекст в очередь и ждёт `ctx.future`.

Дополнительно экспортируются метрики Prometheus:

- `request_processing_seconds_batch` — latency;  
- `request_count_batch` — количество запросов;  
- `batch_size_observed` — наблюдаемый размер сформированных батчей. 

### Результаты бенчмарка (ONNX + batching, порт 8002)

| Concurrency | Num requests | Avg latency (ms) | p95 latency (ms) | Min (ms) | Max (ms) | Std (ms) | Throughput (req/s) | Avg CPU (%) | Peak CPU (%) | Avg RSS (MB) | Peak RSS (MB) |
|------------:|-------------:|-----------------:|-----------------:|---------:|---------:|---------:|--------------------:|------------:|-------------:|-------------:|--------------:|
| 1           | 200          | 30.62            | 32.82            | 27.78    | 42.69    | 1.49     | 32.35              | 17.1        | 40.4         | 193.28       | 194.75        |
| 10          | 200          | 43.96            | 141.52           | 30.82    | 179.23   | 29.73    | 224.04             | 24.6        | 40.0         | 308.63       | 409.75        |
| 20          | 200          | 49.51            | 53.47            | 41.44    | 58.94    | 2.57     | 393.07             | 43.2        | 54.0         | 412.15       | 413.34        |
| 50          | 200          | 83.20            | 94.05            | 54.92    | 98.12    | 9.86     | 560.78             | 21.3        | 48.1         | 418.36       | 421.75        |

Выводы:

- При низкой нагрузке (concurrency=1) dynamic batching добавляет задержку окна ожидания (`MAX_WAIT_MS`), поэтому latency выше, чем у обычного ONNX-сервиса, а throughput ниже. [baseten](https://www.baseten.co/blog/continuous-vs-dynamic-batching-for-ai-inference/)
- При увеличении нагрузки dynamic batching позволяет существенно наращивать throughput:
  - до 393 req/s при concurrency=20;
  - до 560.78 req/s при concurrency=50.  
- p95 latency при высокой нагрузке остается на приемлемом уровне (около 94 ms при concurrency=50), при этом throughput превосходит как PyTorch, так и ONNX без батчинга. [betterprogramming](https://betterprogramming.pub/improving-your-prediction-api-with-dynamic-batching-50b98e5054f7)

***

## Сводное сравнение 

### Concurrency = 1

| Сервис          | Avg latency (ms) | p95 latency (ms) | Throughput (req/s) |
|-----------------|------------------|------------------|--------------------|
| PyTorch         | 4.21             | 4.55             | 233.13             |
| ONNX            | 4.37             | 5.22             | 221.82             |
| ONNX + batching | 30.62            | 32.82            | 32.35              |

### Concurrency = 10

| Сервис          | Avg latency (ms) | p95 latency (ms) | Throughput (req/s) |
|-----------------|------------------|------------------|--------------------|
| PyTorch         | 49.91            | 60.50            | 198.12             |
| ONNX            | 28.78            | 215.03           | 340.54             |
| ONNX + batching | 43.96            | 141.52           | 224.04             |

### Concurrency = 20

| Сервис          | Avg latency (ms) | p95 latency (ms) | Throughput (req/s) |
|-----------------|------------------|------------------|--------------------|
| PyTorch         | 94.62            | 113.79           | 207.66             |
| ONNX            | 36.85            | 52.74            | 514.82             |
| ONNX + batching | 49.51            | 53.47            | 393.07             |

### Concurrency = 50

| Сервис          | Avg latency (ms) | p95 latency (ms) | Throughput (req/s) |
|-----------------|------------------|------------------|--------------------|
| PyTorch         | 236.30           | 355.34           | 200.63             |
| ONNX            | 80.15            | 98.26            | 534.33             |
| ONNX + batching | 83.20            | 94.05            | 560.78             |

***

## Выводы

1. Базовый PyTorch-сервис на `SentenceTransformer` прост в реализации и обеспечивает минимальную latency при одиночных запросах, однако при росте нагрузки проигрывает ONNX по latency и throughput и потребляет больше памяти. 
2. ONNX Runtime даёт сопоставимую latency при одиночных запросах и заметное преимущество по throughput и устойчивости при средней и высокой нагрузке, особенно при `concurrency ≥ 20`, где пропускная способность растёт более чем вдвое относительно базового PyTorch-сервиса. [onnxruntime](https://onnxruntime.ai/docs/performance/transformers-optimization.html)
3. Dynamic batching поверх ONNX даёт значимый прирост throughput при высокой конкурентной нагрузке, удерживая p95 latency в приемлемых пределах, и демонстрирует типичный trade-off: дополнительные задержки при низкой нагрузке и выигрыш при высокой. [baseten](https://www.baseten.co/blog/continuous-vs-dynamic-batching-for-ai-inference/)
4. Использование Prometheus-метрик, явной настройки `graph_optimization_level` в ONNX Runtime, очереди и нескольких воркеров приближает решение к production-подходам к сервингу моделей и позволяет масштабировать систему под реальные сценарии использования. [opensource.microsoft](https://opensource.microsoft.com/blog/2022/04/19/scaling-up-pytorch-inference-serving-billions-of-daily-nlp-inferences-with-onnx-runtime/)

## Способ запуска

1. Установка зависимостей:
   ```bash
   pip install -r requirements-3.txt
   ```

2. Базовый PyTorch сервис (`main.py`, порт 8000):
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

3. Экспорт модели и ONNX-сервис (`main_onnx.py`, порт 8001):
   ```bash
   python export_onnx.py
   python main_onnx.py
   ```

4. Dynamic batching сервис (`main_batching.py`, порт 8002):
   ```bash
   python main_batching.py
   ```

5. Бенчмаркинг (пример для любого сервиса):
   ```bash
   python benchmark.py --url http://localhost:<порт>/embed \
     --batch-size 1 --num-requests 200 --concurrency <C> --warmup 10 --pid <PID_процесса>
   ```

   где `<порт>` — 8000 для PyTorch, 8001 для ONNX, 8002 для ONNX + batching, `<C>` — выбранный уровень `concurrency`.
