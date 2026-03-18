import time
import statistics
import psutil
import requests
import concurrent.futures
import json
import threading
import argparse
import sys
from tqdm import tqdm


URL = "http://localhost:8000/embed"
HEADERS = {"Content-Type": "application/json"}
TEST_TEXTS = ["Это тестовый запрос для измерения скорости инференса."]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark for embedding service")
    parser.add_argument("--url", type=str, default=URL, help="Service endpoint URL")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of texts per request")
    parser.add_argument("--num-requests", type=int, default=100, help="Total number of requests to send")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent workers")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warm-up requests")
    parser.add_argument("--pid", type=int, default=None, help="PID of the server process to monitor")
    return parser.parse_args()


def one_request(url, payload):
    start = time.perf_counter()
    resp = requests.post(url, headers=HEADERS, json=payload)
    resp.raise_for_status()
    end = time.perf_counter()
    return end - start, resp.json()


def monitor_resources(stop_event, cpu_samples, mem_samples, pid=None):
    if pid is None:
        proc = psutil.Process()
    else:
        proc = psutil.Process(pid)

    while not stop_event.is_set():
        cpu_samples.append(psutil.cpu_percent(interval=0))
        mem_samples.append(proc.memory_info().rss)
        time.sleep(0.1)


def run_benchmark(args):
    texts = TEST_TEXTS * args.batch_size
    payload = {"texts": texts}

    for _ in tqdm(range(args.warmup)):
        _ = one_request(args.url, payload)
        time.sleep(0.01)

    stop_event = threading.Event()
    cpu_samples = []
    mem_samples = []
    monitor_thread = threading.Thread(
        target=monitor_resources,
        args=(stop_event, cpu_samples, mem_samples, args.pid),
        daemon=True
    )

    monitor_thread.start()
    
    latencies = []
    start_total = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(one_request, args.url, payload) for _ in range(args.num_requests)]
        
        for f in concurrent.futures.as_completed(futures):
            lat, _ = f.result()
            latencies.append(lat)

    end_total = time.perf_counter()
    stop_event.set()
    monitor_thread.join()
    
    total_time = end_total - start_total
    throughput = args.num_requests / total_time

    lat_ms = [l * 1000 for l in latencies]
    avg_lat = statistics.mean(lat_ms)
    p95_lat = statistics.quantiles(lat_ms, n=20)[18]  # 95-й перцентиль
    min_lat = min(lat_ms)
    max_lat = max(lat_ms)
    std_lat = statistics.stdev(lat_ms) if len(lat_ms) > 1 else 0.0

    if cpu_samples:
        avg_cpu = sum(cpu_samples) / len(cpu_samples)
        max_cpu = max(cpu_samples)
    else:
        avg_cpu = max_cpu = 0.0

    if mem_samples:
        avg_mem_mb = (sum(mem_samples) / len(mem_samples)) / (1024 * 1024)
        max_mem_mb = max(mem_samples) / (1024 * 1024)
    else:
        avg_mem_mb = max_mem_mb = 0.0

    print("\n=== BENCHMARK RESULTS ===")
    print(f"Batch size:          {args.batch_size}")
    print(f"Number of requests:  {args.num_requests}")
    print(f"Concurrency:         {args.concurrency}")
    print(f"Total time:          {total_time:.2f} s")
    print("--- Latency ---")
    print(f"Average:             {avg_lat:.2f} ms")
    print(f"95th percentile:     {p95_lat:.2f} ms")
    print(f"Min:                 {min_lat:.2f} ms")
    print(f"Max:                 {max_lat:.2f} ms")
    print(f"Std dev:             {std_lat:.2f} ms")
    print("--- Throughput ---")
    print(f"Requests/sec:        {throughput:.2f}")
    print("--- CPU ---")
    print(f"Average CPU usage:   {avg_cpu:.1f} %")
    print(f"Peak CPU usage:      {max_cpu:.1f} %")
    print("--- Memory (RSS) ---")
    print(f"Average memory:      {avg_mem_mb:.2f} MB")
    print(f"Peak memory:         {max_mem_mb:.2f} MB")
    

if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)