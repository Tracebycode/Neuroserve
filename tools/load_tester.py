#!/usr/bin/env python3
import time
import json
import argparse
import urllib.request
import urllib.error
import concurrent.futures
import statistics

def check_server(base_url):
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=3) as resp:
            data = json.loads(resp.read().decode())
            print(f"[Health Check] /health: {data}")
        with urllib.request.urlopen(f"{base_url}/ready", timeout=3) as resp:
            data = json.loads(resp.read().decode())
            print(f"[Readiness Check] /ready: {data}")
        return True
    except Exception as e:
        print(f"[Error] Could not connect to server at {base_url}: {e}")
        return False

def send_infer_request(base_url, model_name, request_id):
    payload = {
        "model_name": model_name,
        "inputs": [
            {
                "name": "input",
                "shape": [1, 8],
                "data": [0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.8]
            }
        ]
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        f"{base_url}/v1/infer",
        data=data,
        headers={"Content-Type": "application/json"}
    )

    t0 = time.perf_counter()
    status_code = 0
    success = False
    queue_ms = 0.0
    infer_ms = 0.0

    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            t1 = time.perf_counter()
            status_code = resp.status
            body = json.loads(resp.read().decode('utf-8'))
            success = body.get("success", False)
            latency_obj = body.get("latency_ms", {})
            queue_ms = latency_obj.get("queue", 0.0)
            infer_ms = latency_obj.get("inference", 0.0)
    except urllib.error.HTTPError as he:
        t1 = time.perf_counter()
        status_code = he.code
    except Exception:
        t1 = time.perf_counter()
        status_code = 0

    roundtrip_ms = (t1 - t0) * 1000.0
    return {
        "status": status_code,
        "success": success,
        "roundtrip_ms": roundtrip_ms,
        "queue_ms": queue_ms,
        "infer_ms": infer_ms
    }

def run_load_test(base_url, model_name, total_requests, concurrency):
    print(f"\n--- Running Load Test: {total_requests} requests @ concurrency={concurrency} ---")
    start_time = time.perf_counter()

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_infer_request, base_url, model_name, i)
            for i in range(total_requests)
        ]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    elapsed_s = time.perf_counter() - start_time
    throughput = len(results) / elapsed_s if elapsed_s > 0 else 0

    latencies = [r["roundtrip_ms"] for r in results]
    latencies.sort()

    successes = sum(1 for r in results if r["status"] == 200 and r["success"])
    shed = sum(1 for r in results if r["status"] == 429)
    errors = sum(1 for r in results if r["status"] not in (200, 429))

    p50 = latencies[int(0.50 * len(latencies))] if latencies else 0
    p95 = latencies[int(0.95 * len(latencies))] if latencies else 0
    p99 = latencies[int(0.99 * len(latencies))] if latencies else 0
    mean_lat = statistics.mean(latencies) if latencies else 0

    print(f"Total Time:      {elapsed_s:.3f} s")
    print(f"Throughput:      {throughput:.1f} req/s")
    print(f"Success:         {successes}/{len(results)} ({successes/len(results)*100:.1f}%)")
    print(f"Load Shed (429): {shed}")
    print(f"Errors:          {errors}")
    print(f"Latency Mean:    {mean_lat:.2f} ms")
    print(f"Latency p50:     {p50:.2f} ms")
    print(f"Latency p95:     {p95:.2f} ms")
    print(f"Latency p99:     {p99:.2f} ms")

    return {
        "concurrency": concurrency,
        "throughput": throughput,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "shed": shed
    }

def main():
    parser = argparse.ArgumentParser(description="NeuroServe Benchmark Load Tester")
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="Base URL of NeuroServe")
    parser.add_argument("--model", default="demo_mlp", help="Model name")
    parser.add_argument("--requests", type=int, default=500, help="Requests per concurrency step")
    parser.add_argument("--sweep", action="store_true", help="Sweep across multiple concurrency levels")
    args = parser.parse_args()

    if not check_server(args.url):
        return

    if args.sweep:
        concurrencies = [1, 2, 4, 8, 16, 32, 64]
        summary = []
        for c in concurrencies:
            res = run_load_test(args.url, args.model, args.requests, c)
            summary.append(res)
            time.sleep(1)

        print("\n========================= CONCURRENCY SWEEP SUMMARY =========================")
        print(f"{'Concurrency':<12} | {'Throughput (req/s)':<20} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | {'p99 (ms)':<10} | {'Shed (429)':<10}")
        print("-" * 80)
        for s in summary:
            print(f"{s['concurrency']:<12} | {s['throughput']:<20.1f} | {s['p50']:<10.2f} | {s['p95']:<10.2f} | {s['p99']:<10.2f} | {s['shed']:<10}")
    else:
        run_load_test(args.url, args.model, args.requests, 8)

if __name__ == '__main__':
    main()

