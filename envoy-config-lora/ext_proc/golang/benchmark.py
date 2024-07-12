import time
import random
import asyncio
import aiohttp

# Global metrics
metrics = {
    "total_tokens_generated": 0,
    "total_prompt_tokens": 0,
    "dropped_requests": 0,
    "timeout_requests": 0,
    "bad_content_type": 0,
    "os_errors": 0,
    "server_disconnected": 0,
}

# Model and IP configurations
models = [
    "tweet-summary", "sql-lora", "tweet-summary-0", "sql-lora-0",
    "tweet-summary-1", "sql-lora-1", "tweet-summary-2", "sql-lora-2",
    "tweet-summary-3", "sql-lora-3", "tweet-summary-4", "sql-lora-4",
]

model_map = {
    "34.124.136.155": models,
}

def create_json(ip: str, model: str = None) -> dict:
    if model is None:
        model = random.choice(model_map[ip])
    return {"prompt": "Is the Necronomicon in the movie: Army of Darkness?", "max_tokens": 750, "model": model}

async def parallelized_benchmarking(session: aiohttp.ClientSession, ip: str, model: str = None):
    try:
        json_data = create_json(ip, model)
        url = f"http://{ip}:80/v1/completions"
        headers = {'lora-adapter': json_data['model']}
        async with session.post(url, json=json_data, headers=headers) as response:
            response_json = await response.json()
            metrics["total_tokens_generated"] += int(response_json['usage']['completion_tokens'])
            metrics["total_prompt_tokens"] += int(response_json['usage']['prompt_tokens'])
            print(metrics["total_tokens_generated"])
    except aiohttp.client_exceptions.ClientConnectorError as client_err:
        metrics["dropped_requests"] += 1
        print(client_err)
    except asyncio.TimeoutError as timeout_err:
        metrics["timeout_requests"] += 1
        print(timeout_err)
    except aiohttp.client_exceptions.ClientOSError:
        metrics["os_errors"] += 1
    except aiohttp.client_exceptions.ContentTypeError as e:
        text = await response.text()
        print(text)
        print(e)
        metrics["bad_content_type"] += 1
    except aiohttp.client_exceptions.ServerDisconnectedError:
        metrics["server_disconnected"] += 1

def ips(n_reqs: int):
    available_ips = ["34.124.136.155"]
    for _ in range(n_reqs):
        yield random.choice(available_ips)

async def test_main(n_reqs: int, model: str = None):
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(
            *[parallelized_benchmarking(session, ip, model) for ip in ips(n_reqs)]
        )

def clear_metrics():
    global metrics
    for key in metrics:
        metrics[key] = 0

if __name__ == "__main__":
    # Warm-up phase
    for model in models:
        print(f"Warming up {model}")
        asyncio.run(test_main(1, model))
        print(f"Done warming up {model}")
        time.sleep(2)

    # Clear metrics after warm-up
    clear_metrics()

    # Main benchmarking phase
    n_reqs = 1000
    start = time.perf_counter()
    asyncio.run(test_main(n_reqs))
    end = time.perf_counter()
    
    bad_requests = sum(metrics[key] for key in ["dropped_requests", "timeout_requests", "os_errors", "bad_content_type", "server_disconnected"])
    total_complete_reqs = n_reqs - metrics["dropped_requests"]

    # Results output
    print(f"Total time: {end-start}")
    print(f"Requests per second: {total_complete_reqs / (end-start)}")
    print(f"Total generated tokens: {metrics['total_tokens_generated']}")
    print(f"Average output tokens per request: {metrics['total_tokens_generated'] / total_complete_reqs}")
    print(f"Total prompt tokens: {metrics['total_prompt_tokens']}")
    print(f"Average input tokens per request: {metrics['total_prompt_tokens'] / total_complete_reqs}")
    print(f"Output tokens per second: {metrics['total_tokens_generated'] / (end-start)}")
    print(f"Bad content type errors: {metrics['bad_content_type']}")
    print(f"Timeout requests: {metrics['timeout_requests']}")
    print(f"Dropped requests: {metrics['dropped_requests']}")
    print(f"Server disconnections: {metrics['server_disconnected']}")
    print(f"OS errors: {metrics['os_errors']}")
    print(f"Total dropped requests: {bad_requests}")
