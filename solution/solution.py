import os
import time
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv()

PRICING_PER_1K_TOKENS = {
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}

OPENAI_MODEL = os.getenv("LAB_MODEL", "gpt-4o")
OPENAI_MINI_MODEL = os.getenv("LAB_MINI_MODEL", "gpt-4o-mini")


def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start
    return response.choices[0].message.content or "", float(latency)


def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    return call_openai(
        prompt,
        model=OPENAI_MINI_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


def compare_models(prompt: str) -> dict:
    gpt4o_response, gpt4o_latency = call_openai(prompt)
    mini_response, mini_latency = call_openai_mini(prompt)
    gpt4o_cost_estimate = (
        (len(gpt4o_response.split()) / 0.75) / 1000
        * PRICING_PER_1K_TOKENS["gpt-4o"]["output"]
    )
    return {
        "gpt4o_response": gpt4o_response,
        "mini_response": mini_response,
        "gpt4o_latency": gpt4o_latency,
        "mini_latency": mini_latency,
        "gpt4o_cost_estimate": gpt4o_cost_estimate,
    }


def chat_with_system_prompt(
    system_prompt: str,
    user_prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> tuple[str, float]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start
    return response.choices[0].message.content or "", float(latency)


def count_tokens(text: str, model: str = OPENAI_MODEL) -> int:
    try:
        import tiktoken

        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def estimate_cost(prompt: str, response: str, model: str = OPENAI_MODEL) -> dict:
    input_tokens = count_tokens(prompt, model)
    output_tokens = count_tokens(response, model)
    pricing = PRICING_PER_1K_TOKENS.get(
        model, PRICING_PER_1K_TOKENS["gpt-4o"]
    )
    input_cost = input_tokens / 1000 * pricing["input"]
    output_cost = output_tokens / 1000 * pricing["output"]
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


def streaming_chatbot() -> None:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    history = []
    while True:
        user_msg = input("Bạn: ")
        if user_msg.strip().lower() in ("quit", "exit"):
            break

        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=history + [{"role": "user", "content": user_msg}],
            stream=True,
        )
        reply_parts = []
        print("Trợ lý: ", end="", flush=True)
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            reply_parts.append(delta)
        print()

        history.extend(
            [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": "".join(reply_parts)},
            ]
        )
        history = history[-6:]


def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * (2 ** attempt))


def run_assistant(
    persona: str,
    get_input: Callable[[], str] = None,
    max_turns: int = None,
) -> dict:
    if get_input is None:
        get_input = input

    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    history = []
    num_turns = 0
    total_tokens = 0
    total_cost = 0.0

    while True:
        if max_turns is not None and num_turns >= max_turns:
            break
        user_msg = get_input()
        if user_msg.strip().lower() in ("quit", "exit"):
            break

        messages = (
            [{"role": "system", "content": persona}]
            + history
            + [{"role": "user", "content": user_msg}]
        )
        stream = retry_with_backoff(
            lambda: client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                stream=True,
            )
        )

        reply_parts = []
        print("Trợ lý: ", end="", flush=True)
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            reply_parts.append(delta)
        print()

        reply = "".join(reply_parts)
        history.extend(
            [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": reply},
            ]
        )
        history = history[-6:]
        num_turns += 1
        total_tokens += count_tokens(user_msg) + count_tokens(reply)
        total_cost += estimate_cost(user_msg, reply)["total_cost"]

    return {
        "num_turns": num_turns,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "history": history,
    }


def batch_compare(prompts: list[str]) -> list[dict]:
    return [{"prompt": prompt, **compare_models(prompt)} for prompt in prompts]


def format_comparison_table(results: list[dict]) -> str:
    headers = [
        "Prompt",
        "GPT-4o Response",
        "Mini Response",
        "GPT-4o Latency",
        "Mini Latency",
    ]

    def shorten(value: Any, limit: int = 40) -> str:
        text = str(value).replace("\n", " ").strip()
        return text if len(text) <= limit else text[: limit - 3] + "..."

    rows = [headers]
    for result in results:
        rows.append(
            [
                shorten(result.get("prompt", "")),
                shorten(result.get("gpt4o_response", "")),
                shorten(result.get("mini_response", "")),
                f"{result.get('gpt4o_latency', 0.0):.3f}s",
                f"{result.get('mini_latency', 0.0):.3f}s",
            ]
        )

    widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]
    lines = [
        " | ".join(value.ljust(widths[index]) for index, value in enumerate(rows[0]))
    ]
    lines.append("-+-".join("-" * width for width in widths))
    lines.extend(
        " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows[1:]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    print("=== So sánh model ===")
    comparison = compare_models(
        "Giải thích sự khác nhau giữa temperature và top_p trong một câu."
    )
    for key, value in comparison.items():
        print(f"{key}: {value}")

    print("\n=== Trợ lý CLI (gõ 'quit' để thoát) ===")
    stats = run_assistant(
        persona=(
            "Bạn là trợ giảng thân thiện của khóa AI, "
            "trả lời ngắn gọn bằng tiếng Việt."
        )
    )
    print("\n--- Thống kê phiên chat ---")
    for key, value in stats.items():
        if key != "history":
            print(f"{key}: {value}")
