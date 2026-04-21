import json
import os
import hashlib
import random
import threading
import time
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFENSE_LAYER_DIR = ROOT_DIR / "defense_layer"
if str(DEFENSE_LAYER_DIR) not in sys.path:
    sys.path.insert(0, str(DEFENSE_LAYER_DIR))

from adaptive_defense_framework.schema import DialogueSample, Turn


DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "api_config.json"
DEFAULT_EXAMPLE_CONFIG_PATH = ROOT_DIR / "configs" / "api_config.example.json"
DEFAULT_CACHE_DIR = ROOT_DIR / "results" / "online_target_cache"


def add_online_target_args(parser, default_model: str = "gpt-3.5-turbo"):
    parser.add_argument("--online-target", action="store_true", help="启用在线 target_model 回放模式")
    parser.add_argument("--config-path", type=str, default=str(DEFAULT_CONFIG_PATH), help="在线 API 配置文件")
    parser.add_argument("--base-url", type=str, default="", help="OpenAI 鍏煎鎺ュ彛 base_url")
    parser.add_argument("--api-key", type=str, default="", help="OpenAI 鍏煎鎺ュ彛 api_key")
    parser.add_argument("--model", type=str, default=default_model, help="在线 target_model 名称")
    parser.add_argument("--timeout", type=int, default=300, help="单次在线请求超时秒数")
    parser.add_argument("--temperature", type=float, default=0.0, help="在线 target_model temperature")
    parser.add_argument("--max-retries", type=int, default=5, help="在线请求最大重试次数")
    parser.add_argument("--min-request-interval", type=float, default=0.1, help="相邻在线请求最小间隔秒数")
    return parser


def load_api_config(config_path: Path) -> Dict[str, str]:
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8-sig"))
    if DEFAULT_EXAMPLE_CONFIG_PATH.exists():
        return json.loads(DEFAULT_EXAMPLE_CONFIG_PATH.read_text(encoding="utf-8-sig"))
    return {}


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int = 120,
        temperature: float = 0.0,
        max_retries: int = 5,
        min_request_interval: float = 0.1,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_retries = max_retries
        self.min_request_interval = max(0.0, float(min_request_interval))
        self.cache_dir = DEFAULT_CACHE_DIR / self.model.replace("/", "_")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._request_lock = threading.Lock()
        self._next_request_time = 0.0

    def _cache_path(self, messages: List[Dict[str, str]]) -> Path:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.txt"

    def _post_json(self, url: str, payload: dict) -> dict:
        """POST using requests library for reliable redirect and SSL handling."""
        import requests as _requests
        self._throttle()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Connection": "close",
        }
        resp = _requests.post(url, json=payload, headers=headers, timeout=self.timeout, allow_redirects=True, proxies={"http": None, "https": None})
        if resp.status_code == 400:
            lowered = resp.text.lower()
            if "content_filter" in lowered or "content management policy" in lowered:
                return {"choices": [{"finish_reason": "content_filter", "message": {"content": ""}}]}
            raise urllib.error.HTTPError(url, resp.status_code, resp.text, {}, None)
        if resp.status_code == 429 or resp.status_code >= 500:
            raise urllib.error.HTTPError(url, resp.status_code, resp.text, {}, None)
        if resp.status_code != 200:
            raise urllib.error.HTTPError(url, resp.status_code, resp.text, {}, None)
        return resp.json()

    def _throttle(self):
        if self.min_request_interval <= 0:
            return
        with self._request_lock:
            now = time.monotonic()
            wait = max(0.0, self._next_request_time - now)
            self._next_request_time = max(self._next_request_time, now) + self.min_request_interval
        if wait > 0:
            time.sleep(wait)

    def _fallback_response(self) -> str:
        return "I can't help with that request."

    def _retry_sleep(self, attempt: int):
        base = min(2 ** (attempt - 1), 16)
        jitter = random.uniform(0.1, 0.8)
        time.sleep(base + jitter)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        cache_path = self._cache_path(messages)
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                body = self._post_json(url, payload)
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore") if exc.fp else str(exc)
                lowered = detail.lower()
                if exc.code == 400 and ("content_filter" in lowered or "content management policy" in lowered):
                    content = "I can't help with that request."
                    cache_path.write_text(content, encoding="utf-8")
                    return content
                retryable = exc.code >= 500 or exc.code == 429
                last_error = RuntimeError(f"在线请求失败: HTTP {exc.code} {detail}")
                if not retryable or attempt >= self.max_retries:
                    if retryable:
                        content = self._fallback_response()
                        cache_path.write_text(content, encoding="utf-8")
                        return content
                    raise last_error from exc
            except Exception as exc:
                last_error = RuntimeError(f"在线请求失败: {exc}")
                if attempt >= self.max_retries:
                    content = self._fallback_response()
                    cache_path.write_text(content, encoding="utf-8")
                    return content
            self._retry_sleep(attempt)

        choices = body.get("choices") or []
        if choices:
            finish_reason = choices[0].get("finish_reason", "")
            message = choices[0].get("message") or {}
            if finish_reason == "content_filter" and not message.get("content"):
                content = "I can't help with that request."
                cache_path.write_text(content, encoding="utf-8")
                return content

        try:
            content = body["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"鎺ュ彛杩斿洖鏍煎紡寮傚父: {body}") from exc
        cache_path.write_text(content, encoding="utf-8")
        return content


class RoundRobinClient:
    """将多个 OpenAICompatibleClient 以线程本地方式分配——每个线程固定使用同一个后端，避免连接切换开销。"""

    def __init__(self, clients: List[OpenAICompatibleClient]):
        self._clients = clients
        self._counter = 0
        self._lock = threading.Lock()
        self._local = threading.local()

    @property
    def model(self) -> str:
        return self._clients[0].model

    @property
    def base_url(self) -> str:
        return ",".join(c.base_url for c in self._clients)

    def _get_client(self) -> OpenAICompatibleClient:
        if not hasattr(self._local, "client"):
            with self._lock:
                idx = self._counter % len(self._clients)
                self._counter += 1
            self._local.client = self._clients[idx]
        return self._local.client

    def chat(self, messages: List[Dict[str, str]]) -> str:
        return self._get_client().chat(messages)


def build_online_client(args) -> Optional["OpenAICompatibleClient | RoundRobinClient"]:
    if not getattr(args, "online_target", False):
        return None
    config = load_api_config(Path(args.config_path))
    base_url_str = args.base_url or os.getenv("OPENAI_BASE_URL") or config.get("base_url", "")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or config.get("api_key", "")
    model = args.model or os.getenv("OPENAI_MODEL") or config.get("model", "gpt-3.5-turbo")
    if not base_url_str or not api_key or api_key == "YOUR_API_KEY":
        raise RuntimeError(
            "未找到可用的在线接口配置。请通过 --base-url/--api-key 传入，"
            "或在 configs/api_config.json 中填写真实配置，"
            "或设置 OPENAI_BASE_URL / OPENAI_API_KEY。"
        )
    base_urls = [u.strip() for u in base_url_str.split(",") if u.strip()]
    clients = [
        OpenAICompatibleClient(
            base_url=url,
            api_key=api_key,
            model=model,
            timeout=args.timeout,
            temperature=args.temperature,
            max_retries=args.max_retries,
            min_request_interval=args.min_request_interval,
        )
        for url in base_urls
    ]
    if len(clients) == 1:
        return clients[0]
    return RoundRobinClient(clients)


def to_online_replay_source(sample: DialogueSample, attacker_model: str = "dataset_replay") -> DialogueSample:
    user_turns = [Turn(role="user", content=t.content) for t in sample.turns if t.role == "user" and t.content.strip()]
    meta = dict(sample.meta)
    meta["attacker_model"] = attacker_model
    meta["target_model"] = meta.get("target_model", "")
    meta["reference_turns"] = [{"role": t.role, "content": t.content} for t in sample.turns]
    return DialogueSample(
        sample_id=sample.sample_id,
        source=sample.source,
        label=sample.label,
        turns=user_turns,
        meta=meta,
    )


MAX_CONTEXT_TURNS = 6  # 发给 API 的最大历史轮数（user+assistant 对数），避免长对话触发服务端 524


def replay_with_online_target(sample: DialogueSample, client: OpenAICompatibleClient) -> DialogueSample:
    messages: List[Dict[str, str]] = []
    replay_turns: List[Turn] = []
    for turn in sample.turns:
        if turn.role != "user":
            continue
        user_text = turn.content
        messages.append({"role": "user", "content": user_text})
        # 只保留最近 MAX_CONTEXT_TURNS 轮上下文，避免超长 context 触发服务端超时
        context_to_send = messages[-(MAX_CONTEXT_TURNS * 2):]
        assistant_text = client.chat(context_to_send)
        replay_turns.append(Turn(role="user", content=user_text))
        replay_turns.append(Turn(role="assistant", content=assistant_text))
        messages.append({"role": "assistant", "content": assistant_text})

    meta = dict(sample.meta)
    meta["attacker_model"] = meta.get("attacker_model", "dataset_replay")
    meta["target_model"] = client.model
    meta["base_url"] = client.base_url
    return DialogueSample(
        sample_id=sample.sample_id,
        source=sample.source,
        label=sample.label,
        turns=replay_turns,
        meta=meta,
    )
