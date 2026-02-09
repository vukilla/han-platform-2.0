from functools import lru_cache

import redis

from app.core.config import get_settings


@lru_cache
def _redis_client() -> redis.Redis:
    settings = get_settings()
    return redis.Redis.from_url(settings.redis_url, decode_responses=True)


def get_queue_depth(queue_name: str) -> int:
    try:
        return int(_redis_client().llen(queue_name))
    except redis.RedisError:
        return 0


def is_queue_full(queue_name: str, max_depth: int) -> bool:
    if max_depth <= 0:
        return False
    return get_queue_depth(queue_name) >= max_depth
