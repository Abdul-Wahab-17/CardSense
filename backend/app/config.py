from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    redis_url: str = "redis://redis:6379/0"
    frame_stream: str = "frames"
    detection_stream: str = "detections"
    frame_consumer_group: str = "yolo_workers"
    backend_consumer_group: str = "backend_broadcaster"
    backend_consumer_name: str = "backend-1"
    max_queue_len: int = 5000
    static_dir: str = "frontend"
    hand_size: int = 13
    min_card_conf: float = 0.35
    hand_memory_events: int = 120
    card_labels: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
