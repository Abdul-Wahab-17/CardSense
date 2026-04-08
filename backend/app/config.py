from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Roles / process topology
    app_role: str = "all"  # api|reducer|all
    worker_enabled: int = 1

    redis_url: str = "redis://redis:6379/0"
    frame_stream: str = "frames"
    detection_stream: str = "detections"
    frame_consumer_group: str = "yolo_workers"
    backend_consumer_group: str = "backend_broadcaster"
    backend_consumer_name: str = "backend-1"
    worker_log_stream: str = "worker_logs"
    worker_status_prefix: str = "worker:status"
    max_queue_len: int = 5000
    static_dir: str = "frontend"
    hand_size: int = 13
    min_card_conf: float = 0.02
    hand_memory_events: int = 120
    min_card_votes: int = 1
    card_labels: str = ""
    frame_id_dedupe_ttl_sec: int = 120

    # Reducer correctness / idempotency
    reducer_lock_ttl_sec: int = 5
    reducer_lock_retry_ms: int = 20
    reducer_batch_size: int = 50
    reducer_max_event_lag_ms: int = 2000

    # Database / durability
    database_url: str = ""
    db_pool_min_size: int = 1
    db_pool_max_size: int = 10
    db_statement_timeout_ms: int = 5000
    db_connect_timeout_ms: int = 3000

    # Security
    auth_enabled: int = 1
    jwt_issuer: str = ""
    jwt_audience: str = ""
    jwt_private_key_pem: str = ""
    jwt_public_key_pem: str = ""
    jwt_hs256_secret: str = "dev-secret-change-me"
    access_token_ttl_sec: int = 1800
    refresh_token_ttl_sec: int = 2592000
    api_key_hashes: str = ""
    admin_role_name: str = "admin"
    operator_role_name: str = "operator"
    cors_allowed_origins: str = "*"

    # Privacy / governance
    store_raw_frames: int = 0
    frame_retention_days: int = 0
    event_retention_days: int = 30
    recommendation_retention_days: int = 30
    pii_redaction_enabled: int = 1

    # Observability
    metrics_enabled: int = 1
    metrics_namespace: str = "cardsense"
    trace_enabled: int = 0
    trace_sample_rate: float = 0.1
    log_level: str = "INFO"
    log_json: int = 1

    # Scaling / ingest control
    worker_autoscale_min: int = 1
    worker_autoscale_max: int = 4
    target_queue_depth_per_worker: int = 200
    frame_ingest_rate_limit_fps: float = 8.0
    ws_broadcast_batch_size: int = 64

    # Vision-state bridge / reconciliation
    hand_occlusion_ttl_ms: int = 2000
    table_occlusion_ttl_ms: int = 1200
    hand_to_table_grace_ms: int = 1200
    max_transition_gap_ms: int = 3000
    max_table_growth_per_event: int = 1

    # Deployment and release
    deploy_env: str = "dev"
    release_version: str = "dev-local"
    canary_percent: int = 0

    # Feature flags
    feature_calibrated_seat_mapping: int = 1
    feature_strict_transition_reject: int = 1
    feature_policy_simulator_v2: int = 1
    feature_belief_bayes_update: int = 1
    feature_db_event_persistence: int = 1
    feature_replay_api: int = 1
    feature_enforce_camera_binding: int = 1
    persist_recommendation_on_transition: int = 1

    # Reconciliation safety
    allow_won_pile_replay: int = 0

    # Auth hardening
    auth_strict_iss_aud: int = 1
    auth_enforce_jti: int = 1
    auth_prod_secret_required: int = 1

    # Camera readiness / stale gating
    camera_stale_threshold_ms: int = 5000

    # Multi-machine model registry
    model_provider: str = "filesystem"  # filesystem|http|s3
    model_ref_hand: str = ""
    model_ref_table: str = ""
    model_registry_url: str = ""
    model_registry_token: str = ""
    model_sha256_hand: str = ""
    model_sha256_table: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
