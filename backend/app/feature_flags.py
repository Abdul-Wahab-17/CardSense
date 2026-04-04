from __future__ import annotations

from .config import settings


FEATURE_FLAGS_METADATA = {
    "FEATURE_CALIBRATED_SEAT_MAPPING": {
        "default": bool(settings.feature_calibrated_seat_mapping),
        "owner": "vision-team",
        "removal_date": "2026-07-31",
        "rollback_behavior": "Fallback to x-order seat mapping",
    },
    "FEATURE_STRICT_TRANSITION_REJECT": {
        "default": bool(settings.feature_strict_transition_reject),
        "owner": "state-engine-team",
        "removal_date": "2026-07-31",
        "rollback_behavior": "Disable strict reject threshold and accept soft transitions",
    },
    "FEATURE_POLICY_SIMULATOR_V2": {
        "default": bool(settings.feature_policy_simulator_v2),
        "owner": "decision-team",
        "removal_date": "2026-08-15",
        "rollback_behavior": "Use legacy simulator weights/policies",
    },
    "FEATURE_BELIEF_BAYES_UPDATE": {
        "default": bool(settings.feature_belief_bayes_update),
        "owner": "belief-team",
        "removal_date": "2026-08-15",
        "rollback_behavior": "Use uniform-only belief baseline",
    },
    "FEATURE_DB_EVENT_PERSISTENCE": {
        "default": bool(settings.feature_db_event_persistence),
        "owner": "platform-team",
        "removal_date": "2026-09-01",
        "rollback_behavior": "Keep in-memory event log and sqlite fallback",
    },
    "FEATURE_REPLAY_API": {
        "default": bool(settings.feature_replay_api),
        "owner": "platform-team",
        "removal_date": "2026-09-01",
        "rollback_behavior": "Disable replay endpoint and broadcast path",
    },
}
