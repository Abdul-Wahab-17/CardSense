from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional


class FrameIn(BaseModel):
    game_id: str = Field(default="default-game", min_length=1)
    camera_id: str = Field(..., min_length=1)
    camera_type: str = Field(default="hand", pattern="^(hand|table)$")
    frame_id: str = Field(default="")
    target_model_type: Literal["hand", "table"] | None = None
    device_id: str | None = None
    session_id: str | None = None
    frame_b64: str = Field(..., min_length=10, description="Base64 JPEG/PNG frame")
    ts_ms: int = Field(..., ge=0)


class FrameBatchIn(BaseModel):
    frames: List[FrameIn] = Field(..., min_length=1, max_length=32)


class CameraRegister(BaseModel):
    camera_id: str = Field(..., min_length=1)
    source: Optional[str] = None


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    cls_id: int
    cls_name: str
    conf: float
    card_label_top1: str
    card_conf_top1: float
    card_probs_topk: List[List[float | str]] = Field(default_factory=list)
    det_conf: float
    calibrated_card_conf_top1: float = 0.0
    is_blurry: bool = False
    is_dark: bool = False
    is_overexposed: bool = False
    bbox: BBox


class DetectionEvent(BaseModel):
    game_id: str
    frame_id: str
    camera_type: Literal["hand", "table"]
    camera_id: str
    ts_ms: int
    model_name: str
    detections: List[Detection]


class HandCard(BaseModel):
    label: str
    confidence: float
    slot_id: int | None = None


class HandState(BaseModel):
    camera_id: str
    updated_ts_ms: int
    max_cards: int
    cards: List[HandCard]


class HelpTip(BaseModel):
    title: str
    detail: str


class HandAdvice(BaseModel):
    camera_id: str
    updated_ts_ms: int
    summary: str
    discard_candidates: List[str]
    tips: List[HelpTip]


class HandOverrideIn(BaseModel):
    cards: List[str] = Field(..., min_length=0, max_length=52)


class TurnState(BaseModel):
    leader: str = "me"
    to_play: str = "me"
    lead_suit: str = ""


class GameRules(BaseModel):
    variant_name: str = "default"
    seat_order: List[str] = Field(default_factory=lambda: ["me", "left", "across", "right"])
    play_direction: Literal["clockwise", "counterclockwise"] = "clockwise"
    trick_size: int = 4
    must_follow_suit: bool = True
    must_trump_if_void: bool = False
    trump_mode: str = "variant_config"
    trump_suit: str = ""
    cards_per_player: int = 13
    rank_order_map: Dict[str, int] = Field(
        default_factory=lambda: {
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "10": 10,
            "J": 11,
            "Q": 12,
            "K": 13,
            "A": 14,
        }
    )


class TableTrickCard(BaseModel):
    seat: str
    card: str
    confidence: float = 0.0


class PlayedHistoryItem(BaseModel):
    trick_no: int
    seat: str
    card: str


class UncertaintyBlock(BaseModel):
    hand_conf: float = 0.0
    table_conf: float = 0.0
    state_conf: float = 0.0
    notes: List[str] = Field(default_factory=list)


class CanonicalGameState(BaseModel):
    game_id: str
    hand_cards: List[str] = Field(default_factory=list)
    table_trick_cards: List[TableTrickCard] = Field(default_factory=list)
    played_history: List[PlayedHistoryItem] = Field(default_factory=list)
    won_piles: Dict[str, List[str]] = Field(default_factory=lambda: {"team_me": [], "team_opp": []})
    turn: TurnState = Field(default_factory=TurnState)
    rules: GameRules = Field(default_factory=GameRules)
    trick_id: int = 1
    trick_open_ts_ms: int = 0
    trick_close_ts_ms: int = 0
    trick_cards_by_seat: Dict[str, str] = Field(default_factory=dict)
    state_version: int = 0
    last_transition_id: str = ""
    uncertainty: UncertaintyBlock = Field(default_factory=UncertaintyBlock)
    updated_ts_ms: int = 0


class GameStartIn(BaseModel):
    game_id: str = Field(..., min_length=1)
    variant_name: str = "default"
    player_count: int = 4
    cards_per_player: int = 13
    seat_order: List[str] = Field(default_factory=lambda: ["me", "left", "across", "right"])
    play_direction: Literal["clockwise", "counterclockwise"] = "clockwise"
    rounds_total: int | None = None
    target_points: int | None = None
    trick_size: int = 4
    team_map: Dict[str, str] = Field(default_factory=lambda: {"me": "team_me", "across": "team_me", "left": "team_opp", "right": "team_opp"})
    leader: str = "me"
    to_play: str = "me"
    must_follow_suit: bool = True
    must_trump_if_void: bool = False
    trump_mode: str = "variant_config"
    trump_suit: str = ""
    rank_order_map: Dict[str, int] | None = None
    deck_cards: List[str] | None = None


class GameConfigIn(BaseModel):
    variant_name: Optional[str] = None
    seat_order: Optional[List[str]] = None
    play_direction: Optional[Literal["clockwise", "counterclockwise"]] = None
    trick_size: Optional[int] = None
    leader: Optional[str] = None
    to_play: Optional[str] = None
    lead_suit: Optional[str] = None
    must_follow_suit: Optional[bool] = None
    must_trump_if_void: Optional[bool] = None
    trump_mode: Optional[str] = None
    trump_suit: Optional[str] = None
    rank_order_map: Optional[Dict[str, int]] = None


class ActionScore(BaseModel):
    card: str
    rule_validity_score: float = 0.0
    state_stability_score: float = 0.0
    info_value_score: float = 0.0
    p_win_current_trick: float
    expected_future_trick_value: float
    partner_support_value: float
    risk: float
    ev_now: float = 0.0
    ev_future: float = 0.0
    ev_partner_synergy: float = 0.0
    ev_information_gain: float = 0.0
    total_ev: float


class AdvisorBestAction(BaseModel):
    card: str
    ev: float
    confidence: float


class AdvisorAlternative(BaseModel):
    card: str
    ev: float


class AdvisorResponse(BaseModel):
    best_action: AdvisorBestAction
    alternatives: List[AdvisorAlternative]
    why: List[str]
    uncertainty_notes: List[str]
    recommendation_mode: Literal["normal", "cautious"]
    freshness_ts_ms: int
    action_scores: List[ActionScore] = Field(default_factory=list)


class TableCalibrationIn(BaseModel):
    table_homography: List[List[float]] = Field(default_factory=list)
    seat_regions: Dict[str, List[List[float]]] = Field(default_factory=dict)
    trick_region: List[List[float]] = Field(default_factory=list)
    won_pile_regions: Dict[str, List[List[float]]] = Field(default_factory=dict)
    occlusion_timeout_ms: int = 2000


class VariantConfig(BaseModel):
    variant_id: str = "default"
    player_count: int = 4
    seat_order: List[str] = Field(default_factory=lambda: ["me", "left", "across", "right"])
    play_direction: Literal["clockwise", "counterclockwise"] = "clockwise"
    team_mode: Literal["teams", "solo"] = "teams"
    team_map: Dict[str, str] = Field(default_factory=lambda: {"me": "team_me", "across": "team_me", "left": "team_opp", "right": "team_opp"})
    cards_per_player: int = 13
    trick_size: int = 4
    rounds_total: int | None = None
    target_points: int | None = None
    must_follow_suit: bool = True
    must_trump_if_void: bool = False
    trump_policy: Literal["none", "fixed", "round_declared"] = "fixed"
    trump_suit: str = ""
    rank_order_map: Dict[str, int] = Field(default_factory=lambda: GameRules().rank_order_map)
    deck_cards: List[str] = Field(default_factory=list)


class CardLifecycleEntry(BaseModel):
    location: Literal["in_hand", "on_table", "won_pile", "unknown"] = "unknown"
    owner_seat: str | None = None
    owner_team: str | None = None
    seen_ts_ms: int = 0
    last_transition: str = ""
    confidence: float = 0.0


class MatchState(BaseModel):
    game_id: str
    phase: Literal["setup", "deal", "in_round", "round_end", "match_end"] = "setup"
    round_index: int = 0
    dealer_seat: str = "me"
    leader_seat: str = "me"
    to_play_seat: str = "me"
    lead_suit: str = ""
    trick_id: int = 1
    table_cards: List[TableTrickCard] = Field(default_factory=list)
    played_history: List[PlayedHistoryItem] = Field(default_factory=list)
    team_points: Dict[str, int] = Field(default_factory=dict)
    team_tricks: Dict[str, int] = Field(default_factory=dict)
    round_winner_team: str | None = None
    match_winner_team: str | None = None
    variant: VariantConfig = Field(default_factory=VariantConfig)
    card_lifecycle: Dict[str, CardLifecycleEntry] = Field(default_factory=dict)
    hand_cards_by_seat: Dict[str, List[str]] = Field(default_factory=dict)
    rule_violations_recent: List[str] = Field(default_factory=list)
    invariant_violations_recent: List[str] = Field(default_factory=list)
    card_duplicate_rejected_total: int = 0
    hand_count_violation_total: int = 0


class CameraBindIn(BaseModel):
    camera_id: str
    camera_type: Literal["hand", "table"]
    seat_hint: str | None = None
    device_label: str = ""
    stream_profile: Literal["low_latency", "balanced", "high_accuracy"] = "balanced"


class CameraSessionState(BaseModel):
    game_id: str
    camera_id: str
    camera_type: Literal["hand", "table"]
    seat_hint: str | None = None
    device_label: str = ""
    stream_profile: Literal["low_latency", "balanced", "high_accuracy"] = "balanced"
    camera_online: bool = False
    last_frame_ts_ms: int = 0
    frame_rate_actual: float = 0.0
    latency_ms_p50: float = 0.0
    latency_ms_p95: float = 0.0
    drop_rate: float = 0.0


class AuthRegisterIn(BaseModel):
    email: str
    password: str = Field(min_length=8)
    display_name: str = ""


class AuthLoginIn(BaseModel):
    email: str
    password: str


class AuthRefreshIn(BaseModel):
    refresh_token: str


class AuthLogoutIn(BaseModel):
    refresh_token: str


class VariantSetIn(BaseModel):
    variant: VariantConfig


class RoundStartIn(BaseModel):
    round_index: int
    dealer_seat: str
