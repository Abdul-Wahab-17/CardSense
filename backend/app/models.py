from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional


class FrameIn(BaseModel):
    game_id: str = Field(default="default-game", min_length=1)
    camera_id: str = Field(..., min_length=1)
    camera_type: str = Field(default="hand", pattern="^(hand|table)$")
    frame_id: str = Field(default="")
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
    cards: List[str] = Field(..., min_length=0, max_length=13)


class TurnState(BaseModel):
    leader: str = "me"
    to_play: str = "me"
    lead_suit: str = ""


class GameRules(BaseModel):
    variant_name: str = "default"
    must_follow_suit: bool = True
    must_trump_if_void: bool = False
    trump_mode: str = "variant_config"
    trump_suit: str = ""
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
    leader: str = "me"
    to_play: str = "me"
    must_follow_suit: bool = True
    must_trump_if_void: bool = False
    trump_mode: str = "variant_config"
    trump_suit: str = ""
    rank_order_map: Dict[str, int] | None = None


class GameConfigIn(BaseModel):
    variant_name: Optional[str] = None
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
