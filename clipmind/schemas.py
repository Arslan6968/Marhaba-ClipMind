"""Universal data schemas for Marhaba ClipMind v3.
Clean, framework-independent Pydantic models ready for CLI, Agent Skill, or Web API integration.
"""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

DIMENSIONS = (
    "hook_strength",
    "content_value",
    "emotional_value",
    "standalone_clarity",
    "context_completeness",
    "audio_quality",
    "visual_suitability",
    "boundary_quality",
    "engagement_potential",
)

DEFAULT_WEIGHTS = {
    "hook_strength": 0.18,
    "content_value": 0.16,
    "emotional_value": 0.12,
    "standalone_clarity": 0.14,
    "context_completeness": 0.14,
    "audio_quality": 0.08,
    "visual_suitability": 0.06,
    "boundary_quality": 0.06,
    "engagement_potential": 0.06,
}


class UserIntent(BaseModel):
    objective: str = "best_overall"      # best_overall, funny, educational, powerful, controversial, emotional, story, discussion
    topic: Optional[str] = None          # semantic topic filter, e.g., "artificial intelligence"
    speaker: Optional[str] = None        # speaker / creator appearance focus inside video
    target_duration: str = "medium"      # short (15-35s), medium (25-60s), long (50-90s), any
    preset: Optional[str] = None         # clean, dynamic, podcast, gaming, educational, social_media
    max_clips: int = 5
    aspect_ratio: str = "9:16"           # 9:16, 16:9, 1:1, original
    caption_lang: str = "original"       # original, roman_urdu, none
    language: Optional[str] = None       # forced Whisper language code, e.g., 'ur', 'en', or None (auto)
    brain: str = "auto"                  # auto, agent, local, rules, none
    whisper_model: str = "base"          # tiny, base, small
    section: Optional[str] = None        # e.g., "1:00:00-2:30:00"
    sfx: bool = False                    # subtle whoosh/pop sound design
    normalize_audio: bool = True         # -14 LUFS broadcast loudness normalization
    fast_encode: bool = False            # try hardware h264 encoder if available


class SourceInfo(BaseModel):
    url_or_path: str
    source_type: str                     # local_file, platform_url, direct_url
    platform: str                        # youtube, twitch, vimeo, kick, direct, local
    is_supported: bool
    title: str = "Untitled Video"
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 30.0
    has_audio: bool = True
    channel_name: Optional[str] = None
    channel_url: Optional[str] = None
    view_count: Optional[int] = None
    is_live: bool = False
    live_status: Optional[str] = None
    language: Optional[str] = None
    error_message: Optional[str] = None


class ChannelVideo(BaseModel):
    video_id: str
    title: str
    url: str
    channel_name: str = ""
    duration: float = 0.0
    views: int = 0
    likes: int = 0
    upload_date: str = ""
    posted_timestamp: float = 0.0
    is_live: bool = False
    heat_score: float = 0.0
    topic_relevance: float = 1.0
    language: Optional[str] = None
    reason: str = ""


class WordToken(BaseModel):
    w: str
    s: float
    e: float
    p: float = 0.7


class TranscriptSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    uncertain: bool = False
    speaker_turn: bool = False
    words: List[WordToken] = Field(default_factory=list)


class Transcript(BaseModel):
    language: str = "en"
    language_probability: float = 1.0
    segments: List[TranscriptSegment] = Field(default_factory=list)
    words: List[WordToken] = Field(default_factory=list)
    full_text: str = ""


class ContentClassification(BaseModel):
    genre: str = "general"               # podcast, interview, lecture_education, gaming, business, commentary, general
    confidence: float = 0.5
    recommended_preset: str = "clean"
    genre_scores: Dict[str, float] = Field(default_factory=dict)
    rationale: str = ""


class EvaluationDimensions(BaseModel):
    hook_strength: float = 5.0
    content_value: float = 5.0
    emotional_value: float = 5.0
    standalone_clarity: float = 5.0
    context_completeness: float = 5.0
    audio_quality: float = 5.0
    visual_suitability: float = 5.0
    boundary_quality: float = 5.0
    engagement_potential: float = 5.0


class ClipCandidate(BaseModel):
    id: str
    start: float
    end: float
    duration: float = 0.0
    hook_time: float = 0.0
    trigger_time: Optional[float] = None
    payoff_time: Optional[float] = None
    dimensions: EvaluationDimensions
    overall_score: float = 0.0
    title: str = ""
    headline: str = ""
    hook_text: str = ""
    ab_hooks: List[str] = Field(default_factory=list)
    event_text: str = ""
    context_text: str = ""
    payoff_text: str = ""
    rationale: str = ""
    transcript_text: str = ""
    words: List[WordToken] = Field(default_factory=list)
    speaker_label: Optional[str] = None
    boundary_adjusted: bool = False
    adjustment_notes: List[str] = Field(default_factory=list)
    story_sheet_path: Optional[str] = None
    check_strip_path: Optional[str] = None
    picked_by: str = "rules"
    verified: bool = True
    rejection_reason: Optional[str] = None


class PublishingPackage(BaseModel):
    title: str
    headline: str = ""
    description: str
    caption: str
    hashtags: List[str] = Field(default_factory=list)
    category: str
    duration_sec: float
    quality_score: float
    rationale: str
    ab_hooks: List[str] = Field(default_factory=list)


class RenderedClip(BaseModel):
    clip_id: str
    output_path: str
    check_strip_path: Optional[str] = None
    story_sheet_path: Optional[str] = None
    duration: float
    preset: str
    aspect_ratio: str = "9:16"
    layout_mode: str = "fit_blur"
    dimensions: EvaluationDimensions
    publishing: PublishingPackage
    verification: Dict[str, Any] = Field(default_factory=dict)
    render_meta: Dict[str, Any] = Field(default_factory=dict)
