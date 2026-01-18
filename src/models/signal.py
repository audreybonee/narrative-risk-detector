"""
NarrativeSignal model for the Emergent Narrative Detection System.

Represents a detected signal indicating noteworthy narrative activity.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum

from .enums import SignalSeverity, NarrativeStage, PatternType


class SignalType(str, Enum):
    """Types of narrative signals that can be detected."""

    # Emergence signals
    RAPID_EMERGENCE = "rapid_emergence"          # Fast initial spread
    CROSS_SPECTRUM_SPREAD = "cross_spectrum"     # Story crossing political boundaries
    WIRE_CASCADE = "wire_cascade"                # Wire story triggering broad pickup

    # Pattern signals
    PR_AMPLIFICATION = "pr_amplification"        # PR-sourced narrative gaining traction
    FRAME_CONVERGENCE = "frame_convergence"      # Multiple outlets converging on framing
    SYNCHRONIZED_FRAMING = "synchronized"        # Potential coordinated messaging

    # Velocity signals
    VELOCITY_SPIKE = "velocity_spike"            # Sudden increase in coverage
    SUSTAINED_MOMENTUM = "sustained_momentum"    # Continued high velocity

    # Lifecycle signals
    STAGE_TRANSITION = "stage_transition"        # Narrative moving to new stage
    NARRATIVE_REVIVAL = "narrative_revival"      # Dormant narrative reactivating


class NarrativeSignal(BaseModel):
    """
    A detected signal indicating noteworthy narrative activity.

    Signals are generated when the system detects patterns worth monitoring,
    such as rapid emergence, cross-spectrum spread, or potential coordination.
    """

    id: str = Field(..., description="Unique signal identifier")
    cluster_id: str = Field(..., description="Associated narrative cluster")
    signal_type: SignalType = Field(..., description="Type of signal detected")
    severity: SignalSeverity = Field(..., description="Signal severity level")

    # Signal details
    title: str = Field(..., description="Brief signal description")
    description: str = Field(..., description="Detailed explanation")
    detected_at: datetime = Field(default_factory=datetime.utcnow)

    # Context
    headline: str = Field(..., description="Representative headline from cluster")
    topic: Optional[str] = Field(default=None, description="Narrative topic")

    # Metrics at time of detection
    cluster_size: int = Field(..., description="Number of articles in cluster")
    unique_outlets: int = Field(..., description="Number of unique outlets")
    source_diversity: int = Field(..., description="Number of outlet types")
    velocity: float = Field(..., description="Articles per hour")
    bias_spread: float = Field(..., description="Std dev of outlet bias")

    # Pattern context
    detected_pattern: Optional[PatternType] = Field(default=None)
    narrative_stage: NarrativeStage = Field(...)

    # Optional frame information
    frame_text: Optional[str] = Field(default=None, description="Detected frame if applicable")

    # Analyst workflow
    reviewed: bool = Field(default=False)
    analyst_notes: Optional[str] = Field(default=None)
    false_positive: Optional[bool] = Field(default=None)

    class Config:
        use_enum_values = False

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "cluster_id": self.cluster_id,
            "signal_type": self.signal_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "detected_at": self.detected_at.isoformat(),
            "headline": self.headline,
            "topic": self.topic,
            "metrics": {
                "cluster_size": self.cluster_size,
                "unique_outlets": self.unique_outlets,
                "source_diversity": self.source_diversity,
                "velocity": round(self.velocity, 2),
                "bias_spread": round(self.bias_spread, 3),
            },
            "detected_pattern": self.detected_pattern.value if self.detected_pattern else None,
            "narrative_stage": self.narrative_stage.value,
            "frame_text": self.frame_text,
            "reviewed": self.reviewed,
            "analyst_notes": self.analyst_notes,
            "false_positive": self.false_positive,
        }


def create_signal(
    cluster_id: str,
    signal_type: SignalType,
    severity: SignalSeverity,
    title: str,
    description: str,
    headline: str,
    cluster_size: int,
    unique_outlets: int,
    source_diversity: int,
    velocity: float,
    bias_spread: float,
    narrative_stage: NarrativeStage,
    topic: Optional[str] = None,
    detected_pattern: Optional[PatternType] = None,
    frame_text: Optional[str] = None,
) -> NarrativeSignal:
    """Factory function to create a NarrativeSignal with auto-generated ID."""
    import uuid

    signal_id = f"sig_{signal_type.value}_{uuid.uuid4().hex[:8]}"

    return NarrativeSignal(
        id=signal_id,
        cluster_id=cluster_id,
        signal_type=signal_type,
        severity=severity,
        title=title,
        description=description,
        headline=headline,
        topic=topic,
        cluster_size=cluster_size,
        unique_outlets=unique_outlets,
        source_diversity=source_diversity,
        velocity=velocity,
        bias_spread=bias_spread,
        detected_pattern=detected_pattern,
        narrative_stage=narrative_stage,
        frame_text=frame_text,
    )