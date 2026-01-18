"""
Taxonomy enums for the Emergent Narrative Detection System.

Defines the core categorizations for outlets, bias, patterns, and narrative lifecycle.
"""

from enum import Enum


class OutletType(str, Enum):
    """Classification of news outlet types."""
    WIRE_SERVICE = "wire_service"        # Reuters, AP - primary originators
    MAJOR_NATIONAL = "major_national"    # CNN, Fox, NYT, WSJ - high reach
    LOCAL_REGIONAL = "local_regional"    # Denver Post, Miami Herald - geographic focus
    INDEPENDENT = "independent"          # ProPublica, Marshall Project - investigative


class OutletBias(str, Enum):
    """Political lean classification for outlets."""
    LEFT = "left"
    CENTER_LEFT = "center-left"
    CENTER = "center"
    CENTER_RIGHT = "center-right"
    RIGHT = "right"

    @property
    def numeric_value(self) -> float:
        """Convert bias to numeric scale (-1.0 to 1.0)."""
        mapping = {
            "left": -1.0,
            "center-left": -0.5,
            "center": 0.0,
            "center-right": 0.5,
            "right": 1.0,
        }
        return mapping[self.value]


class PatternType(str, Enum):
    """
    Narrative propagation pattern types.

    Ordered roughly by signal strength for detecting coordinated/influenced narratives.
    """
    # Organic patterns (low signal)
    WIRE_ECHO = "wire_echo"                          # Standard wire propagation
    INDEPENDENT_REPORTING = "independent_reporting"  # Original journalism

    # Influenced patterns (medium signal)
    PR_AMPLIFICATION = "pr_amplification"            # Corporate/institutional PR pickup
    NARRATIVE_CONVERGENCE = "narrative_convergence"  # Organic frame alignment

    # High-signal patterns
    SYNCHRONIZED_MESSAGING = "synchronized_messaging"  # Coordinated framing

    @property
    def signal_strength(self) -> str:
        """Return the signal strength category."""
        if self in (PatternType.WIRE_ECHO, PatternType.INDEPENDENT_REPORTING):
            return "low"
        elif self in (PatternType.PR_AMPLIFICATION, PatternType.NARRATIVE_CONVERGENCE):
            return "medium"
        else:
            return "high"


class NarrativeStage(str, Enum):
    """
    Lifecycle stage of an emergent narrative.

    Tracks how a story evolves from first appearance to dormancy.
    """
    NASCENT = "nascent"          # 1-2 sources, < 6 hours old
    EMERGING = "emerging"        # 3-5 sources, cross-outlet spread beginning
    SPREADING = "spreading"      # 6+ sources, clear velocity increase
    ESTABLISHED = "established"  # Wide coverage, consistent framing
    DECLINING = "declining"      # Velocity decreasing
    DORMANT = "dormant"          # Coverage stopped

    @property
    def sort_order(self) -> int:
        """Return numeric order for sorting by lifecycle stage."""
        order = {
            "nascent": 0,
            "emerging": 1,
            "spreading": 2,
            "established": 3,
            "declining": 4,
            "dormant": 5,
        }
        return order[self.value]


class SignalSeverity(str, Enum):
    """Severity level for narrative signals."""
    INFO = "info"          # Informational, no action needed
    LOW = "low"            # Worth monitoring
    MEDIUM = "medium"      # Warrants attention
    HIGH = "high"          # Significant pattern detected
    CRITICAL = "critical"  # Requires immediate review


class NewsSection(str, Enum):
    """Common news section categories."""
    POLITICS = "Politics"
    BUSINESS = "Business"
    TECHNOLOGY = "Technology"
    HEALTH = "Health"
    NATIONAL = "National"
    WORLD = "World"
    INVESTIGATIONS = "Investigations"
    SCIENCE = "Science"
    ENTERTAINMENT = "Entertainment"
    SPORTS = "Sports"
    OPINION = "Opinion"


class WireSource(str, Enum):
    """Wire service or content origin classification."""
    REUTERS = "Reuters"
    AP = "AP"
    AFP = "AFP"
    ORIGINAL = "Original"
    PRESS_RELEASE = "Press Release"
    UNKNOWN = "Unknown"