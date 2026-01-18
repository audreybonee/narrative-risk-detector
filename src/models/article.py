"""
Article model for the Emergent Narrative Detection System.

Represents a single news article with all metadata and content.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, computed_field
import ast

from .enums import OutletType, OutletBias, PatternType, NewsSection, WireSource


class Article(BaseModel):
    """
    A news article with outlet metadata and content.

    Attributes:
        id: Unique article identifier
        outlet: Publication name
        outlet_type: Classification of outlet (wire, national, local, independent)
        outlet_bias: Political lean of outlet
        title: Article headline
        body: Article content
        author: Byline
        published_at: Publication timestamp
        url: Article URL
        wire_source: Origin wire service or content type
        section: News section category
        pattern_type: Ground truth narrative pattern (for labeled data)
        pattern_topic: Story topic identifier
        keywords: Extracted keywords
        pr_source: PR origin if applicable
        convergent_frame: Convergent framing text if applicable
        coordinated_frame: Coordinated framing text if applicable
    """

    id: str = Field(..., description="Unique article identifier")
    outlet: str = Field(..., description="Publication name")
    outlet_type: OutletType = Field(..., description="Outlet classification")
    outlet_bias: OutletBias = Field(..., description="Political lean")
    title: str = Field(..., description="Article headline")
    body: str = Field(default="", description="Article content")
    author: Optional[str] = Field(default=None, description="Byline")
    published_at: datetime = Field(..., description="Publication timestamp")
    url: Optional[str] = Field(default=None, description="Article URL")
    wire_source: WireSource = Field(default=WireSource.UNKNOWN, description="Content origin")
    section: Optional[str] = Field(default=None, description="News section")

    # Ground truth labels (for labeled datasets)
    pattern_type: Optional[PatternType] = Field(default=None, description="Narrative pattern type")
    pattern_topic: Optional[str] = Field(default=None, description="Story topic identifier")
    keywords: list[str] = Field(default_factory=list, description="Extracted keywords")
    pr_source: Optional[str] = Field(default=None, description="PR origin if applicable")
    convergent_frame: Optional[str] = Field(default=None, description="Convergent framing")
    coordinated_frame: Optional[str] = Field(default=None, description="Coordinated framing")

    # Computed after embedding
    embedding: Optional[list[float]] = Field(default=None, exclude=True)

    @computed_field
    @property
    def combined_text(self) -> str:
        """Combine title and body for embedding generation."""
        if self.body:
            return f"{self.title}\n\n{self.body}"
        return self.title

    @computed_field
    @property
    def bias_numeric(self) -> float:
        """Get numeric bias value for calculations."""
        return self.outlet_bias.numeric_value

    @computed_field
    @property
    def is_wire_origin(self) -> bool:
        """Check if article originated from a wire service."""
        return self.wire_source in (WireSource.REUTERS, WireSource.AP, WireSource.AFP)

    @computed_field
    @property
    def is_pr_origin(self) -> bool:
        """Check if article originated from PR/press release."""
        return self.wire_source == WireSource.PRESS_RELEASE or self.pr_source is not None

    @computed_field
    @property
    def has_coordinated_frame(self) -> bool:
        """Check if article has a coordinated framing label."""
        return self.coordinated_frame is not None and len(self.coordinated_frame) > 0

    @computed_field
    @property
    def has_convergent_frame(self) -> bool:
        """Check if article has a convergent framing label."""
        return self.convergent_frame is not None and len(self.convergent_frame) > 0

    class Config:
        use_enum_values = False  # Keep enum objects for methods


class ArticleCreate(BaseModel):
    """Schema for creating an article from raw CSV data."""

    id: str
    outlet: str
    outlet_type: str
    outlet_bias: str
    title: str
    body: Optional[str] = ""
    author: Optional[str] = None
    published_at: str  # Will be parsed to datetime
    url: Optional[str] = None
    wire_source: Optional[str] = None
    section: Optional[str] = None
    pattern_type: Optional[str] = None
    pattern_topic: Optional[str] = None
    keywords: Optional[str] = None  # Stored as string repr of list
    pr_source: Optional[str] = None
    convergent_frame: Optional[str] = None
    coordinated_frame: Optional[str] = None

    def to_article(self) -> Article:
        """Convert raw data to Article model."""
        # Parse keywords from string representation
        keywords_list = []
        if self.keywords:
            try:
                keywords_list = ast.literal_eval(self.keywords)
            except (ValueError, SyntaxError):
                keywords_list = []

        # Parse wire source
        wire_source = WireSource.UNKNOWN
        if self.wire_source:
            wire_mapping = {
                "Reuters": WireSource.REUTERS,
                "AP": WireSource.AP,
                "AFP": WireSource.AFP,
                "Original": WireSource.ORIGINAL,
                "Press Release": WireSource.PRESS_RELEASE,
            }
            wire_source = wire_mapping.get(self.wire_source, WireSource.UNKNOWN)

        # Parse pattern type
        pattern_type = None
        if self.pattern_type:
            try:
                pattern_type = PatternType(self.pattern_type)
            except ValueError:
                pattern_type = None

        return Article(
            id=self.id,
            outlet=self.outlet,
            outlet_type=OutletType(self.outlet_type),
            outlet_bias=OutletBias(self.outlet_bias),
            title=self.title,
            body=self.body or "",
            author=self.author,
            published_at=datetime.fromisoformat(self.published_at),
            url=self.url,
            wire_source=wire_source,
            section=self.section,
            pattern_type=pattern_type,
            pattern_topic=self.pattern_topic,
            keywords=keywords_list,
            pr_source=self.pr_source if self.pr_source and str(self.pr_source) != "nan" else None,
            convergent_frame=self.convergent_frame if self.convergent_frame and str(self.convergent_frame) != "nan" else None,
            coordinated_frame=self.coordinated_frame if self.coordinated_frame and str(self.coordinated_frame) != "nan" else None,
        )