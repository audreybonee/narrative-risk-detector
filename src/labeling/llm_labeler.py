"""
LLM-assisted labeling using Hugging Face Inference API.
Uses LLAMA 3.1 8B for claim extraction and coordination assessment.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import json
import re
from huggingface_hub import InferenceClient

from src.models.article import Document
from src.models.cluster import Cluster
from src.models.signal import Alert
from src.config import settings


class ClaimTemplate(BaseModel):
    """Extracted claim template."""
    template: str
    claim_type: str  # causal, hoax, manipulation, conspiracy
    instances: List[str] = []
    frequency: int = 1


class StanceEmotion(BaseModel):
    """Stance and emotion analysis."""
    stance: str  # supportive, opposing, neutral, mixed
    emotions: Dict[str, float] = {}  # anger, fear, excitement, etc.
    manipulation_indicators: List[str] = []


class CoordinationAssessment(BaseModel):
    """LLM coordination risk assessment."""
    risk_level: str  # low, medium, high, critical
    confidence: float
    indicators: List[str] = []
    counter_indicators: List[str] = []
    recommendation: str = ""


class HuggingFaceLabeler:
    """
    LLM labeler using Hugging Face Inference API.

    Uses LLAMA 3.1 8B Instruct for:
    - Claim template extraction
    - Stance and emotion analysis
    - Coordination risk assessment
    - Alert explanation generation
    """

    # Prompt templates
    CLAIM_TEMPLATE_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert analyst detecting coordinated messaging patterns. Extract recurring claim templates from social media posts. Focus on patterns, not individuals. Output valid JSON only.<|eot_id|><|start_header_id|>user<|end_header_id|>

Analyze these posts and extract claim templates (recurring patterns like "X causes Y", "Z is a hoax", "They want to control X").

Posts:
{posts}

Extract claim templates in this exact JSON format:
{{"templates": [
    {{"template": "pattern", "claim_type": "causal|hoax|manipulation|conspiracy", "frequency": N}}
]}}

Only include patterns appearing 2+ times. No personal identifiers.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    STANCE_EMOTION_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert at analyzing sentiment and emotional tone in social media. Analyze aggregate patterns only, not individuals. Output valid JSON only.<|eot_id|><|start_header_id|>user<|end_header_id|>

Analyze the stance and emotional tone of these posts. DO NOT identify individuals.

Posts:
{posts}

Provide analysis in this exact JSON format:
{{
    "stance": "supportive|opposing|neutral|mixed",
    "emotions": {{"anger": 0.0-1.0, "fear": 0.0-1.0, "excitement": 0.0-1.0, "disgust": 0.0-1.0}},
    "manipulation_indicators": ["list of manipulation tactics detected"]
}}

Focus on: us-vs-them framing, urgency language, emotional appeals, conspiracy framing.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    COORDINATION_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a Trust & Safety analyst assessing coordination signals in social media clusters. Provide objective analysis. Output valid JSON only.<|eot_id|><|start_header_id|>user<|end_header_id|>

Assess if this cluster shows signs of coordinated messaging:

Cluster Statistics:
- Size: {size} posts
- Unique authors: {authors}
- Coordination ratio: {ratio:.2f} (>2.0 suspicious, >5.0 likely coordinated)
- Semantic coherence: {coherence:.2f}
- Time window: {window}

Sample posts:
{posts}

Assess coordination risk in this exact JSON format:
{{
    "risk_level": "low|medium|high|critical",
    "confidence": 0.0-1.0,
    "indicators": ["specific coordination indicators"],
    "counter_indicators": ["reasons this might be organic"],
    "recommendation": "brief actionable recommendation"
}}

Focus on message patterns, not individual users.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    NARRATIVE_SUMMARY_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert at summarizing coordinated messaging campaigns. Generate concise narrative labels that capture the coordination intent. Output a short phrase only, no explanation.<|eot_id|><|start_header_id|>user<|end_header_id|>

Analyze these posts from a coordination cluster:

{posts}

Generate a 5-8 word narrative label that captures the coordination intent.

Examples of good labels:
- "Coordinated Push to Halt Selling"
- "Bot Campaign for Price Target $100"
- "Organized Stock Pump Campaign"
- "FOMO Manipulation for CYBR"

Respond with ONLY the narrative label, nothing else.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    EXPLANATION_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a Trust & Safety analyst explaining coordination alerts to reviewers. Be clear, factual, and actionable. No personally identifiable information.<|eot_id|><|start_header_id|>user<|end_header_id|>

Explain this coordination alert for an analyst review:

Alert Details:
- Type: {alert_type}
- Severity: {severity}
- Cluster size: {size} posts from {authors} unique authors
- Coordination ratio: {ratio:.2f}
- Duplicate content score: {dup_score:.0%}

Evidence samples:
{evidence}

Write a clear explanation (2-3 paragraphs):
1. What pattern was detected and why it's significant
2. Specific indicators observed
3. Recommended next steps for investigation

No personal identifiers. Focus on patterns.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    def __init__(
        self,
        token: Optional[str] = None,
        model_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ):
        """
        Initialize HuggingFace labeler.

        Args:
            token: HuggingFace API token (uses settings if not provided)
            model_id: Model ID (uses settings if not provided)
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.token = token or settings.hf_token
        self.model_id = model_id or settings.hf_model_id
        self.temperature = temperature or settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.timeout = timeout or settings.llm_timeout

        if not self.token:
            raise ValueError(
                "HuggingFace token required. Set HF_TOKEN in .env or pass token parameter."
            )

        self.client = InferenceClient(token=self.token)

    def _call_llm(self, prompt: str) -> str:
        """
        Call Hugging Face Inference API.

        Args:
            prompt: Full prompt with template

        Returns:
            Generated text response
        """
        response = self.client.text_generation(
            prompt,
            model=self.model_id,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            return_full_text=False,
        )
        return response

    def _format_posts(
        self,
        documents: List[Document],
        max_posts: int = 10,
        max_chars: int = 300
    ) -> str:
        """Format posts for prompt, respecting privacy."""
        posts = []
        for i, doc in enumerate(documents[:max_posts]):
            text = doc.combined_text[:max_chars]
            posts.append(f"[Post {i+1}]: {text}")
        return "\n".join(posts)

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from LLM response."""
        # Try to find JSON in response
        try:
            # Look for JSON block
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # Try parsing entire response
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return None

    def extract_claim_templates(
        self,
        documents: List[Document]
    ) -> List[ClaimTemplate]:
        """
        Extract claim templates from documents.

        Args:
            documents: Documents to analyze

        Returns:
            List of extracted claim templates
        """
        prompt = self.CLAIM_TEMPLATE_PROMPT.format(
            posts=self._format_posts(documents)
        )

        response = self._call_llm(prompt)
        data = self._extract_json(response)

        if not data or "templates" not in data:
            return []

        return [
            ClaimTemplate(
                template=t.get("template", ""),
                claim_type=t.get("claim_type", "unknown"),
                frequency=t.get("frequency", 1)
            )
            for t in data["templates"]
            if t.get("template")
        ]

    def analyze_stance_emotion(
        self,
        documents: List[Document]
    ) -> StanceEmotion:
        """
        Analyze stance and emotion in documents.

        Args:
            documents: Documents to analyze

        Returns:
            StanceEmotion analysis
        """
        prompt = self.STANCE_EMOTION_PROMPT.format(
            posts=self._format_posts(documents)
        )

        response = self._call_llm(prompt)
        data = self._extract_json(response)

        if not data:
            return StanceEmotion(stance="unknown")

        return StanceEmotion(
            stance=data.get("stance", "unknown"),
            emotions=data.get("emotions", {}),
            manipulation_indicators=data.get("manipulation_indicators", [])
        )

    def assess_coordination(
        self,
        cluster: Cluster,
        documents: List[Document]
    ) -> CoordinationAssessment:
        """
        Assess coordination risk for a cluster.

        Args:
            cluster: Cluster to assess
            documents: All documents (filtered to cluster)

        Returns:
            CoordinationAssessment
        """
        # Get cluster documents
        cluster_doc_ids = {m.document_id for m in cluster.members}
        cluster_docs = [d for d in documents if d.id in cluster_doc_ids]

        prompt = self.COORDINATION_PROMPT.format(
            size=cluster.size,
            authors=cluster.unique_authors,
            ratio=cluster.coordination_ratio,
            coherence=cluster.semantic_coherence,
            window=f"{cluster.window_start.strftime('%Y-%m-%d %H:%M')} to {cluster.window_end.strftime('%H:%M')}",
            posts=self._format_posts(cluster_docs)
        )

        response = self._call_llm(prompt)
        data = self._extract_json(response)

        if not data:
            return CoordinationAssessment(
                risk_level="unknown",
                confidence=0.0
            )

        return CoordinationAssessment(
            risk_level=data.get("risk_level", "unknown"),
            confidence=data.get("confidence", 0.0),
            indicators=data.get("indicators", []),
            counter_indicators=data.get("counter_indicators", []),
            recommendation=data.get("recommendation", "")
        )

    def explain_alert(
        self,
        alert: Alert,
        documents: List[Document]
    ) -> str:
        """
        Generate human-readable explanation for an alert.

        Args:
            alert: Alert to explain
            documents: Documents for context

        Returns:
            Explanation text
        """
        # Format evidence
        evidence_text = "\n".join([
            f"- {e.text_snippet[:200]}..."
            for e in alert.evidence_snippets[:5]
        ])

        prompt = self.EXPLANATION_PROMPT.format(
            alert_type=alert.alert_type.value,
            severity=alert.severity.value,
            size=alert.cluster_size,
            authors=alert.unique_authors,
            ratio=alert.coordination_ratio,
            dup_score=alert.duplicate_burst_score or 0,
            evidence=evidence_text
        )

        return self._call_llm(prompt)

    def summarize_narrative(
        self,
        documents: List[Document],
        fallback_keywords: Optional[List[str]] = None
    ) -> str:
        """
        Generate a concise narrative summary for a cluster.

        Args:
            documents: Documents in the cluster
            fallback_keywords: Keywords to use if LLM fails

        Returns:
            Short narrative label (5-8 words)
        """
        try:
            prompt = self.NARRATIVE_SUMMARY_PROMPT.format(
                posts=self._format_posts(documents, max_posts=8)
            )
            response = self._call_llm(prompt)

            # Clean up response - get just the narrative label
            summary = response.strip().strip('"').strip()

            # Validate length (should be short)
            if len(summary) < 100 and len(summary) > 5:
                return summary
        except Exception:
            pass

        # Fallback: generate from keywords
        if fallback_keywords and len(fallback_keywords) >= 2:
            # Try to generate a meaningful label from keywords
            kws = fallback_keywords[:3]
            ticker = next((k for k in kws if k.startswith('$')), None)
            if ticker:
                action_words = {'buy': 'Buy', 'sell': 'Sell', 'hold': 'Hold',
                               'moon': 'Pump', 'squeeze': 'Squeeze', 'target': 'Target'}
                action = next((action_words[k] for k in kws if k in action_words), 'Campaign')
                return f"{action} Campaign for {ticker.upper()}"
            return f"Coordinated {' '.join(kws[:2]).title()} Campaign"

        return "Unclassified Coordination Pattern"

    def batch_label_clusters(
        self,
        clusters: List[Cluster],
        documents: List[Document],
        include_claims: bool = True,
        include_stance: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Batch label multiple clusters.

        Args:
            clusters: Clusters to label
            documents: All documents
            include_claims: Extract claim templates
            include_stance: Analyze stance/emotion

        Returns:
            List of label dictionaries
        """
        results = []

        for cluster in clusters:
            cluster_doc_ids = {m.document_id for m in cluster.members}
            cluster_docs = [d for d in documents if d.id in cluster_doc_ids]

            result = {
                "cluster_id": cluster.id,
                "coordination_assessment": self.assess_coordination(cluster, documents).model_dump(),
            }

            if include_claims:
                claims = self.extract_claim_templates(cluster_docs)
                result["claim_templates"] = [c.model_dump() for c in claims]

            if include_stance:
                stance = self.analyze_stance_emotion(cluster_docs)
                result["stance_emotion"] = stance.model_dump()

            results.append(result)

        return results


def get_llm_labeler() -> HuggingFaceLabeler:
    """Get singleton LLM labeler instance."""
    if not hasattr(get_llm_labeler, "_instance"):
        get_llm_labeler._instance = HuggingFaceLabeler()
    return get_llm_labeler._instance
