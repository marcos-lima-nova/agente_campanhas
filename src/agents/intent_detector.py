"""
Intent Detector
===============
Classifies a user message and its attachments into one of three intents:

- ``ANALYSIS``  — the user wants a document analyzed (briefing/edital/RFP).
- ``QA``        — the user asks a question (may reference prior analysis or the
                   knowledge base).
- ``MIXED``     — the user wants both: analyze a document *and* ask a question
                   (e.g., "analyze this briefing and suggest similar campaigns").

Detection strategy
------------------
A **rule-based hybrid** approach is used — no LLM call required:

1. If files/attachments are present AND the message also contains QA signals
   → ``MIXED``
2. If files/attachments are present (no QA signals)
   → ``ANALYSIS``
3. If no attachments but message contains analysis-only signals
   → ``ANALYSIS``
4. Otherwise (default)
   → ``QA``

All pattern matching is case-insensitive and works for both Portuguese-BR and
English inputs.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.utils.logging_config import setup_logging

logger = setup_logging("intent_detector")


# ---------------------------------------------------------------------------
# Intent enum
# ---------------------------------------------------------------------------

class Intent(str, Enum):
    """The three possible routing intents."""

    ANALYSIS = "analysis"
    """Route to the Analysis Agent (DocumentOrchestrator)."""

    QA = "qa"
    """Route to the Q&A Agent (RAGPipeline)."""

    MIXED = "mixed"
    """Run Analysis Agent first, then Q&A Agent with the updated state."""


# ---------------------------------------------------------------------------
# Keyword lists  (PT-BR + English)
# ---------------------------------------------------------------------------

# Substrings that strongly signal the user wants a NEW document analyzed.
# These are only actioned when a NEW file is present OR no prior analysis exists.
_ANALYSIS_PATTERNS: List[str] = [
    # Portuguese — imperative / request forms
    "analisa", "anális", "analise",     # analisar / análise
    "resumir", "resumo", "resume",      # summary
    "extrair", "extrai",                # extract
    "sintetiz",                         # synthesize
    "sumariz",                          # summarize
    "classific",                        # classify
    "avali",                            # evaluate
    # NOTE: generic nouns like "document", "briefing", "edital" are intentionally
    # excluded here.  They are too common in follow-up questions (e.g., "what does
    # the briefing say?") and caused false ANALYSIS triggers.
    # English — imperative / request forms
    "analyz", "analys",
    "summari",
    "extract",
    "review this",
    "summarize this",
    "analyze this",
    "analyse this",
]

# Substrings that unambiguously request a RE-ANALYSIS of the same document.
# These override the has_prior_analysis downgrade and always trigger Analysis Agent.
_REANALYSIS_PATTERNS: List[str] = [
    # Portuguese
    "reanalis", "re-analis",            # reanalisar
    "analisar novamente", "analisar de novo",
    "analise novamente", "analise de novo",
    "refazer a analise", "refazer análise",
    "processar novamente", "reprocessar",
    "nova análise", "nova analise",
    "atualizar análise", "atualizar analise",
    # English
    "reanalyz", "re-analyz",
    "analyze again", "analyse again",
    "redo the analysis", "redo analysis",
    "reprocess",
    "run analysis again",
]

# Substrings that signal the user is asking a question.
_QA_PATTERNS: List[str] = [
    # Portuguese (question words / verbs)
    "quais", "qual ", "como ", "quando", "onde", "por que", "por qu",
    "o que", "quem",
    "suger", "recomend",                # suggest / recommend
    "buscar", "encontrar", "procurar",  # search / find
    "similar", "semelhant", "parecid",  # similar
    "campanha", "campanhas",            # campaign(s)
    "propor", "propost",                # propose
    "respond", "resposta",              # answer / response
    "explicar", "explique", "explic",   # explain
    "detalh",                           # detail
    # English
    "what ", "which ", "how ", "who ", "when ", "where ", "why ",
    "suggest", "recommend",
    "search", "find", "look for",
    "similar", "comparable",
    "propose", "should we",
    "answer", "explain",
    "tell me",
]

# A lone "?" is also a strong QA signal.
_QUESTION_MARK_RE = re.compile(r"\?")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class IntentResult:
    """The outcome of intent detection for a single user turn.

    Attributes
    ----------
    intent          : Detected intent.
    confidence      : Rough confidence score in [0.0, 1.0].
    reasoning       : Human-readable explanation of the decision.
    analysis_signals: Analysis-related keywords/patterns found in the message.
    qa_signals      : QA-related keywords/patterns found in the message.
    """

    intent: Intent
    confidence: float
    reasoning: str
    analysis_signals: List[str] = field(default_factory=list)
    qa_signals: List[str] = field(default_factory=list)
    is_reanalysis: bool = False


# ---------------------------------------------------------------------------
# IntentDetector
# ---------------------------------------------------------------------------

class IntentDetector:
    """Classifies a user message into ANALYSIS, QA, or MIXED intent.

    Usage::

        detector = IntentDetector()
        result = detector.detect_intent(
            message="analyze this briefing and suggest similar campaigns",
            has_attachments=True,
        )
        print(result.intent)   # Intent.MIXED
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_intent(
        self,
        message: str,
        has_attachments: bool = False,
        state: Optional[Any] = None,
    ) -> IntentResult:
        """Detect the intent of a user message.

        Parameters
        ----------
        message        : The latest user message text.
        has_attachments: Whether the request contains file attachments.
        state          : The current ``Session`` object.  When the session
                         already contains a completed analysis
                         (``state.last_analysis`` is not ``None``) and no new
                         files are attached, any analysis/mixed intent is
                         **downgraded** to QA — the user is asking about an
                         already-analyzed document, not requesting a new
                         analysis.

        Returns
        -------
        IntentResult with the detected ``Intent``, confidence, and signals.
        """
        msg_lower = message.lower()

        analysis_signals = self._find_signals(msg_lower, _ANALYSIS_PATTERNS)
        reanalysis_signals = self._find_signals(msg_lower, _REANALYSIS_PATTERNS)
        qa_signals = self._find_signals(msg_lower, _QA_PATTERNS)

        # Question mark always counts as QA signal
        if _QUESTION_MARK_RE.search(message):
            if "?" not in qa_signals:
                qa_signals.append("?")

        has_analysis_signal = bool(analysis_signals)
        has_reanalysis_request = bool(reanalysis_signals)
        has_qa_signal = bool(qa_signals)

        # Check whether a prior analysis already exists in the session.
        has_prior_analysis = (
            state is not None
            and getattr(state, "last_analysis", None) is not None
        )
        # Check if there's an active document in focus (new or existing)
        has_active_document = (
            state is not None
            and getattr(state, "active_document_id", None) is not None
        )

        # Debug: log session state for intent detection
        logger.debug(
            f"IntentDetector: has_attachments={has_attachments}, "
            f"has_prior_analysis={has_prior_analysis}, "
            f"has_active_document={has_active_document}, "
            f"has_analysis_signal={has_analysis_signal}, "
            f"has_reanalysis_request={has_reanalysis_request}, "
            f"has_qa_signal={has_qa_signal}"
        )

        # --- Decision tree ---
        # Priority (highest to lowest):
        # 1. Explicit re-analysis request       → ANALYSIS (even if no new file)
        # 2. New file attachment + question     → MIXED
        # 3. New file attachment only           → ANALYSIS
        # 4. Prior analysis exists + no file   → QA (NEVER re-analyze automatically)
        # 5. Analysis keywords + no prior doc  → ANALYSIS
        # 6. Both analysis + QA, no prior doc  → MIXED
        # 7. Default                           → QA

        if has_reanalysis_request:
            # User explicitly asked to re-analyze — always honor it
            result = IntentResult(
                intent=Intent.ANALYSIS,
                confidence=0.98,
                reasoning=(
                    f"Explicit re-analysis request detected "
                    f"({', '.join(reanalysis_signals[:2])}) — routing to Analysis Agent."
                ),
                analysis_signals=analysis_signals + reanalysis_signals,
                qa_signals=qa_signals,
                is_reanalysis=True,
            )

        elif has_attachments and has_qa_signal:
            # New files present AND the user also asks a question → run both
            result = IntentResult(
                intent=Intent.MIXED,
                confidence=0.92,
                reasoning=(
                    "Attachments present and QA signals detected — "
                    "will run Analysis Agent first then Q&A Agent."
                ),
                analysis_signals=analysis_signals,
                qa_signals=qa_signals,
            )

        elif has_attachments:
            # New files present, no explicit question → pure analysis
            result = IntentResult(
                intent=Intent.ANALYSIS,
                confidence=0.95,
                reasoning=(
                    "Attachments present without explicit QA signals — "
                    "routing to Analysis Agent."
                ),
                analysis_signals=analysis_signals,
                qa_signals=qa_signals,
            )

        elif has_prior_analysis and not has_attachments:
            # Session already has a completed analysis; no new files supplied.
            # Unconditionally route to QA — the user is asking about the
            # existing document, not requesting a new analysis.
            result = IntentResult(
                intent=Intent.QA,
                confidence=0.92,
                reasoning=(
                    "Prior analysis in session, no new attachments, no explicit "
                    "re-analysis request — routing to Q&A Agent (cached analysis used)."
                ),
                analysis_signals=analysis_signals,
                qa_signals=qa_signals,
            )

        elif has_active_document and not has_attachments:
            # An active document is set but analysis keywords are matched;
            # since no new file was provided, route to QA.
            result = IntentResult(
                intent=Intent.QA,
                confidence=0.88,
                reasoning=(
                    "Active document in session, no new file — routing to Q&A Agent."
                ),
                analysis_signals=analysis_signals,
                qa_signals=qa_signals,
            )

        elif has_analysis_signal and not has_qa_signal:
            # Text only, no prior analysis, clearly asking for analysis
            result = IntentResult(
                intent=Intent.ANALYSIS,
                confidence=0.80,
                reasoning=(
                    f"Analysis keywords detected ({', '.join(analysis_signals[:3])}) "
                    "without QA signals — routing to Analysis Agent."
                ),
                analysis_signals=analysis_signals,
                qa_signals=qa_signals,
            )

        elif has_analysis_signal and has_qa_signal:
            # Text only, no prior analysis, contains both kinds of signals
            result = IntentResult(
                intent=Intent.MIXED,
                confidence=0.75,
                reasoning=(
                    "Both analysis and QA signals detected in text-only message — "
                    "routing as MIXED (no prior analysis in session)."
                ),
                analysis_signals=analysis_signals,
                qa_signals=qa_signals,
            )

        else:
            # Default: Q&A
            result = IntentResult(
                intent=Intent.QA,
                confidence=0.85 if has_qa_signal else 0.60,
                reasoning=(
                    "QA signals detected — routing to Q&A Agent."
                    if has_qa_signal
                    else "No strong signals detected — defaulting to Q&A Agent."
                ),
                analysis_signals=analysis_signals,
                qa_signals=qa_signals,
            )

        logger.info(
            f"Intent detected: {result.intent.value} "
            f"(confidence={result.confidence:.2f}, "
            f"attachments={has_attachments}, "
            f"prior_analysis={has_prior_analysis}, "
            f"reanalysis_request={has_reanalysis_request}, "
            f"analysis_signals={analysis_signals}, "
            f"qa_signals={qa_signals[:3]})"
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_signals(text_lower: str, patterns: List[str]) -> List[str]:
        """Return the subset of *patterns* that appear as substrings in *text_lower*."""
        found = []
        for pattern in patterns:
            if pattern in text_lower:
                found.append(pattern.strip())
        return found
