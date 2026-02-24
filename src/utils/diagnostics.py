import traceback
import time
import json
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from src.utils.logging_config import setup_trace_logging, TRACE_LOG_FILE

# Initialize trace logger
trace_logger = setup_trace_logging()

class AnalysisDiagnostics:
    """
    Helper class for capturing and logging diagnostic information during
    file analysis decision points.
    """

    @staticmethod
    def log_event(
        event_type: str,
        message: str,
        session_id: str = "-",
        extra_state: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Logs a diagnostic event with a simplified call stack and session state.
        
        Args:
            event_type: Category of the event (e.g., 'REQUEST_ENTRY', 'DECISION_POINT', 'ANALYSIS_START')
            message: Descriptive message
            session_id: Current session identifier
            extra_state: Dictionary of additional state to log (e.g., file_ids, active_doc_id)
        """
        # Capture simplified stack trace (skipping the log_event itself)
        stack = traceback.format_stack()
        # Filter stack to show relevant project calls (approximate)
        filtered_stack = [
            line.strip() for line in stack 
            if "site-packages" not in line and "setup_trace_logging" not in line
        ][-5:]  # Keep last 5 frames for context

        payload = {
            "version": "1.0",
            "type": event_type,
            "msg": message,
            "timestamp": time.time(),
            "stack": filtered_stack,
            "state": extra_state or {}
        }

        # Log via the trace logger. 
        # We pass session_id in the 'extra' dict for the SessionContextFilter
        trace_logger.debug(
            f"DIAGNOSTIC | {event_type} | {message} | DATA: {json.dumps(payload)}",
            extra={"session_id": session_id}
        )

    @staticmethod
    def get_summary() -> str:
        """
        Reads the trace log file and generates a summary of analysis triggers.
        """
        if not TRACE_LOG_FILE.exists():
            return "Trace log file not found."

        summary = []
        try:
            with open(TRACE_LOG_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            analysis_starts = 0
            cache_hits = 0
            sessions = set()

            for line in lines:
                if "DIAGNOSTIC | ANALYSIS_START" in line:
                    analysis_starts += 1
                if "DIAGNOSTIC | CACHE_HIT" in line:
                    cache_hits += 1
                if "session=" in line:
                    sid = line.split("session=")[1].split("|")[0].strip()
                    if sid != "-":
                        sessions.add(sid)

            summary.append("# File Analysis Diagnostic Summary")
            summary.append(f"- **Total Trace Logs**: {len(lines)}")
            summary.append(f"- **Total Analysis Initiated**: {analysis_starts}")
            summary.append(f"- **Total Cache Hits**: {cache_hits}")
            summary.append(f"- **Unique Sessions Tracked**: {len(sessions)}")
            
            if analysis_starts > 0:
                summary.append("\n## Redundancy Check")
                if analysis_starts > len(sessions):
                    summary.append("> [!WARNING]")
                    summary.append("> Analysis count exceeds session count. Possible redundant processing detected.")
                else:
                    summary.append("> [!NOTE]")
                    summary.append("> Analysis count is within expected limits (<= sessions).")

        except Exception as e:
            return f"Error reading trace log: {e}"

        return "\n".join(summary)
