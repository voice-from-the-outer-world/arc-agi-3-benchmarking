from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from arcagi3.agent import HUMAN_ACTIONS, HUMAN_ACTIONS_LIST, MultimodalAgent
from arcagi3.prompts import PromptManager
from arcagi3.schemas import GameStep
from arcagi3.utils.context import SessionContext
from arcagi3.utils.formatting import grid_to_text_matrix
from arcagi3.utils.parsing import extract_json_from_response

logger = logging.getLogger(__name__)


class MalformedJsonReset(RuntimeError):
    pass


class StateMemoryAgent(MultimodalAgent):
    """
    State-memory-only agent.

    Per turn, the model sees only:
    - memory scratchpad
    - previous state (grid)
    - chosen action in previous state
    - current state (grid)
    - available actions

    Memory is intentionally unbounded by this agent implementation.
    """

    def __init__(
        self,
        *args,
        use_vision: bool = False,
        show_images: bool = False,
        memory_word_limit: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Accepted for compatibility with ARC3Tester constructor wiring.
        # StateMemoryAgent is text-grid only by design.
        self.use_vision = False
        self.show_images = show_images
        self.memory_word_limit = memory_word_limit
        self.prompt_manager = PromptManager()

    def _get_available_action_pairs(self, context: SessionContext) -> List[tuple[str, str]]:
        if context.game.available_actions:
            action_pairs: List[tuple[str, str]] = []
            for raw_action in context.game.available_actions:
                try:
                    idx = int(str(raw_action))
                except (TypeError, ValueError):
                    continue
                if 1 <= idx <= len(HUMAN_ACTIONS_LIST):
                    action_name = HUMAN_ACTIONS_LIST[idx - 1]
                    action_pairs.append((action_name, HUMAN_ACTIONS[action_name]))
            if action_pairs:
                return action_pairs
        return [(name, desc) for name, desc in HUMAN_ACTIONS.items()]

    @staticmethod
    def _normalize_action_name(raw_action: Any) -> Optional[str]:
        if raw_action is None:
            return None
        action_text = str(raw_action).strip().upper()
        if not action_text:
            return None
        if action_text.startswith("ACTION"):
            suffix = action_text.replace("ACTION", "", 1)
            if suffix.isdigit():
                return f"ACTION{int(suffix)}"
            return action_text
        if action_text.isdigit():
            return f"ACTION{int(action_text)}"
        return action_text

    def _coerce_decision_action(
        self, context: SessionContext, decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        action_pairs = self._get_available_action_pairs(context)
        valid_actions = {name for name, _ in action_pairs}

        action_name = self._normalize_action_name(decision.get("action"))
        if action_name not in valid_actions:
            action_name = None

        if action_name is None:
            human_action_raw = str(decision.get("human_action", "")).strip().lower()
            for candidate_name, candidate_desc in action_pairs:
                desc = candidate_desc.lower()
                if human_action_raw == desc or human_action_raw.startswith(desc):
                    action_name = candidate_name
                    break

        if action_name is None:
            raise ValueError(
                f"No valid action in model response. Received action={decision.get('action')!r}, "
                f"human_action={decision.get('human_action')!r}, valid={sorted(valid_actions)}"
            )

        action_payload: Dict[str, Any] = {"action": action_name}
        if action_name == "ACTION6":
            try:
                raw_x = int(decision.get("x", 0))
            except (TypeError, ValueError):
                raw_x = 0
            try:
                raw_y = int(decision.get("y", 0))
            except (TypeError, ValueError):
                raw_y = 0

            # State-memory is text-grid-first: use direct 64x64 grid coordinates.
            # Backward compatibility: if model outputs legacy 0..127 values, map them down.
            if raw_x > 63 or raw_y > 63:
                grid_x = max(0, min(raw_x, 127)) // 2
                grid_y = max(0, min(raw_y, 127)) // 2
            else:
                grid_x = max(0, min(raw_x, 63))
                grid_y = max(0, min(raw_y, 63))

            # Provide explicit API payload to bypass generic ACTION6 downscaling in base harness.
            action_payload["data"] = {"x": grid_x, "y": grid_y}
            action_payload["x"] = grid_x
            action_payload["y"] = grid_y
        return action_payload

    def _parse_json_with_retry(
        self,
        context: SessionContext,
        messages: List[Dict[str, Any]],
        step_name: str,
    ) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        last_message = ""
        for attempt in range(2):
            response = self.provider.call_with_tracking(
                context,
                messages,
                step_name=f"{step_name}.attempt_{attempt + 1}",
            )
            message_text = self.provider.extract_content(response)
            last_message = message_text or ""
            try:
                return extract_json_from_response(message_text)
            except Exception as e:
                last_error = e
                logger.warning(
                    "Malformed JSON in %s step (attempt %s/2): %s",
                    step_name,
                    attempt + 1,
                    str(e),
                )
                if attempt == 0:
                    continue

        logger.error(
            "Two consecutive malformed JSON outputs during %s. Resetting game. "
            "Last error: %s. Last response (truncated): %s",
            step_name,
            last_error,
            last_message[:200],
        )
        raise MalformedJsonReset("Malformed JSON twice in a row.")

    def _build_turn_context(self, context: SessionContext) -> Dict[str, str]:
        memory = context.datastore.get("memory_prompt", "")
        if not isinstance(memory, str) or not memory.strip():
            memory = "(empty)"

        previous_action = context.datastore.get("previous_action")
        if isinstance(previous_action, dict):
            previous_action_text = json.dumps(previous_action, ensure_ascii=True)
        else:
            previous_action_text = "None"

        previous_grid = context.frames.previous_grids[-1] if context.frames.previous_grids else None
        current_grid = context.frames.frame_grids[-1] if context.frames.frame_grids else None

        previous_state_text = (
            grid_to_text_matrix(previous_grid) if previous_grid is not None else "None"
        )
        current_state_text = grid_to_text_matrix(current_grid) if current_grid is not None else "None"

        action_pairs = self._get_available_action_pairs(context)
        available_actions_list = "\n".join(f"- {name} = {desc}" for name, desc in action_pairs)
        return {
            "memory": memory,
            "previous_state": previous_state_text,
            "previous_action": previous_action_text,
            "current_state": current_state_text,
            "available_actions": available_actions_list,
        }

    def validate_action(self, context: SessionContext, action_name: str) -> bool:
        if not action_name or not action_name.startswith("ACTION"):
            return False
        if not context.game.available_actions:
            return True
        try:
            action_num = action_name.replace("ACTION", "")
            normalized_available = {str(a) for a in context.game.available_actions}
            return action_num in normalized_available
        except Exception:
            return False

    def step(self, context: SessionContext) -> GameStep:
        try:
            user_header = self.prompt_manager.render("state_memory_user", {})
            turn_context = self._build_turn_context(context)

            messages = [
                {
                    "role": "system",
                    "content": self.prompt_manager.render("state_memory_system", {}),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_header},
                        {"type": "text", "text": f"[MEMORY]\n{turn_context['memory']}"},
                        {
                            "type": "text",
                            "text": f"[PREVIOUS_STATE]\n{turn_context['previous_state']}",
                        },
                        {
                            "type": "text",
                            "text": f"[PREVIOUS_ACTION]\n{turn_context['previous_action']}",
                        },
                        {
                            "type": "text",
                            "text": f"[CURRENT_STATE]\n{turn_context['current_state']}",
                        },
                        {
                            "type": "text",
                            "text": f"[AVAILABLE_ACTIONS]\n{turn_context['available_actions']}",
                        },
                    ],
                },
            ]
            decision = self._parse_json_with_retry(context, messages, "state_memory.decide")
            action_payload = self._coerce_decision_action(context, decision)
            action_name = action_payload["action"]

            if not self.validate_action(context, str(action_name)):
                raise ValueError(
                    f"Invalid action '{action_name}' for available_actions={context.game.available_actions}"
                )

            memory_text = decision.get("memory")
            if isinstance(memory_text, str):
                context.datastore["memory_prompt"] = memory_text

            context.datastore["previous_action"] = {
                "human_action": decision.get("human_action"),
                "action": action_name,
                "x": action_payload.get("x"),
                "y": action_payload.get("y"),
                "reasoning": decision.get("reasoning"),
                "expected_result": decision.get("expected_result"),
            }

            reasoning = {
                "decision": {
                    "human_action": decision.get("human_action"),
                    "action": action_name,
                    "x": action_payload.get("x"),
                    "y": action_payload.get("y"),
                    "reasoning": decision.get("reasoning"),
                    "expected_result": decision.get("expected_result"),
                }
            }
            return GameStep(action=action_payload, reasoning=reasoning)
        except MalformedJsonReset:
            logger.error("Forcing RESET action due to malformed JSON in state-memory agent.")
            return GameStep(
                action={"action": "RESET"},
                reasoning={"system": "malformed_json_reset"},
            )


__all__ = ["StateMemoryAgent"]
