from __future__ import annotations

from arcagi3.adcr_agent import ADCRAgent, StateMemoryAgent

definition = {
    "name": "adcr",
    "description": "Analyze -> Decide -> Convert -> Review reference agent",
    "agent_class": ADCRAgent,
}

state_memory_definition = {
    "name": "state-memory",
    "description": "Unbounded-memory agent with state-only turn input",
    "agent_class": StateMemoryAgent,
}

agents = [definition, state_memory_definition]

__all__ = ["definition", "state_memory_definition", "agents"]
