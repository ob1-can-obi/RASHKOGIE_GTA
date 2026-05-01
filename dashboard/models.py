"""
Pydantic models for the RASHKOGIE Training Dashboard API.

Covers: metrics rows, decision counts, session info, parameter updates,
WebSocket messages, and checkpoint metadata.
"""

from typing import Optional

from pydantic import BaseModel, Field


class MetricsRow(BaseModel):
    session_id: str
    module: str
    step: int
    epoch: Optional[int] = None
    loss: Optional[float] = None
    reward: Optional[float] = None
    episode_return: Optional[float] = None
    grad_norm: Optional[float] = None
    clipped: Optional[int] = None
    lr: Optional[float] = None
    timestamp: str


class DecisionCountRow(BaseModel):
    session_id: str
    step: int
    explore: int = 0
    rollback: int = 0
    interrupt: int = 0
    commit_next: int = 0
    nodes_expanded: Optional[int] = None
    search_depth: Optional[int] = None
    timestamp: str


class SessionInfo(BaseModel):
    session_id: str
    module: str
    started_at: str
    ended_at: Optional[str] = None
    status: str = "running"
    final_metric: Optional[float] = None
    total_steps: Optional[int] = None
    checkpoint_path: Optional[str] = None


class ParamUpdate(BaseModel):
    module: str
    params: dict  # keys: lr, entropy_coeff, think_cost, batch_size, convergence.threshold, convergence.patience


class WSMessage(BaseModel):
    type: str
    data: Optional[dict] = None
    module: Optional[str] = None
    params: Optional[dict] = None
    session_id: Optional[str] = None
    config_version: Optional[int] = None


class CheckpointInfo(BaseModel):
    session_id: str
    module: str
    filename: str
    file_size: int
    timestamp: str
    final_metric: Optional[float] = None
