from typing import Dict, Optional

from .._models import BaseModel

__all__ = [
    "GetSchedulerStateResponse",
    "SchedulerLimits",
    "SchedulerOverloadBehavior",
    "SchedulerQueueDepths",
]


class SchedulerLimits(BaseModel):
    recommended_max_outstanding_sampling_requests: Optional[int] = None
    recommended_max_outstanding_eval_requests: Optional[int] = None
    max_queued_requests_per_subject: Optional[int] = None
    max_active_requests_per_subject: Optional[int] = None
    max_queued_interactive_sampling_requests: Optional[int] = None
    max_queued_checkpoint_interactive_sampling_requests: Optional[int] = None
    max_queued_bulk_sampling_requests: Optional[int] = None
    max_queued_checkpoint_bulk_sampling_requests: Optional[int] = None
    max_active_bulk_sampling_requests: Optional[int] = None
    max_queued_compute_logprobs_requests: Optional[int] = None
    max_active_compute_logprobs_requests: Optional[int] = None
    sampling_coalescing_window_ms: Optional[int] = None
    compute_logprobs_coalescing_window_ms: Optional[int] = None


class SchedulerOverloadBehavior(BaseModel):
    sampling: Optional[str] = None
    compute_logprobs: Optional[str] = None


class SchedulerQueueDepths(BaseModel):
    total_pending_requests: Optional[int] = None
    by_request_class: Optional[Dict[str, int]] = None
    by_queue_class: Optional[Dict[str, int]] = None
    by_subject: Optional[Dict[str, int]] = None


class GetSchedulerStateResponse(BaseModel):
    scheduler_mode: Optional[str] = None
    scheduler_subject_key: Optional[str] = None
    fairness_policy: Optional[str] = None
    limits: Optional[SchedulerLimits] = None
    overload_behavior: Optional[SchedulerOverloadBehavior] = None
    queue_depths: Optional[SchedulerQueueDepths] = None
