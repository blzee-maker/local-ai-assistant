"""Background jobs that run without the user present.

Every job runs with no confirmer attached, so any capability requiring
permission refuses: an absent user is not consent.
"""
from .jobs import Job, JobResult, default_jobs
from .journal import Journal, JobRun, JobState
from .scheduler import Scheduler, briefing

__all__ = [
    "Job",
    "JobResult",
    "default_jobs",
    "Journal",
    "JobRun",
    "JobState",
    "Scheduler",
    "briefing",
]
