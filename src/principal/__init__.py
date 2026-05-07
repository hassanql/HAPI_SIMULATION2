from src.principal.benign import make_benign_task
from src.principal.covert import CovertStrategy, make_covert_task
from src.principal.direct import make_direct_task
from src.principal.task_spec import TaskSpec

__all__ = [
    "CovertStrategy",
    "TaskSpec",
    "make_benign_task",
    "make_covert_task",
    "make_direct_task",
]
