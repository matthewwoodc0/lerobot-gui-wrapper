from __future__ import annotations

from .gui_qt_workflows_page import QtWorkflowsPage


class QtWorkflowQueuePage(QtWorkflowsPage):
    def __init__(self, *, workflow_queue, **kwargs):
        super().__init__(workflows_service=workflow_queue, **kwargs)

__all__ = ["QtWorkflowQueuePage", "QtWorkflowsPage"]
