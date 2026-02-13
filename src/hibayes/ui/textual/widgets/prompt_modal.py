"""Yes/No modal dialog for user prompts."""

from __future__ import annotations

import threading

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label


class PromptModal(ModalScreen[bool]):
    """A modal dialog with a question and Yes/No buttons.

    The pipeline thread blocks on ``event.wait()`` until the user responds.
    """

    DEFAULT_CSS = """
    PromptModal {
        align: center middle;
    }
    PromptModal > Vertical {
        width: 60;
        height: auto;
        max-height: 20;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    PromptModal > Vertical > Label {
        width: 100%;
        text-align: center;
        margin: 1 0;
    }
    PromptModal > Vertical > Horizontal {
        width: 100%;
        height: 3;
        align: center middle;
    }
    PromptModal > Vertical > Horizontal > Button {
        margin: 0 2;
    }
    """

    def __init__(
        self,
        question: str,
        event: threading.Event,
        result_holder: list,
    ) -> None:
        super().__init__()
        self._question = question
        self._event = event
        self._result_holder = result_holder

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self._question)
            with Horizontal():
                yield Button("Yes", id="yes", variant="success")
                yield Button("No", id="no", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self._result_holder[0] = event.button.id == "yes"
        self._event.set()
        self.app.pop_screen()
