"""Always-visible searchable picker list used by the connection modal."""

from rich.text import Text
from textual.message import Message
from textual.widgets import OptionList
from textual.widgets.option_list import Option


class PickerList(OptionList):
    """An option list with a Select-like value API.

    Keeping the value API small makes provider state handling independent of
    how choices are rendered, while presenting choices as a visible list.
    """

    class Changed(Message):
        """Posted when the user or application changes the selected value."""

        def __init__(self, picker: "PickerList", value: str | None) -> None:
            super().__init__()
            self.picker = picker
            self.value = value

        @property
        def control(self) -> "PickerList":
            return self.picker

    def __init__(
        self,
        options: list[tuple[Text | str, str]],
        *,
        value: str | None = None,
        id: str | None = None,
    ) -> None:
        self._value: str | None = None
        self._values: list[str] = []
        super().__init__(id=id, compact=True, markup=False)
        self.set_options(options)
        self._set_value(value, post=False)

    @property
    def value(self) -> str | None:
        return self._value

    @value.setter
    def value(self, value: object) -> None:
        self._set_value(value if isinstance(value, str) else None, post=True)

    def _set_value(self, value: str | None, *, post: bool) -> None:
        if value not in self._values:
            value = None
        changed = value != self._value
        self._value = value
        self.highlighted = self._values.index(value) if value in self._values else None
        if post and changed and self.is_mounted:
            self.post_message(self.Changed(self, value))

    def set_options(self, options: list[tuple[Text | str, str]]) -> None:
        """Replace the visible rows while retaining selection when possible."""
        previous = self._value
        self.clear_options()
        self._values = [value for _, value in options]
        self.add_options([Option(label, id=value) for label, value in options])
        self._set_value(previous if previous in self._values else None, post=False)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        option_id = event.option.id
        self._set_value(option_id if isinstance(option_id, str) else None, post=True)
