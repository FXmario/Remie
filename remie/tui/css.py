"""Stylesheet for the Remie TUI."""

CSS = """
Screen {
    layout: vertical;
}

HeaderIcon {
    display: none;
}

#log {
    width: 1fr;
    height: 1fr;
    padding: 0 1;
    border: round $primary;
    margin: 0 1;
}

#prompt {
    height: 3;
    width: 1fr;
    margin: 0;
    border: round $primary;
    border-title-align: right;
}

#prompt .text-area--placeholder {
    color: grey;
}

#prompt-box {
    height: 4;
    width: 1fr;
}

#input-row.slash-open {
    height: auto;
}

#input-row.slash-open #prompt-box {
    height: auto;
}

#slash-command-popup {
    width: 100%;
    height: auto;
    max-height: 6;
    padding: 0;
    border: round $primary;
    background: $surface;
    display: none;
}

#slash-command-popup > .option-list--option {
    padding: 0 1;
}

#slash-command-popup > .option-list--option-highlighted {
    color: $text;
    background: $primary 35%;
    text-style: bold;
}

#input-row {
    height: 5;
    width: 100%;
    padding: 0 1 1 1;
    align: left middle;
}

#status {
    width: 8;
    height: 4;
    margin-right: 0;
    content-align: center middle;
    background: $panel;
}

#status-gif {
    width: 8;
    height: 4;
}

#tmux-spinner {
    width: 1;
    height: 1;
    margin-right: 0;
    content-align: center middle;
    display: none;
}

#model-row {
    dock: top;
    align: right top;
    height: 1;
    width: 100%;
    padding: 0 1;
}

#model-badge {
    height: 1;
    width: auto;
    padding: 0 1;
    margin-right: 1;
    content-align: center middle;
    border: none;
    color: $text;
    background: $panel;
}

#model-badge:hover {
    background: $primary 20%;
}
"""


