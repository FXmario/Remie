"""Stylesheet for the Remie TUI."""

CSS = """
Screen {
    layout: vertical;
}

HeaderIcon {
    display: none;
}

#workspace {
    height: 1fr;
    width: 100%;
}

#chat-pane {
    height: 1fr;
    width: 1fr;
}

#tab-sidebar {
    width: 26;
    height: 1fr;
    padding: 1;
    border-right: solid $primary;
}

#tabs-header {
    width: 100%;
    height: 2;
}

#tabs-title {
    width: 1fr;
    height: 1;
    text-style: bold;
}

#tab-list {
    height: 1fr;
    overflow-y: auto;
}

#tab-sidebar Button {
    width: 100%;
    height: 3;
    margin-bottom: 1;
    text-align: left;
}

#tab-sidebar Button.active {
    background: $primary 30%;
}

#tabs-show, #tab-sidebar Button#tab-hide {
    width: 10;
    min-width: 10;
    height: 1;
    min-height: 1;
    border: none;
}

#tab-sidebar Button#tab-hide {
    width: 8;
    min-width: 8;
    margin: 0;
    text-align: right;
}

#tabs-show {
    display: none;
    margin-left: 1;
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

#prompt-box.has-images {
    height: 5;
}

#image-attachments {
    display: none;
    height: 1;
    width: 100%;
    padding: 0 1;
    scrollbar-size-horizontal: 0;
}

#image-attachments.has-images {
    display: block;
}

#image-attachments Button {
    width: auto;
    min-width: 11;
    height: 1;
    min-height: 1;
    margin-right: 1;
    padding: 0 1;
    border: none;
    background: $panel;
    color: $text;
}

#image-attachments Button:hover {
    background: $primary 35%;
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

#input-row.has-images {
    height: 6;
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


