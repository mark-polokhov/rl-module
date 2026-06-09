from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QTabWidget,
    QSpinBox,
    QLineEdit,
    QScrollArea,
    QMessageBox,
)

import sys


CONFIGS_DIR = Path("configs")
CONFIGS_DIR.mkdir(exist_ok=True)

REWARD_FILE = CONFIGS_DIR / "reward_function.py"


DARK_STYLE = """
QWidget {
    background-color: #2b2b2b;
    color: #dddddd;
    font-size: 12pt;
}

QLineEdit, QTextEdit, QSpinBox {
    background-color: #3c3f41;
    border: 1px solid #555555;
    border-radius: 6px;
    padding: 4px;
}

QPushButton {
    background-color: #4c566a;
    border: none;
    border-radius: 6px;
    padding: 8px;
}

QPushButton:hover {
    background-color: #5e81ac;
}

QTabWidget::pane {
    border: 1px solid #555555;
}

QTabBar::tab {
    background: #3c3f41;
    padding: 8px;
}

QTabBar::tab:selected {
    background: #5e81ac;
}
"""


class PythonRewardTab(QWidget):

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        title = QLabel("Custom Python Reward Function")
        title.setFont(QFont("Arial", 14))

        self.editor = QTextEdit()

        self.editor.setPlainText(
            """def reward_fn(self, obs, action) -> float:
    reward = 0.0

    return float(reward)
"""
        )

        save_button = QPushButton("Save Reward")

        save_button.clicked.connect(self.save_reward)

        layout.addWidget(title)
        layout.addWidget(self.editor)
        layout.addWidget(save_button)

    def save_reward(self):
        REWARD_FILE.write_text(
            self.editor.toPlainText(),
            encoding="utf-8"
        )

        QMessageBox.information(
            self,
            "Success",
            f"Saved:\n{REWARD_FILE}"
        )


class LinearRewardTab(QWidget):

    def __init__(self):
        super().__init__()

        self.coefficient_inputs = []

        main_layout = QVBoxLayout(self)

        controls_layout = QHBoxLayout()

        self.obs_count = QSpinBox()
        self.obs_count.setRange(1, 500)
        self.obs_count.setValue(13)

        self.act_count = QSpinBox()
        self.act_count.setRange(1, 100)
        self.act_count.setValue(2)

        generate_button = QPushButton("Generate Inputs")
        generate_button.clicked.connect(self.generate_inputs)

        controls_layout.addWidget(QLabel("Observations"))
        controls_layout.addWidget(self.obs_count)

        controls_layout.addWidget(QLabel("Actions"))
        controls_layout.addWidget(self.act_count)

        controls_layout.addWidget(generate_button)

        main_layout.addLayout(controls_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)

        self.scroll_area.setWidget(self.container)

        main_layout.addWidget(self.scroll_area)

        save_button = QPushButton("Save Reward")
        save_button.clicked.connect(self.save_reward)

        main_layout.addWidget(save_button)

    def generate_inputs(self):

        self.coefficient_inputs.clear()

        while self.container_layout.count():
            item = self.container_layout.takeAt(0)

            if item.widget():
                item.widget().deleteLater()

        obs_n = self.obs_count.value()
        act_n = self.act_count.value()

        for i in range(obs_n):

            row = QHBoxLayout()

            coeff = QLineEdit("0.0")

            row.addWidget(coeff)
            row.addWidget(QLabel(f"obs_{i}"))

            self.container_layout.addLayout(row)

            self.coefficient_inputs.append(
                ("obs", i, coeff)
            )

        for i in range(act_n):

            row = QHBoxLayout()

            coeff = QLineEdit("0.0")

            row.addWidget(coeff)
            row.addWidget(QLabel(f"act_{i}"))

            self.container_layout.addLayout(row)

            self.coefficient_inputs.append(
                ("act", i, coeff)
            )

    def save_reward(self):

        lines = [
            "def reward_fn(self, obs, action) -> float:",
            "    reward = ("
        ]

        for variable_type, idx, widget in self.coefficient_inputs:

            coefficient = widget.text().strip()

            if variable_type == "obs":
                lines.append(
                    f"        {coefficient} * obs[{idx}] +"
                )
            else:
                lines.append(
                    f"        {coefficient} * action[{idx}] +"
                )

        if len(lines) > 2:
            lines[-1] = lines[-1].rstrip(" +")

        lines.extend([
            "    )",
            "",
            "    return float(reward)",
            ""
        ])

        REWARD_FILE.write_text(
            "\n".join(lines),
            encoding="utf-8"
        )

        QMessageBox.information(
            self,
            "Success",
            f"Saved:\n{REWARD_FILE}"
        )


class RewardEditorWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("RL Reward Function Editor")
        self.resize(1000, 700)

        layout = QVBoxLayout(self)

        tabs = QTabWidget()

        tabs.addTab(
            PythonRewardTab(),
            "Python"
        )

        tabs.addTab(
            LinearRewardTab(),
            "Builder"
        )

        layout.addWidget(tabs)


def main():

    app = QApplication(sys.argv)

    app.setStyleSheet(DARK_STYLE)

    window = RewardEditorWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()