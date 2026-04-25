import sys
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QFileDialog
)
from pypdf import PdfReader

from retriever_tfidf import TfidfRetriever
from answerer import build_short_answer


class PdfLoadWorker(QObject):
    finished = Signal(object, int)
    failed = Signal(str)
    progress = Signal(int, str)

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            reader = PdfReader(self.file_path)
            total_pages = max(len(reader.pages), 1)
            pages = []

            for i, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                pages.append({"page": i, "text": text})
                percent = int((i / total_pages) * 60)
                self.progress.emit(percent, f"Reading pages ({i}/{total_pages})")

            chunks = []
            chunk_id = 0
            chunk_size = 800
            overlap = 100

            total_page_items = max(len(pages), 1)
            for page_idx, page_data in enumerate(pages, start=1):
                page_num = page_data["page"]
                text = page_data["text"]
                if text:
                    start = 0
                    while start < len(text):
                        end = start + chunk_size
                        chunk_text = text[start:end].strip()
                        if chunk_text:
                            chunks.append(
                                {
                                    "chunk_id": chunk_id,
                                    "page": page_num,
                                    "text": chunk_text,
                                }
                            )
                            chunk_id += 1
                        start += chunk_size - overlap

                percent = 60 + int((page_idx / total_page_items) * 30)
                self.progress.emit(percent, f"Creating chunks ({page_idx}/{total_page_items})")

            retriever = TfidfRetriever()
            self.progress.emit(95, "Building search index")
            retriever.fit(chunks)
            self.progress.emit(100, "Done")
            self.finished.emit(retriever, len(chunks))
        except Exception as e:
            self.failed.emit(str(e))


class PdfAssistantApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Professional PDF Book Assistant")
        self.resize(900, 700)

        self.file_path = ""
        self.retriever = TfidfRetriever()
        self._load_thread = None
        self._load_worker = None

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        self.title_label = QLabel("Professional PDF Book Assistant")
        layout.addWidget(self.title_label)

        file_layout = QHBoxLayout()
        self.select_button = QPushButton("Select PDF")
        self.select_button.clicked.connect(self.select_pdf)

        self.file_input = QLineEdit()
        self.file_input.setReadOnly(True)

        self.load_button = QPushButton("Load PDF")
        self.load_button.clicked.connect(self.load_pdf)

        file_layout.addWidget(self.select_button)
        file_layout.addWidget(self.file_input)
        file_layout.addWidget(self.load_button)
        layout.addLayout(file_layout)

        self.status_label = QLabel("Status: No file loaded")
        layout.addWidget(self.status_label)

        question_layout = QHBoxLayout()
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("Enter your question here...")
        self.ask_button = QPushButton("Ask")
        self.ask_button.clicked.connect(self.ask_question)

        question_layout.addWidget(self.question_input)
        question_layout.addWidget(self.ask_button)
        layout.addLayout(question_layout)

        self.answer_label = QLabel("Short Answer:")
        layout.addWidget(self.answer_label)

        self.answer_output = QTextEdit()
        self.answer_output.setReadOnly(True)
        layout.addWidget(self.answer_output)

        self.passages_label = QLabel("Relevant Passages:")
        layout.addWidget(self.passages_label)

        self.passages_output = QTextEdit()
        self.passages_output.setReadOnly(True)
        layout.addWidget(self.passages_output)

        self.setLayout(layout)

    def select_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select PDF", "", "PDF Files (*.pdf)")
        if file_path:
            self.file_path = file_path
            self.file_input.setText(file_path)

    def load_pdf(self):
        if not self.file_path:
            self.status_label.setText("Status: Please select a PDF first")
            return

        if self._load_thread is not None and self._load_thread.isRunning():
            self.status_label.setText("Status: PDF is still loading...")
            return

        self.status_label.setText("Status: Loading PDF, please wait...")
        self.load_button.setEnabled(False)
        self.ask_button.setEnabled(False)

        self._load_thread = QThread()
        self._load_worker = PdfLoadWorker(self.file_path)
        self._load_worker.moveToThread(self._load_thread)

        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.progress.connect(self._on_load_progress)
        self._load_worker.finished.connect(self._on_load_success)
        self._load_worker.failed.connect(self._on_load_failure)
        self._load_worker.finished.connect(self._load_thread.quit)
        self._load_worker.failed.connect(self._load_thread.quit)
        self._load_thread.finished.connect(self._on_load_cleanup)

        self._load_thread.start()

    def _on_load_success(self, retriever: TfidfRetriever, chunk_count: int):
        self.retriever = retriever
        self.status_label.setText(f"Status: PDF loaded successfully ({chunk_count} chunks created)")

    def _on_load_progress(self, percent: int, message: str):
        self.status_label.setText(f"Status: {message}... {percent}%")

    def _on_load_failure(self, error_message: str):
        self.status_label.setText(f"Status: Failed to load PDF ({error_message})")

    def _on_load_cleanup(self):
        self.load_button.setEnabled(True)
        self.ask_button.setEnabled(True)
        self._load_worker = None
        self._load_thread = None

    def ask_question(self):
        query = self.question_input.text().strip()

        if not query:
            self.answer_output.setPlainText("Please enter a question.")
            return

        results = self.retriever.search(query, top_k=3)

        short_answer = build_short_answer(results)
        self.answer_output.setPlainText(short_answer)

        if not results:
            self.passages_output.setPlainText("No relevant passages found.")
            return

        text_blocks = []
        for i, result in enumerate(results, start=1):
            text_blocks.append(
                f"[Result {i}] Page {result['page']} | Score: {result['score']:.3f}\n"
                f"{result['text']}\n"
            )

        self.passages_output.setPlainText("\n\n".join(text_blocks))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PdfAssistantApp()
    window.show()
    sys.exit(app.exec())