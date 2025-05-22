import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from graph_utils import draw_graph, highlighted_graph
from tree_sitter import Parser, Range, Tree
import matplotlib.pyplot as plt
from ts_utils import Languages, Parsers
from difflib import SequenceMatcher

c1 = """
void foo() {
  int x = source();
  if (x < MAX) {
    int y = 2 * x;
    sink(y);
  }
}
"""

c2 = """
void foo() {
  int x = source();
  if (x < MAX) {
    int z = 2 * x;
    sink(z);
  }
}
"""


def source_edits(
    old_source: bytes, new_source: bytes
) -> list[tuple[int, int, int, int]]:
    """
    Perform a sequence match between the old and new source code and return
    the list of source code edits.

    :param old_source: The old source code
    :param new_source: The new source code
    :return: A list of tuples (old_start, old_end, new_start, new_end) representing the source code edits
    """

    edits = []
    for tag, old_start, old_end, new_start, new_end in SequenceMatcher(
        None, old_source, new_source
    ).get_opcodes():
        if tag == "equal":
            continue
        edits.append((old_start, old_end, new_start, new_end))

    return edits


class TSCompApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tree-Sitter Based AST Comparison Tool")
        self.geometry("1200x800")

        # Dark mode color scheme
        self.dark_bg = "#1e1e1e"
        self.dark_fg = "#ffffff"
        self.text_bg = "#2d2d2d"
        self.highlight = "#3e3e3e"
        self.accent = "#569cd6"

        # Configure matplotlib dark theme
        plt.style.use("dark_background")

        # Configure main window colors
        self.configure(bg=self.dark_bg)
        self._set_dark_theme()

        # Ensure proper cleanup
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        # Language selector dropdown
        self.lang_var = tk.StringVar()
        self.lang_var.set("c")
        self.lang_selector = ttk.Combobox(
            self,
            values=[k for k in Languages.supported_langs.keys()],
            textvariable=self.lang_var,
        )
        self.lang_selector.grid(row=0, column=0, columnspan=2, sticky="ew")

        # Left editor
        self.left_editor = tk.Text(
            self,
            wrap=tk.NONE,
            bg=self.text_bg,
            fg=self.dark_fg,
            insertbackground=self.dark_fg,
            selectbackground=self.highlight,
        )
        self.left_editor.grid(row=1, column=0, sticky="nsew")

        self.left_root_txt = tk.Label(
            self, relief=tk.SUNKEN, wraplength=800, bg=self.text_bg, fg=self.accent
        )
        self.left_root_txt.grid(row=2, column=0, sticky="ew")

        # Right editor
        self.right_editor = tk.Text(
            self,
            wrap=tk.NONE,
            bg=self.text_bg,
            fg=self.dark_fg,
            insertbackground=self.dark_fg,
            selectbackground=self.highlight,
        )
        self.right_editor.grid(row=1, column=1, sticky="nsew")

        self.right_root_txt = tk.Label(
            self, relief=tk.SUNKEN, wraplength=800, bg=self.text_bg, fg=self.accent
        )
        self.right_root_txt.grid(row=2, column=1, sticky="ew")

        # Changed ranges
        self.changed_ranges = tk.Label(
            self, relief=tk.SUNKEN, wraplength=800, bg=self.text_bg, fg=self.dark_fg
        )
        self.changed_ranges.grid(row=3, column=0, columnspan=2, sticky="ew")

        # Matplotlib figure setup with dark theme
        self.figure = plt.figure(figsize=(10, 6), facecolor=self.dark_bg)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().configure(bg=self.dark_bg)
        self.canvas.get_tk_widget().grid(row=4, column=0, columnspan=2, sticky="nsew")
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(self.text_bg)

        # Bind text modifications
        self.left_editor.bind("<<Modified>>", self.on_text_modified)
        self.right_editor.bind("<<Modified>>", self.on_text_modified)

        # Set initial code
        self.left_editor.insert("1.0", c1.strip())
        self.right_editor.insert("1.0", c2.strip())

        # Configure grid weights
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(4, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Initial update
        self.update_display()

    def _set_dark_theme(self):
        """Applies dark theme to all widgets"""
        ttk.Style().theme_use("clam")
        self.tk_setPalette(
            background=self.dark_bg,
            foreground=self.dark_fg,
            activeBackground=self.highlight,
            activeForeground=self.dark_fg,
        )

    def on_text_modified(self, event):
        if event.widget.edit_modified():
            self.update_display()
            event.widget.edit_modified(False)

    def update_display(self):
        # Get code from both editors
        code1 = self.left_editor.get("1.0", "end-1c")
        code2 = self.right_editor.get("1.0", "end-1c")

        # Parse code
        parser: Parser = getattr(Parsers, self.lang_var.get())

        old_source = bytes(code1, "utf-8")
        new_source = bytes(code2, "utf-8")

        orig_tree: Tree = parser.parse(old_source)

        changes = source_edits(old_source, new_source)
        for old_start, old_end, new_start, new_end in changes:
            sp = (0, 0)
            ep = orig_tree.root_node.end_point
            if node := orig_tree.root_node.descendant_for_byte_range(
                old_start, old_end
            ):
                sp = node.start_point
                ep = node.end_point
            orig_tree.edit(old_start, new_start, new_end, sp, ep, ep)

        new_tree: Tree = parser.parse(new_source, orig_tree)

        changed_ranges = set(orig_tree.changed_ranges(new_tree))

        # Update labels
        self.left_root_txt.config(text=f"Old Structure: {orig_tree.root_node}")
        self.right_root_txt.config(text=f"New Structure: {new_tree.root_node}")

        self.changed_ranges.config(
            text=f"Source Edits : {[(a,b) for (a,b,_,_) in changes]} | TS Changed Ranges: {[(r.start_byte,r.end_byte) for r in changed_ranges]}"
        )

        # Generate and draw graph with dark theme
        self.ax.clear()
        graph = highlighted_graph(
            new_tree,
            changed_ranges,
        )
        draw_graph(graph, self.ax, font_color="white")
        self.ax.set_facecolor(self.text_bg)
        cfg = self.ax.get_figure()
        cfg.set_facecolor(self.dark_bg) if cfg else None
        self.canvas.draw()

    def destroy(self):
        plt.close(self.figure)
        super().destroy()


if __name__ == "__main__":
    app = TSCompApp()
    app.mainloop()
