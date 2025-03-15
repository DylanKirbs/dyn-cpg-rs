import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from graph_utils import PY_PARSER, draw_graph, gen_highlighted_change_graph, plt

c1 = """
if x < y:
    print()
"""

c2 = """
if x . y:
    pass
"""


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

        # Left editor
        self.editor1 = tk.Text(
            self,
            wrap=tk.NONE,
            bg=self.text_bg,
            fg=self.dark_fg,
            insertbackground=self.dark_fg,
            selectbackground=self.highlight,
        )
        self.editor1.grid(row=0, column=0, sticky="nsew")

        self.label1 = tk.Label(
            self, relief=tk.SUNKEN, wraplength=800, bg=self.text_bg, fg=self.accent
        )
        self.label1.grid(row=1, column=0, sticky="ew")

        # Right editor
        self.editor2 = tk.Text(
            self,
            wrap=tk.NONE,
            bg=self.text_bg,
            fg=self.dark_fg,
            insertbackground=self.dark_fg,
            selectbackground=self.highlight,
        )
        self.editor2.grid(row=0, column=1, sticky="nsew")

        self.label2 = tk.Label(
            self, relief=tk.SUNKEN, wraplength=800, bg=self.text_bg, fg=self.accent
        )
        self.label2.grid(row=1, column=1, sticky="ew")

        # Changed ranges
        self.label3 = tk.Label(
            self, relief=tk.SUNKEN, wraplength=800, bg=self.text_bg, fg=self.dark_fg
        )
        self.label3.grid(row=2, column=0, columnspan=2, sticky="ew")

        # Matplotlib figure setup with dark theme
        self.figure = plt.figure(figsize=(10, 6), facecolor=self.dark_bg)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().configure(bg=self.dark_bg)
        self.canvas.get_tk_widget().grid(row=3, column=0, columnspan=2, sticky="nsew")
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(self.text_bg)

        # Bind text modifications
        self.editor1.bind("<<Modified>>", self.on_text_modified)
        self.editor2.bind("<<Modified>>", self.on_text_modified)

        # Set initial code
        self.editor1.insert("1.0", c1.strip())
        self.editor2.insert("1.0", c2.strip())

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)
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
        code1 = self.editor1.get("1.0", "end-1c")
        code2 = self.editor2.get("1.0", "end-1c")

        # Parse code
        orig_tree = PY_PARSER.parse(bytes(code1, "utf-8"))
        new_tree = PY_PARSER.parse(bytes(code2, "utf-8"))

        # Update labels
        self.label1.config(text=f"Original Root Node: {orig_tree.root_node}")
        self.label2.config(text=f"Modified Root Node: {new_tree.root_node}")
        self.label3.config(text=f"Changed Ranges: {orig_tree.changed_ranges(new_tree)}")

        # Generate and draw graph with dark theme
        self.ax.clear()
        graph = gen_highlighted_change_graph(orig_tree, new_tree)
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
