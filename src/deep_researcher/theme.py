# Copyright (c) 2026 C10 Hackathon G1
"""Visual system for the Gradio app: flat, minimalist, black type on white, green accent, with a
dark mode (near-black canvas, light type, the same green).

One palette, used by the Gradio theme (component chrome, light + ``_dark`` variants), the extra CSS
(layout details Gradio's theme variables do not reach, with ``.dark`` overrides) and the pipeline
boxes (``pipeline_view.CSS``). Dark mode is the default; the header toggle flips the ``dark`` class Gradio
itself uses and the choice is remembered in ``localStorage`` (see ``THEME_INIT_JS`` / ``THEME_TOGGLE_JS``).
"""
from __future__ import annotations

from typing import Any

PALETTE = {
    "bg": "#FFFFFF",
    "surface": "#FAFAFA",
    "ink": "#0A0A0A",
    "ink_2": "#3F3F46",
    "ink_3": "#71717A",
    "line": "#E4E4E7",
    "line_2": "#D4D4D8",
    # Matrix green (UstymUkhman/matrix-theme): #00CC00 is its editor foreground and most-used
    # colour, #88FF88 its bright accent, #008800 / #006600 / #003300 its deeper steps. Pure green
    # on white is unreadable as type, so text and links use #006600 from the same theme.
    "green": "#00CC00",
    "green_dark": "#88FF88",      # hover (the theme's bright accent)
    "green_soft": "#E6FFE6",
    "green_ink": "#006600",
    "red": "#DC2626",
    "red_soft": "#FEE2E2",
}
# Dark mode is the Matrix theme proper: its editor/sidebar blacks, green borders and green-tinted type.
DARK = {
    "bg": "#0C0C0C",              # editor.background
    "surface": "#080808",         # sideBar.background
    "panel": "#000000",           # activityBar / statusBar
    "ink": "#DDFFDD",             # editorCursor.foreground: readable near-white with a green cast
    "ink_2": "#BBFFBB",
    "ink_3": "#88FF88",
    "line": "#003300",            # editor.selectionBackground
    "line_2": "#006600",          # input.border
    "green": "#00CC00",
    "green_dark": "#88FF88",      # hover
    "green_soft": "#003300",
    "green_ink": "#88FF88",
}


def build_theme() -> Any:
    import gradio as gr
    from gradio.themes.utils import colors, sizes

    green = colors.Color(
        name="dr-matrix", c50="#E6FFE6", c100="#DDFFDD", c200="#BBFFBB", c300="#88FF88", c400="#33FF33",
        c500="#00CC00", c600="#00AA00", c700="#008800", c800="#006600", c900="#003300", c950="#001A00",
    )
    p, d = PALETTE, DARK
    return gr.themes.Base(
        primary_hue=green, secondary_hue=colors.zinc, neutral_hue=colors.zinc,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
        radius_size=sizes.radius_md, spacing_size=sizes.spacing_md, text_size=sizes.text_md,
    ).set(
        # canvas
        body_background_fill=p["bg"], body_text_color=p["ink"], body_text_color_subdued=p["ink_3"],
        background_fill_primary=p["bg"], background_fill_secondary=p["surface"],
        border_color_primary=p["line"], border_color_accent=p["green"],
        color_accent=p["green"], color_accent_soft=p["green_soft"],
        link_text_color=p["green_dark"],
        shadow_drop="none", shadow_drop_lg="none",
        # blocks / panels: flat, 1px lines, no shadows
        block_background_fill=p["bg"], block_border_width="1px", block_border_color=p["line"],
        block_shadow="none", block_radius="10px",
        block_title_text_color=p["ink_2"], block_title_text_weight="600",
        block_label_text_color=p["ink_3"],
        panel_background_fill=p["surface"], panel_border_width="1px",
        section_header_text_weight="600", accordion_text_color=p["ink"],
        # inputs
        input_background_fill=p["bg"], input_border_color=p["line_2"], input_border_width="1px",
        input_border_color_focus=p["green"], input_radius="8px", input_shadow="none",
        input_shadow_focus="0 0 0 3px rgba(0,204,0,.30)",
        # buttons: every button is a solid lime fill with black type (outskill.com's primary button)
        button_primary_background_fill=p["green"], button_primary_background_fill_hover=p["green_dark"],
        button_primary_text_color=p["ink"], button_primary_border_color=p["green"],
        button_secondary_background_fill=p["green"], button_secondary_background_fill_hover=p["green_dark"],
        button_secondary_text_color=p["ink"], button_secondary_border_color=p["green"],
        button_border_width="1.5px", button_large_radius="8px",
        # tables / toggles / slider
        table_border_color=p["line"], table_even_background_fill=p["bg"],
        table_odd_background_fill=p["surface"],
        checkbox_background_color_selected=p["green"], checkbox_border_color_selected=p["green"],
        slider_color=p["green"],
        layout_gap="12px", form_gap_width="0px",
        # ---- dark mode (same system, inverted canvas)
        body_background_fill_dark=d["bg"], body_text_color_dark=d["ink"],
        body_text_color_subdued_dark=d["ink_3"],
        background_fill_primary_dark=d["bg"], background_fill_secondary_dark=d["surface"],
        border_color_primary_dark=d["line"], border_color_accent_dark=d["green"],
        color_accent_soft_dark=d["green_soft"], link_text_color_dark=d["green_dark"],
        block_background_fill_dark=d["panel"], block_border_color_dark=d["line"],
        block_title_text_color_dark=d["ink_2"], block_label_text_color_dark=d["ink_3"],
        panel_background_fill_dark=d["surface"], accordion_text_color_dark=d["ink"],
        input_background_fill_dark=d["bg"], input_border_color_dark=d["line_2"],
        input_border_color_focus_dark=d["green"], input_shadow_focus_dark="0 0 0 3px rgba(0,204,0,.35)",
        button_primary_background_fill_dark=d["green"],
        button_primary_background_fill_hover_dark=d["green_dark"],
        button_primary_text_color_dark="#0A0A0A", button_primary_border_color_dark=d["green"],
        button_secondary_background_fill_dark=d["green"],
        button_secondary_background_fill_hover_dark=d["green_dark"],
        button_secondary_text_color_dark="#0A0A0A", button_secondary_border_color_dark=d["green"],
        table_border_color_dark=d["line"], table_even_background_fill_dark=d["bg"],
        table_odd_background_fill_dark=d["surface"],
        checkbox_background_color_selected_dark=d["green"], checkbox_border_color_selected_dark=d["green"],
        slider_color_dark=d["green"],
    )


CSS = f"""
:root {{ --dr-ink: {PALETTE['ink']}; --dr-ink-3: {PALETTE['ink_3']}; --dr-line: {PALETTE['line']};
         --dr-green: {PALETTE['green']}; --dr-green-soft: {PALETTE['green_soft']}; }}
.gradio-container {{ max-width: 1180px !important; margin: 0 auto !important; }}
#dr-header {{ display:flex; align-items:baseline; gap:12px; margin: 6px 0 2px 0; }}
#dr-header h1 {{ font-size: 24px; font-weight: 700; letter-spacing: -0.02em; margin: 0;
                  color: var(--dr-ink); }}
#dr-header h1::before {{ content:""; display:inline-block; width:12px; height:12px; border-radius:50%;
                         background: var(--dr-green); margin-right:10px; vertical-align: 1px; }}
#dr-sub {{ color: var(--dr-ink-3); font-size: 13.5px; margin: 0 0 10px 0; }}
#dr-sub p {{ margin: 0; }}
/* Gradio 6 tabs: the visible strip is .tab-container > button[role=tab]; .selected marks the active one */
.tab-container {{ border-bottom: 1px solid var(--dr-line) !important; }}
.tab-container button[role="tab"] {{ font-weight: 600; color: var(--dr-ink-3) !important;
                                     box-shadow: none !important; }}
.tab-container button[role="tab"].selected {{ color: var(--dr-ink) !important;
                                              box-shadow: inset 0 -2px 0 var(--dr-green) !important; }}
.dr-step {{ font-size: 11px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
            color: var(--dr-ink-3); margin: 14px 0 4px 0; }}
.dr-step p {{ margin: 0; }}
#dr-busy {{ display:flex; align-items:center; min-height: 44px; font-size: 13px; color: var(--dr-ink-3); }}
#dr-busy p {{ margin: 0; }}
#dr-activity textarea {{ font-family: var(--font-mono); font-size: 12px; line-height: 1.5;
                          background: {PALETTE['surface']}; color: {PALETTE['ink_2']}; }}
#dr-report {{ border: 1px solid var(--dr-line); border-radius: 10px; padding: 18px 22px; background: #fff; }}
#dr-report h1 {{ font-size: 22px; letter-spacing: -0.02em; }}
#dr-report h2 {{ font-size: 16px; margin-top: 22px; padding-bottom: 4px;
                  border-bottom: 1px solid var(--dr-line); }}
#dr-report h3 {{ font-size: 14px; }}
#dr-report table {{ font-size: 12.5px; }}
#dr-report a {{ color: {PALETTE['green_dark']}; }}
#dr-summary table, #dr-runs table {{ font-size: 12.5px; }}
#dr-summary td:first-child {{ white-space: nowrap; }}
#dr-downloads button {{ border: 1.5px solid var(--dr-green) !important; color: {PALETTE['ink']} !important;
                         background: var(--dr-green) !important; }}
#dr-downloads button:hover {{ background: {PALETTE['green_dark']} !important; }}
footer {{ display: none !important; }}
#dr-footer {{ margin-top: 28px; padding-top: 12px; border-top: 1px solid var(--dr-line);
              color: var(--dr-ink-3); font-size: 12px; text-align: center; }}
#dr-footer p {{ margin: 0; }}
#dr-theme-toggle {{ min-width: 0; max-width: 150px; margin-left: auto; }}

/* ---- dark mode: Gradio adds the `dark` class to <body>; the toggle flips it ---- */
.dark #dr-header h1 {{ color: {DARK['ink']}; }}
.dark #dr-sub, .dark .dr-step, .dark #dr-busy, .dark #dr-footer {{ color: {DARK['ink_3']}; }}
.dark #dr-footer {{ border-top-color: {DARK['line']}; }}
.dark .tab-container {{ border-bottom-color: {DARK['line']} !important; }}
.dark .tab-container button[role="tab"] {{ color: {DARK['ink_3']} !important; }}
.dark .tab-container button[role="tab"].selected {{ color: {DARK['ink']} !important;
                                                    box-shadow: inset 0 -2px 0 {DARK['green']} !important; }}
.dark #dr-activity textarea {{ background: {DARK['surface']}; color: {DARK['ink_2']}; }}
.dark #dr-report {{ background: {DARK['panel']}; border-color: {DARK['line']}; color: {DARK['ink']}; }}
.dark #dr-report h2 {{ border-bottom-color: {DARK['line']}; }}
.dark #dr-report a {{ color: {DARK['green_dark']}; }}
.dark #dr-downloads button {{ background: {DARK['green']} !important; color: #0A0A0A !important;
                              border-color: {DARK['green']} !important; }}
.dark #dr-downloads button:hover {{ background: {DARK['green_dark']} !important; }}
"""

# Runs once when the page loads: dark unless the user chose light; sets the toggle label.
THEME_INIT_JS = """
() => {
  const pref = localStorage.getItem('dr-theme');
  const dark = pref !== 'light';
  document.body.classList.toggle('dark', dark);
  document.querySelector('.gradio-container')?.classList.toggle('dark', dark);
  return dark ? '☀ Light mode' : '☾ Dark mode';
}
"""

# Runs on the toggle button: flip the class, remember it, relabel the button.
THEME_TOGGLE_JS = """
() => {
  const dark = !document.body.classList.contains('dark');
  document.body.classList.toggle('dark', dark);
  document.querySelector('.gradio-container')?.classList.toggle('dark', dark);
  localStorage.setItem('dr-theme', dark ? 'dark' : 'light');
  return dark ? '☀ Light mode' : '☾ Dark mode';
}
"""
