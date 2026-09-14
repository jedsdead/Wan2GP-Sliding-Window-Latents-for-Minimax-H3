"""Wan2GP plugin entry point for MiniMax H3 latent carry."""

import json
import os

import gradio as gr

from shared.utils.plugins import WAN2GPPlugin

from . import patches


PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
INSERT_AFTER_TARGET = "video_info_accordion"   # sibling immediately before generate_btn
TAG = "[sliding-window-latents]"

PlugIn_Name = "Sliding Window Latents"
PlugIn_Id = "SlidingWindowLatents"

DESCRIPTION = (
    "Carries MiniMax H3 latents across sliding windows instead of re-encoding the "
    "previous window's decoded pixels as history conditioning. Wan2GP already pins a run "
    "of previous frames as history; this replaces the pixel round trip used to "
    "build it. Only affects continued windows, so window 1 is never changed."
)

NOTE = (
    "Use this or the Sliding Window Anchor plugin, not both — running them "
    "together can add extra frames at the join."
)

# Anything the panel exposes.  Env vars still set the initial value; the saved
# settings file wins over them once the panel has been used.
TOGGLES = ("enable", "audio", "fix_coords", "moment_match", "diagnose",
           "colour")


class SlidingWindowLatentsPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PlugIn_Name
        self.version = patches.VERSION
        self.description = DESCRIPTION

        self.state_dir = os.path.join(PLUGIN_DIR, "state")
        os.makedirs(self.state_dir, exist_ok=True)
        self._settings_path = os.path.join(self.state_dir, "settings.json")
        self._load_settings()

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def setup_ui(self):
        patches.install()

    def post_ui_setup(self, components):
        # Model modules load lazily, so a failed install at setup_ui time is
        # normal and worth retrying here.
        patches.install()

        def build_panel():
            with gr.Accordion(self._panel_title(), open=False) as panel:
                gr.Markdown(DESCRIPTION)

                enable_cb = gr.Checkbox(
                    value=patches.CONFIG.enable,
                    label="Enable latent carry",
                )
                audio_cb = gr.Checkbox(
                    value=patches.CONFIG.audio,
                    label="Carry audio latents",
                    info="Also replaces the continuation audio encode.",
                )
                audio_ctx_dd = gr.Dropdown(
                    choices=[0.0, 0.5, 1.0, 2.0, 4.0],
                    value=float(patches.CONFIG.audio_context),
                    label="Audio context (seconds, experimental)",
                    info="0 uses Wan2GP's window, which is derived from the video "
                         "overlap — 0.75s at overlap 18. Longer gives the audio "
                         "more history without changing the video overlap. Needs "
                         "latent carry and the coordinate correction; the console "
                         "reports what was carried.",
                )
                latents_dd = gr.Dropdown(
                    choices=[7, 12, 17],
                    value=patches.CONFIG.latents or 7,
                    label="Carried latents",
                    info="7 covers 22 frames. Only these values are phase-aligned "
                         "to the model's 5-latent period.",
                )

                with gr.Accordion("Colour consistency (experimental)", open=False):
                    gr.Markdown(
                        "**Experimental.** Corrects grade drift on the carried "
                        "latents rather than the saved video, using a linear "
                        "approximation of the latent-to-RGB map. The arithmetic "
                        "is exact against that map, but the map is a preview "
                        "approximation of the real decoder, so results are not "
                        "guaranteed. Off by default. Watch the console: if it "
                        "reports the noise floor window after window, latent "
                        "carry has already removed the drift and this is not "
                        "earning its place.")
                    colour_cb = gr.Checkbox(
                        value=patches.CONFIG.colour,
                        label="Match colour across windows (experimental)",
                    )
                    colour_axes_cb = gr.CheckboxGroup(
                        choices=list(patches._COLOUR_AXES),
                        value=list(patches.CONFIG.colour_axes),
                        label="Axes",
                        info="Brightness and cast are offsets and rest on the "
                             "linear latent-to-RGB map lightly. Contrast and "
                             "saturation are gains and lean on it harder - "
                             "measure before trusting them.",
                    )
                    colour_match_dd = gr.Dropdown(
                        choices=["previous", "first"],
                        value=patches.CONFIG.colour_match,
                        label="Match against",
                        info="Previous window, or the opening one.",
                    )
                    colour_strength_sl = gr.Slider(
                        0.0, 1.0, value=patches.CONFIG.colour_strength, step=0.05,
                        label="Correction strength",
                    )

                with gr.Accordion("Testing options", open=False):
                    coords_cb = gr.Checkbox(
                        value=patches.CONFIG.fix_coords,
                        label="Coordinate correction",
                        info="Off reproduces the uncorrected 1-frame offset at the "
                             "join. Leave on except for A/B.",
                    )
                    moment_cb = gr.Checkbox(
                        value=patches.CONFIG.moment_match,
                        label="Moment matching",
                        info="Rescales carried latents onto the encoded "
                             "distribution. Costs one encode per window — the very "
                             "round trip this plugin avoids.",
                    )
                    diagnose_cb = gr.Checkbox(
                        value=patches.CONFIG.diagnose,
                        label="Diagnostics in console",
                        info="Logs how carried and encoded latents differ per "
                             "channel. Also costs one encode per window.",
                    )

                status_md = gr.Markdown(self._status_text())
                gr.Markdown(NOTE)

                timer = gr.Timer(3)

                controls = [enable_cb, audio_cb, audio_ctx_dd, latents_dd,
                            coords_cb, moment_cb, diagnose_cb,
                            colour_cb, colour_axes_cb, colour_match_dd,
                            colour_strength_sl]

                def _apply(enable, audio, audio_context, carried, fix_coords, moment_match, diagnose,
                           colour, colour_axes, colour_match, colour_strength):
                    patches.CONFIG.enable = bool(enable)
                    patches.CONFIG.audio = bool(audio)
                    patches.CONFIG.audio_context = float(audio_context)
                    patches.CONFIG.latents = int(carried)
                    patches.CONFIG.fix_coords = bool(fix_coords)
                    patches.CONFIG.moment_match = bool(moment_match)
                    patches.CONFIG.diagnose = bool(diagnose)
                    patches.CONFIG.colour = bool(colour)
                    patches.CONFIG.colour_axes = tuple(colour_axes or ())
                    patches.CONFIG.colour_match = str(colour_match)
                    patches.CONFIG.colour_strength = float(colour_strength)
                    self._save_settings()
                    return gr.update(label=self._panel_title()), self._status_text()

                for control in controls:
                    control.change(_apply, inputs=controls,
                                   outputs=[panel, status_md])

                def _poll():
                    return self._status_text(), gr.update(label=self._panel_title())

                timer.tick(_poll, inputs=[], outputs=[status_md, panel])

            return panel

        try:
            self.insert_after(
                target_component_id=INSERT_AFTER_TARGET,
                new_component_constructor=build_panel,
            )
        except Exception as error:
            # The panel is only controls and a readout; the patches are already
            # installed and keep working without it.
            print(f"{TAG} could not place the panel next to "
                  f"'{INSERT_AFTER_TARGET}' ({error}). The plugin still works; "
                  f"settings fall back to environment variables.")
        return {}

    # ------------------------------------------------------------------ #
    # panel display
    # ------------------------------------------------------------------ #

    def _panel_title(self):
        if not patches.CONFIG.enable:
            return "Sliding Window Latents (off)"
        extras = []
        if not patches.CONFIG.fix_coords:
            extras.append("uncorrected coords")
        if patches.CONFIG.moment_match:
            extras.append("moment matched")
        suffix = f" — {', '.join(extras)}" if extras else ""
        return f"Sliding Window Latents{suffix}"

    def _status_text(self):
        stats = patches.STATE.stats
        return (f"Windows carried: **{stats.get('engaged', 0)}** "
                f"&nbsp;·&nbsp; re-encoded: **{stats.get('fell_back', 0)}** "
                f"&nbsp;·&nbsp; audio carried: **{stats.get('audio_engaged', 0)}**  \n"
                f"Colour corrected: **{stats.get('colour_applied', 0)}**  \n"
                f"Last: {patches.STATE.last_message}")

    # ------------------------------------------------------------------ #
    # settings
    # ------------------------------------------------------------------ #

    def _load_settings(self):
        try:
            if not os.path.isfile(self._settings_path):
                return
            with open(self._settings_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            for name in TOGGLES:
                if name in data:
                    setattr(patches.CONFIG, name, bool(data[name]))
            if "latents" in data:
                patches.CONFIG.latents = int(data["latents"])
            if "audio_context" in data:
                patches.CONFIG.audio_context = float(data["audio_context"])
            if "colour_axes" in data:
                patches.CONFIG.colour_axes = tuple(data["colour_axes"])
            if "colour_match" in data:
                patches.CONFIG.colour_match = str(data["colour_match"])
            if "colour_strength" in data:
                patches.CONFIG.colour_strength = float(data["colour_strength"])
        except Exception as error:
            print(f"{TAG} could not load settings ({error}); using defaults.")

    def _save_settings(self):
        try:
            os.makedirs(self.state_dir, exist_ok=True)
            payload = {name: bool(getattr(patches.CONFIG, name)) for name in TOGGLES}
            payload["latents"] = int(patches.CONFIG.latents or 7)
            payload["audio_context"] = float(patches.CONFIG.audio_context)
            payload["colour_axes"] = list(patches.CONFIG.colour_axes)
            payload["colour_match"] = str(patches.CONFIG.colour_match)
            payload["colour_strength"] = float(patches.CONFIG.colour_strength)
            with open(self._settings_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception as error:
            print(f"{TAG} could not save settings: {error}")
