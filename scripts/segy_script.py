from pathlib import Path
import numpy as np
from filark.gui.app import run_app
from filark.io.tapeio import ArrayTape
from filark.gui.config import GuiConfig
from cigsegy import SegyNP

class SegyLoader:
    suffixes: tuple[str, ...] = (".sgy", ".segy", ".Sgy")
    
    def load(self, path):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        if p.is_dir():
            raise IsADirectoryError(
                "SegyLoader handles single file only; folder stitching is handled by FiLark."
            )
        d = SegyNP(str(p), ndim=2)

        return ArrayTape(d, dims="nt_nc")


cfg = GuiConfig(loaders=[SegyLoader()],
                x_role="trace",
                y_role="time",
                x_label="Trace",
                y_label="Time (ms)",
    )

run_app(cfg=cfg)
