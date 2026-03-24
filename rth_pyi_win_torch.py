# PyInstaller runtime hook (see RaveenCut.spec). Runs before app.py.
# PyTorch 2.9+ on Windows can hit WinError 1114 loading c10.dll if other native
# libs (e.g. Qt) initialize first; load torch as early as possible (pytorch#166628).
import os
import sys

if sys.platform == "win32" and getattr(sys, "frozen", False):
    _meipass = getattr(sys, "_MEIPASS", None)
    if _meipass:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        _dll_dirs = []
        for _rel in ("", "torch/lib", "numpy.libs"):
            _d = os.path.join(_meipass, _rel) if _rel else _meipass
            if os.path.isdir(_d):
                _dll_dirs.append(_d)
        if _dll_dirs:
            os.environ["PATH"] = (
                os.pathsep.join(_dll_dirs)
                + os.pathsep
                + os.environ.get("PATH", "")
            )
        for _d in _dll_dirs:
            try:
                os.add_dll_directory(_d)
            except OSError:
                pass
        import torch  # noqa: F401
