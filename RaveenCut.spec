# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for RaveenCut.
#
# Prerequisites before building:
#   1. Place ffmpeg.exe and ffprobe.exe in the vendor\ directory.
#      Download a Windows build from https://github.com/BtbN/FFmpeg-Builds/releases
#   2. pip install pyinstaller
#   3. Run:  pyinstaller RaveenCut.spec --clean
#
# Output: dist\RaveenCut\RaveenCut.exe  (one-folder bundle)

from PyInstaller.utils.hooks import collect_all, collect_data_files

# Collect all files from packages that PyInstaller can miss
torch_datas,   torch_bins,   torch_hidden   = collect_all("torch")
silero_datas,  silero_bins,  silero_hidden  = collect_all("silero_vad")
numpy_datas,   numpy_bins,   numpy_hidden   = collect_all("numpy")

block_cipher = None

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=[
        # Bundled FFmpeg — placed next to the exe inside sys._MEIPASS
        ("vendor/ffmpeg.exe",  "."),
        ("vendor/ffprobe.exe", "."),
        *torch_bins,
        *silero_bins,
        *numpy_bins,
    ],
    datas=[
        *torch_datas,
        *silero_datas,
        *numpy_datas,
    ],
    hiddenimports=[
        "torch",
        "torch.nn",
        "torch.nn.functional",
        "numpy",
        "silero_vad",
        "PyQt6.QtMultimedia",
        "PyQt6.QtMultimediaWidgets",
        *torch_hidden,
        *silero_hidden,
        *numpy_hidden,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RaveenCut",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # no terminal window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="RaveenCut",
)
