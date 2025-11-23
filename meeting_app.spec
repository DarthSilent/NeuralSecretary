# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# --- Настройка путей и файлов ---
# Добавляем бинарники ffmpeg
binaries = [
    ('ffmpeg.exe', '.'), 
    ('ffprobe.exe', '.')
]

# Добавляем файлы данных
datas = [
    ('USER_MANUAL.md', '.'),
    ('icon.png', '.')
]

# Скрытые импорты, которые PyInstaller может не найти сам
hiddenimports = [
    'pyannote.audio',
    'speechbrain',
    'torch',
    'torchaudio',
    'sklearn.utils._cython_blas',
    'sklearn.neighbors.typedefs',
    'sklearn.neighbors.quad_tree',
    'sklearn.tree',
    'sklearn.tree._utils',
    'scipy.special.cython_special',
    'scipy.spatial.transform._rotation_groups',
    'customtkinter',
    'pydub'
]

# Автоматический сбор зависимостей для сложных пакетов
packages_to_collect = ['pyannote.audio', 'speechbrain', 'faster_whisper', 'customtkinter']

for package in packages_to_collect:
    try:
        tmp_ret = collect_all(package)
        datas += tmp_ret[0]
        binaries += tmp_ret[1]
        hiddenimports += tmp_ret[2]
    except Exception as e:
        print(f"Warning: Could not collect {package}: {e}")

a = Analysis(
    ['meeting_app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
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
    name='MeetingApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # False = без консольного окна (GUI)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if os.path.exists('icon.ico') else None
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MeetingApp',
)
