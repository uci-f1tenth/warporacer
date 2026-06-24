```
uv run .\main.py --no-use-wandb --maps-dir-str ./maps/ --no-interactive --no-live-viewer --record-every-iteration 100 --switch-map-iter 20 --iterations 2000 --num-envs 16384
```
Windows users needs cl.exe or c++ compiler for pytorch compiler, can use visual studio build tools and use build tool cmd prompt or add to vscode profile:
```
"terminal.integrated.profiles.windows": {
    "x64 Native Tools (MSVC)": {
        "path": "cmd.exe",
        "args": [
            "/k",
            "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvarsall.bat",
            "x64"
        ],
        "icon": "terminal-cmd",
        "color": "terminal.ansiBlue"
    }
}
```

linux/google colab works automatically so far

macos: /shrug no cuda for you