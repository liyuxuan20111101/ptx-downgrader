# 图形界面:开箱即用的 PTX 降级器。
# 功能:选 fatbin/DLL/PTX 文件 → 自动检测显卡选目标 → 一键降级 → ptxas 校验
#       → 按 kernel 统计 → 打开产物目录。
# 仅用标准库(ctypes 直调 CUDA driver API 检测显卡),无第三方依赖。
import ctypes
import json
import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from downgrader.engine import downgrade_module, SUPPORTED_TARGETS   # noqa: E402
from downgrader.extract import extract_ptx, split_modules           # noqa: E402
from downgrader.parser import parse_module, UnsupportedError        # noqa: E402
from downgrader.validate import validate                            # noqa: E402

SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".ptx_downgrader_gui.json")

TARGET_ORDER = ("sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_89")

TARGET_DESC = {
    "zh": {
        "sm_61": "sm_61 — Pascal (GTX 10系, 驱动 470+)",
        "sm_70": "sm_70 — Volta",
        "sm_75": "sm_75 — Turing (GTX 16/20系)",
        "sm_80": "sm_80 — Ampere (A100)",
        "sm_86": "sm_86 — Ampere (RTX 30系)",
        "sm_89": "sm_89 — Ada (RTX 40系)",
    },
    "en": {
        "sm_61": "sm_61 — Pascal (GTX 10-series, driver 470+)",
        "sm_70": "sm_70 — Volta",
        "sm_75": "sm_75 — Turing (GTX 16/20-series)",
        "sm_80": "sm_80 — Ampere (A100)",
        "sm_86": "sm_86 — Ampere (RTX 30-series)",
        "sm_89": "sm_89 — Ada (RTX 40-series)",
    },
}

# {key: (中文, English)}
T = {
    "title":     ("PTX 降级器 — 老显卡跑新 CUDA 程序",
                  "PTX Downgrader — run new CUDA programs on older GPUs"),
    "file":      ("输入文件:", "Input file:"),
    "browse":    ("浏览…", "Browse…"),
    "hint":      ("支持: 含 CUDA fatbin 的 .dll/.so/.cubin/.fatbin,或单个 .ptx",
                  "Supports .dll/.so/.cubin/.fatbin with embedded CUDA fatbin, or a single .ptx"),
    "target":    ("目标架构:", "Target architecture:"),
    "validate":  ("ptxas 校验", "ptxas validation"),
    "run":       ("开始降级", "Downgrade"),
    "running":   ("降级中…", "Running…"),
    "log":       ("日志 / 结果:", "Log / results:"),
    "open_dir":  ("打开产物目录", "Open output folder"),
    "gpu_none":  ("未检测到 NVIDIA 驱动 — 仍可降级,请自行选择目标架构",
                  "No NVIDIA driver detected — downgrade still works, pick the target manually"),
    "gpu_too_old": ("显卡 {} 是 sm_{},低于本项目最低支持架构 sm_61",
                    "GPU {} is sm_{}, below the minimum supported sm_61"),
    "not_found": ("文件不存在,请重新选择", "File not found, please pick again"),
    "pick_first": ("请先选择输入文件", "Pick an input file first"),
    "goal":      ("目标架构: {}", "Target: {}"),
    "extracting": ("提取 fatbin 内嵌 PTX …", "Extracting embedded PTX from fatbin…"),
    "no_ptx":    ("[警告] 该文件内没有嵌入 PTX 段(可能是纯 SASS 构建),无法降级。"
                  "请用含 -ptx 的构建产物。",
                  "[WARN] No embedded PTX in this file (SASS-only build?). "
                  "Rebuild with -ptx embedded."),
    "found":     ("  发现 PTX 段: {} ({} 个 kernel)",
                  "  Found PTX section: {} ({} kernels)"),
    "skip":      ("  [跳过] {}: 无 .entry", "  [SKIP] {}: no .entry"),
    "ok_ptxas":  ("[ok+ptxas]", "[ok+ptxas]"),
    "ok":        ("[ok]", "[ok]"),
    "ptxas_fail": ("[ptxas失败]", "[ptxas FAILED]"),
    "reject":    ("[拒绝]", "[REJECTED]"),
    "rewrite":   ("      改写: {}", "      rewrite: {}"),
    "more_warns": ("      … 及另外 {} 条改写", "      … and {} more rewrites"),
    "done":      ("完成: {} 成功, {} 失败/拒绝, 共 {} 个 kernel。耗时 {:.1f}s",
                  "Done: {} ok, {} failed/rejected, {} kernels total. {:.1f}s"),
    "outdir":    ("产物目录: {}", "Output folder: {}"),
    "reject_hint": ("提示: 被拒绝的 kernel 依赖暂不支持降级的指令(详见上方原因),"
                    "其余 kernel 仍可正常使用。",
                    "Note: rejected kernels use instructions the downgrader can't "
                    "expand yet (reasons above); the rest still work."),
    "err":       ("[错误]", "[ERROR]"),
    "unsupported": ("[不支持的指令]", "[UNSUPPORTED]"),
    "lang":      ("语言/Language:", "语言/Language:"),
}


def detect_gpu():
    """返回 (GPU 名, 'sm_XX') 或 None。零依赖:ctypes 直调 CUDA driver API。"""
    try:
        cuda = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" \
            else ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None
    if cuda.cuInit(0) != 0:
        return None
    dev = ctypes.c_int()
    if cuda.cuDeviceGet(ctypes.byref(dev), 0) != 0:
        return None
    # cuDeviceGetName 返回 char*(ANSI/UTF-8),须用 string buffer 读取
    buf = ctypes.create_string_buffer(256)
    if cuda.cuDeviceGetName(buf, 256, dev) != 0:
        return None
    name = buf.value.decode("utf-8", errors="replace")
    major, minor = ctypes.c_int(), ctypes.c_int()
    cuda.cuDeviceGetAttribute(ctypes.byref(major), 75, dev)   # CC major
    cuda.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev)   # CC minor
    return name, f"sm_{major.value}{minor.value}"


def _safe_name(s):
    """把任意来源的名字(kernel 名/文件名)压成文件名安全子集。"""
    s = "".join(c if c.isalnum() or c in "._" else "_" for c in s)
    return s.strip("._") or "unnamed"


def _safe_join(directory, filename):
    """目录 + 已清洗的文件名拼接;规范化后校验仍在目录内,防任何穿越。"""
    out = os.path.realpath(os.path.join(directory, filename))
    base = os.path.realpath(directory)
    if out != base and not out.startswith(base + os.sep):
        raise ValueError(f"输出路径越界: {out}")
    return out


class App:
    def __init__(self, root):
        self.root = root
        self.lang = "zh"
        self.out_dir = None
        self._load_settings()
        self._build()
        self._apply_lang()
        threading.Thread(target=self._detect_gpu_worker, daemon=True).start()

    # ------------------------------------------------------------- settings
    def _load_settings(self):
        try:
            with open(SETTINGS_PATH, encoding="utf-8") as f:
                s = json.load(f)
            self._saved_input = s.get("input", "")
            self._saved_target = s.get("target", "")
            self._saved_validate = s.get("validate", True)
            self.lang = s.get("lang", "zh")
        except (OSError, ValueError):
            self._saved_input, self._saved_target = "", ""
            self._saved_validate = True

    def _save_settings(self):
        try:
            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump({"input": self.var_file.get().strip(),
                           "target": self._target(),
                           "validate": self.var_validate.get(),
                           "lang": self.lang}, f)
        except OSError:
            pass

    # ---------------------------------------------------------------- UI
    def _build(self):
        pad = {"padx": 8, "pady": 4}
        top = ttk.Frame(self.root)
        top.pack(fill="x", **pad)

        self.lbl_file = ttk.Label(top)
        self.lbl_file.grid(row=0, column=0, sticky="w")
        self.var_file = tk.StringVar(value=self._saved_input)
        ttk.Entry(top, textvariable=self.var_file).grid(
            row=0, column=1, sticky="ew", padx=4)
        self.btn_browse = ttk.Button(top, command=self.pick_file)
        self.btn_browse.grid(row=0, column=2)
        top.columnconfigure(1, weight=1)
        self.lbl_hint = ttk.Label(top, foreground="#666")
        self.lbl_hint.grid(row=1, column=1, columnspan=2, sticky="w")

        row2 = ttk.Frame(self.root)
        row2.pack(fill="x", **pad)
        self.lbl_target = ttk.Label(row2)
        self.lbl_target.pack(side="left")
        self.var_target = tk.StringVar(
            value=self._saved_target if self._saved_target in TARGET_ORDER
            else "sm_61")
        self.cb_target = ttk.Combobox(row2, textvariable=self.var_target,
                                      state="readonly", width=38)
        self.cb_target.pack(side="left", padx=6)
        self.var_validate = tk.BooleanVar(value=self._saved_validate)
        self.chk_validate = ttk.Checkbutton(row2, variable=self.var_validate)
        self.chk_validate.pack(side="left", padx=6)

        self.lbl_lang = ttk.Label(row2)
        self.lbl_lang.pack(side="right")
        self.cb_lang = ttk.Combobox(row2, state="readonly", width=6,
                                    values=["中文", "EN"])
        self.cb_lang.set("中文" if self.lang == "zh" else "EN")
        self.cb_lang.bind("<<ComboboxSelected>>", self._on_lang)
        self.cb_lang.pack(side="right", padx=(2, 6))

        self.btn_run = ttk.Button(self.root, command=self.run)
        self.btn_run.pack(fill="x", padx=8, pady=6)
        self.btn_open = ttk.Button(self.root, command=self.open_outdir,
                                   state="disabled")
        self.btn_open.pack(fill="x", padx=8, pady=(0, 6))

        self.progress = ttk.Progressbar(self.root, mode="determinate")
        self.progress.pack(fill="x", padx=8)

        self.lbl_log = ttk.Label(self.root)
        self.lbl_log.pack(anchor="w", padx=8)
        self.log = tk.Text(self.root, height=20, state="disabled",
                           font=("Consolas", 9), bg="#111", fg="#ddd")
        self.log.pack(fill="both", expand=True, padx=8, pady=4)

    def _on_lang(self, _event=None):
        self.lang = "zh" if self.cb_lang.get() == "中文" else "en"
        self._apply_lang()

    def _apply_lang(self):
        for key, w in (("title", self.root), ("file", self.lbl_file),
                       ("browse", self.btn_browse), ("hint", self.lbl_hint),
                       ("target", self.lbl_target),
                       ("validate", self.chk_validate),
                       ("run", self.btn_run), ("open_dir", self.btn_open),
                       ("log", self.lbl_log), ("lang", self.lbl_lang)):
            if key == "title":
                w.title(T[key][0 if self.lang == "zh" else 1])
            else:
                w.configure(text=T[key][0 if self.lang == "zh" else 1])
        # 架构下拉随语言重建,保持当前选择
        cur = self.var_target.get()
        self.cb_target["values"] = [TARGET_DESC[self.lang][t]
                                    for t in TARGET_ORDER]
        idx = TARGET_ORDER.index(cur) if cur in TARGET_ORDER else 0
        self.cb_target.current(idx)

    # ------------------------------------------------------------- helpers
    def tr(self, key):
        i = 0 if self.lang == "zh" else 1
        return T[key][i]

    def say(self, msg):
        self.root.after(0, self._say, msg)

    def _say(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def pick_file(self):
        path = filedialog.askopenfilename(
            title=self.tr("file"),
            filetypes=[("CUDA", "*.dll *.so *.cubin *.fatbin *.ptx"),
                       ("*", "*.*")])
        if path:
            self.var_file.set(path)

    def open_outdir(self):
        if not self.out_dir:
            return
        try:
            if os.name == "nt":
                os.startfile(self.out_dir)                 # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", self.out_dir])
            else:
                subprocess.Popen(["xdg-open", self.out_dir])
        except OSError as e:
            self.say(f"{self.tr('err')} {e}")

    def _target(self):
        return TARGET_ORDER[self.cb_target.current()]

    def _detect_gpu_worker(self):
        info = detect_gpu()
        if info:
            name, arch = info
            cc = int(arch[3:])
            floor = next((t for t in reversed(TARGET_ORDER)
                          if int(t[3:]) <= cc), None)
            self.root.after(0, self._on_gpu, name, arch, floor)
        else:
            self.say(self.tr("gpu_none"))

    def _on_gpu(self, name, arch, floor):
        if floor is None:
            self.say(self.tr("gpu_too_old").format(name, arch[3:]))
            return
        # 尊重用户上次保存的选择;首次使用(无设置文件)才自动选
        if self._saved_target in TARGET_ORDER:
            self.say(f"{name} ({arch})")
            return
        idx = TARGET_ORDER.index(floor)
        self.cb_target.current(idx)
        self.say(f"{name} ({arch}) → {floor}")

    # ----------------------------------------------------------------- run
    def run(self):
        path = self.var_file.get().strip()
        if not path:
            messagebox.showwarning(self.tr("title"), self.tr("pick_first"))
            return
        if not os.path.isfile(path):
            messagebox.showwarning(self.tr("title"), self.tr("not_found"))
            return
        self.btn_run.configure(state="disabled", text=self.tr("running"))
        self.btn_open.configure(state="disabled")
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        threading.Thread(target=self._work, args=(path,), daemon=True).start()

    def _work(self, path):
        t0 = time.time()
        try:
            self._pipeline(path)
        except SystemExit as e:
            self.say(f"{self.tr('err')} {e}")
        except UnsupportedError as e:
            self.say(f"{self.tr('unsupported')} {e}")
        except Exception as e:                       # noqa: BLE001 — GUI 兜底
            self.say(f"{self.tr('err')} {type(e).__name__}: {e}")
        finally:
            dt = time.time() - t0
            self.root.after(0, self._done, dt)

    def _done(self, dt):
        self._say(f"{self.tr('done')}".format(
            *getattr(self, "_stats", (0, 0, 0)), dt))
        self.btn_run.configure(state="normal", text=self.tr("run"))
        if self.out_dir:
            self.btn_open.configure(state="normal")
        self._save_settings()

    def _pipeline(self, path):
        target = self._target()
        self.say(self.tr("goal").format(target))
        # 产物固定落在输入文件同目录(用户刚用文件对话框选出的真实路径)。
        in_dir = os.path.dirname(os.path.realpath(os.path.abspath(path)))
        self.out_dir = in_dir
        ext = os.path.splitext(path)[1].lower()
        tasks = []          # (kernel_name, ptx_text, out_path)

        if ext == ".ptx":
            name = _safe_name(os.path.splitext(os.path.basename(path))[0])
            text = open(path, encoding="utf-8", errors="replace").read()
            out = _safe_join(in_dir, f"{name}.{target[3:]}.ptx")
            tasks.append((name, text, out))
        else:
            self.say(self.tr("extracting"))
            blobs = extract_ptx(path)
            if not blobs:
                self.say(self.tr("no_ptx"))
                return
            for arch, text in blobs:
                self.say(self.tr("found").format(
                    arch, len(split_modules(text))))
                # cuobjdump 拼接输出里同名 kernel 会出现,重名加序号防覆盖
                seen = {}
                for i, mtext in enumerate(split_modules(text)):
                    km = _safe_name(_kernel_name(mtext) or f"kernel_{arch}_{i}")
                    n = seen.get(km, 0)
                    seen[km] = n + 1
                    if n:
                        km = f"{km}_{n + 1}"
                    out = _safe_join(in_dir, f"downgraded_{arch}_{km}.ptx")
                    tasks.append((f"{arch}/{km}", mtext, out))

        ok_all, fail = 0, 0
        self.root.after(0, lambda: self.progress.configure(
            maximum=max(len(tasks), 1), value=0))
        for name, text, out in tasks:
            try:
                mod = parse_module(text)
                if not mod["entries"]:
                    self.say(self.tr("skip").format(name))
                    continue
                out_text, warns = downgrade_module(mod, target=target)
                with open(out, "w", encoding="utf-8") as f:
                    f.write(out_text)
                status = self.tr("ok")
                if self.var_validate.get():
                    good, err = validate(out, arch=target)
                    status = self.tr("ok_ptxas") if good \
                        else self.tr("ptxas_fail")
                    if not good:
                        fail += 1
                        self.say(f"  {status} {name}\n{err[:1200]}")
                    else:
                        ok_all += 1
                        self.say(f"  {status} {name}")
                else:
                    ok_all += 1
                    self.say(f"  {status} {name}")
                for w in warns[:5]:
                    self.say(self.tr("rewrite").format(w))
                if len(warns) > 5:
                    self.say(self.tr("more_warns").format(len(warns) - 5))
            except UnsupportedError as e:
                fail += 1
                self.say(f"  {self.tr('reject')} {name}: {e}")
            self.root.after(0, self.progress.step, 1)

        self._stats = (ok_all, fail, len(tasks))
        self.say(self.tr("outdir").format(in_dir))
        if fail:
            self.say(self.tr("reject_hint"))


def _kernel_name(ptx_text):
    for ln in ptx_text.splitlines():
        s = ln.strip()
        if s.startswith(".visible .entry") or s.startswith(".entry"):
            parts = s.split()
            idx = parts.index(".entry") + 1
            if len(parts) > idx:
                return parts[idx].split("(")[0].lstrip("$")
            return None
    return None


def main():
    root = tk.Tk()
    root.geometry("900x680")
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
