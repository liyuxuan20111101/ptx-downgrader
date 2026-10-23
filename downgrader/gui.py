# 图形界面:开箱即用的 PTX 降级器。
# 功能:选 fatbin/DLL/PTX 文件 → 选目标架构 → 一键降级 → ptxas 校验
#       → 按 kernel 统计 → 导出全部产物。
# 仅用标准库 tkinter,无第三方依赖。
import os
import sys
import threading
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

TARGET_LABELS = {
    "sm_61": "sm_61 — Pascal (GTX 10系, 驱动 470+)",
    "sm_70": "sm_70 — Volta",
    "sm_75": "sm_75 — Turing (GTX 16/20系)",
    "sm_80": "sm_80 — Ampere (A100)",
    "sm_86": "sm_86 — Ampere (RTX 30系)",
    "sm_89": "sm_89 — Ada (RTX 40系)",
}


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
        self.root.title("PTX 降级器 — 老显卡跑新 CUDA 程序")
        self.root.geometry("880x640")
        self._build()

    # ---------------------------------------------------------------- UI
    def _build(self):
        pad = {"padx": 8, "pady": 4}
        top = ttk.Frame(self.root)
        top.pack(fill="x", **pad)

        ttk.Label(top, text="输入文件:").grid(row=0, column=0, sticky="w")
        self.var_file = tk.StringVar()
        ttk.Entry(top, textvariable=self.var_file).grid(
            row=0, column=1, sticky="ew", padx=4)
        ttk.Button(top, text="浏览…", command=self.pick_file).grid(row=0, column=2)
        top.columnconfigure(1, weight=1)
        ttk.Label(
            top,
            text="支持: 含 CUDA fatbin 的 .dll/.so/.cubin/.fatbin,或单个 .ptx",
            foreground="#666").grid(row=1, column=1, columnspan=2, sticky="w")

        row2 = ttk.Frame(self.root)
        row2.pack(fill="x", **pad)
        ttk.Label(row2, text="目标架构:").pack(side="left")
        self.var_target = tk.StringVar(value="sm_61")
        cb = ttk.Combobox(row2, textvariable=self.var_target, state="readonly",
                          width=38,
                          values=[TARGET_LABELS[t] for t in SUPPORTED_TARGETS])
        cb.current(0)
        cb.pack(side="left", padx=6)
        self.var_validate = tk.BooleanVar(value=True)
        ttk.Checkbutton(row2, text="ptxas 校验", variable=self.var_validate)\
            .pack(side="left", padx=6)

        self.btn_run = ttk.Button(self.root, text="开始降级", command=self.run)
        self.btn_run.pack(fill="x", padx=8, pady=6)

        self.progress = ttk.Progressbar(self.root, mode="determinate")
        self.progress.pack(fill="x", padx=8)

        ttk.Label(self.root, text="日志 / 结果:").pack(anchor="w", padx=8)
        self.log = tk.Text(self.root, height=20, state="disabled",
                           font=("Consolas", 9), bg="#111", fg="#ddd")
        self.log.pack(fill="both", expand=True, padx=8, pady=4)

    # ------------------------------------------------------------- helpers
    def say(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def pick_file(self):
        path = filedialog.askopenfilename(
            title="选择输入文件",
            filetypes=[("CUDA 相关", "*.dll *.so *.cubin *.fatbin *.ptx"),
                       ("所有文件", "*.*")])
        if path:
            self.var_file.set(path)

    def _target(self):
        return self.var_target.get().split(" —")[0]

    # ----------------------------------------------------------------- run
    def run(self):
        path = self.var_file.get().strip()
        if not path:
            messagebox.showwarning("提示", "请先选择输入文件")
            return
        self.btn_run.configure(state="disabled")
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        threading.Thread(target=self._work, args=(path,), daemon=True).start()

    def _work(self, path):
        try:
            self._pipeline(path)
        except SystemExit as e:
            self.say(f"[错误] {e}")
        except UnsupportedError as e:
            self.say(f"[不支持的指令] {e}")
        except Exception as e:                       # noqa: BLE001 — GUI 兜底
            self.say(f"[错误] {type(e).__name__}: {e}")
        finally:
            self.root.after(0, lambda: self.btn_run.configure(state="normal"))

    def _pipeline(self, path):
        target = self._target()
        self.say(f"目标架构: {target}")
        # 产物固定落在输入文件同目录(用户刚用文件对话框选出的真实路径)。
        in_dir = os.path.dirname(os.path.realpath(os.path.abspath(path)))
        ext = os.path.splitext(path)[1].lower()
        tasks = []          # (kernel_name, ptx_text, out_path)

        if ext == ".ptx":
            name = _safe_name(os.path.splitext(os.path.basename(path))[0])
            text = open(path, encoding="utf-8", errors="replace").read()
            out = _safe_join(in_dir, f"{name}.{target[3:]}.ptx")
            tasks.append((name, text, out))
        else:
            self.say("提取 fatbin 内嵌 PTX …")
            blobs = extract_ptx(path)
            if not blobs:
                self.say("[警告] 该文件内没有嵌入 PTX 段(可能是纯 SASS 构建),"
                         "无法降级。请用含 -ptx 的构建产物。")
                return
            for arch, text in blobs:
                self.say(f"  发现 PTX 段: {arch} "
                         f"({len(split_modules(text))} 个 kernel)")
                for i, mtext in enumerate(split_modules(text)):
                    km = _safe_name(_kernel_name(mtext) or f"kernel_{arch}_{i}")
                    out = _safe_join(in_dir, f"downgraded_{arch}_{km}.ptx")
                    tasks.append((f"{arch}/{km}", mtext, out))

        ok_all, fail = 0, 0
        self.progress.configure(maximum=max(len(tasks), 1), value=0)
        for name, text, out in tasks:
            try:
                mod = parse_module(text)
                if not mod["entries"]:
                    self.say(f"  [跳过] {name}: 无 .entry")
                    continue
                out_text, warns = downgrade_module(mod, target=target)
                with open(out, "w", encoding="utf-8") as f:
                    f.write(out_text)
                status = "[ok]"
                if self.var_validate.get():
                    good, err = validate(out, arch=target)
                    status = "[ok+ptxas]" if good else "[ptxas失败]"
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
                    self.say(f"      改写: {w}")
                if len(warns) > 5:
                    self.say(f"      … 及另外 {len(warns) - 5} 条改写")
            except UnsupportedError as e:
                fail += 1
                self.say(f"  [拒绝] {name}: {e}")
            self.progress.step(1)

        self.say("")
        self.say(f"完成: {ok_all} 成功, {fail} 失败/拒绝, 共 {len(tasks)} 个 kernel。")
        self.say(f"产物目录: {in_dir}")
        if fail:
            self.say("提示: 被拒绝的 kernel 依赖暂不支持降级的指令(详见上方原因),"
                     "其余 kernel 仍可正常使用。")


def _kernel_name(ptx_text):
    for ln in ptx_text.splitlines():
        s = ln.strip()
        if s.startswith(".visible .entry") or s.startswith(".entry"):
            return s.split()[2].lstrip("$") if len(s.split()) > 2 else None
    return None


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
