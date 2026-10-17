# PTX 轻量解析器:文本 → 模块 IR(仅保留降级所需的结构,其余声明原样保留)
# 设计原则:不追求完整 PTX 语法,只保证"能渲染回 ptxas 可接受的等价文本"。
import re


class UnsupportedError(Exception):
    """遇到无法降级的指令,附带人话说明。"""


class Label:
    def __init__(self, name):
        self.name = name

    def render(self):
        return f"{self.name}:"


class Inst:
    """pred 形如 '%p1' 或 '!%p1'(含否定,无前导 @);opcode 含全部修饰符;
    operands 为顶层逗号切分的字符串列表。"""
    def __init__(self, pred, opcode, operands, line_no=None):
        self.pred = pred
        self.opcode = opcode
        self.operands = operands
        self.line_no = line_no

    def render(self):
        ops = ", ".join(self.operands)
        pred = f"@{self.pred} " if self.pred else ""
        return f"{pred}{self.opcode} {ops};".replace(" ;", ";")


class Entry:
    def __init__(self, name, params, body):
        self.name = name
        self.params = params          # 括号内文本(原样)
        self.body = body              # list[Label | Inst | 原样声明字符串]


LABEL_RE = re.compile(r"^([A-Za-z_$][\w.$]*)\s*:\s*(.*)$")
PRED_RE = re.compile(r"^@(!?%[\w.]+)\s+(.*)$")
ENTRY_FULL_RE = re.compile(r"^\s*(?:\.visible\s+)?\.entry\s+([\w.$]+)\s*\((.*)\)\s*\{?\s*$")
ENTRY_OPEN_RE = re.compile(r"^\s*(?:\.visible\s+)?\.entry\s+([\w.$]+)\s*\(")


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return "\n".join(re.sub(r"//.*$", "", ln) for ln in text.splitlines())


def split_top_commas(s):
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def parse_inst(line, line_no):
    m = PRED_RE.match(line)
    pred = None
    if m:
        pred, line = m.group(1), m.group(2)
    line = line.strip().rstrip(";").strip()
    if not line:
        return None
    toks = line.split(None, 1)
    opcode = toks[0]
    operands = split_top_commas(toks[1]) if len(toks) > 1 else []
    return Inst(pred, opcode, operands, line_no)


def _parse_body(lines, i):
    """从 '{' 的下一行开始,吃到独立 '}'。返回 (body, 下一行下标)。
    指令操作数可能被 ptxas/cuobjdump 换行断开(下一行以 ',' 开头或本行
    结尾是 ','),把续行拼接后再解析。"""
    body = []
    pending = None          # 跨行累积的指令文本
    while i < len(lines):
        bl = lines[i].strip()
        i += 1
        if bl == "}":
            if pending:
                inst = parse_inst(pending, i)
                if inst:
                    body.append(inst)
                pending = None
            break
        if not bl:
            if pending:
                inst = parse_inst(pending, i)
                if inst:
                    body.append(inst)
                pending = None
            continue
        lm = LABEL_RE.match(bl)
        is_label = lm and not bl.startswith(".") and not lm.group(2).startswith(":")
        if is_label and not pending:
            body.append(Label(lm.group(1)))
            rest = lm.group(2).strip()
            if rest and rest != ";":
                inst = parse_inst(rest, i)
                if inst:
                    body.append(inst)
        elif bl.startswith("."):
            if pending:
                inst = parse_inst(pending, i)
                if inst:
                    body.append(inst)
                pending = None
            body.append(bl)              # 声明原样保留
        else:
            pending = (pending + " " + bl) if pending else bl
            if bl.endswith(","):         # 操作数列表被换行,继续累积
                continue
            inst = parse_inst(pending, i)
            if inst:
                body.append(inst)
            pending = None
    return body, i


def parse_module(text):
    text = strip_comments(text)
    lines = text.splitlines()
    mod = {"version": None, "target": None, "addr_size": None,
           "header": [], "entries": []}

    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        i += 1
        if not ln:
            continue
        if ln.startswith(".version"):
            mod["version"] = ln.split()[1]
            continue
        if ln.startswith(".target"):
            mod["target"] = ln.split()[1]
            continue
        if ln.startswith(".address_size"):
            mod["addr_size"] = ln.split()[1]
            continue

        m = ENTRY_FULL_RE.match(ln)
        if m:
            name, params = m.group(1), m.group(2).strip()
            has_open_brace = ln.rstrip().endswith("{") or "{" in ln
        else:
            m2 = ENTRY_OPEN_RE.match(ln)
            if not m2:
                mod["header"].append(ln)  # 其它模块级声明原样保留
                continue
            name = m2.group(1)
            buf = ln[m2.end():]
            # 累积直到 ')' 与 '{' 都出现
            while (")" not in buf or "{" not in buf) and i < len(lines):
                buf += " " + lines[i].strip()
                i += 1
            params = buf[:buf.rindex(")")].strip()
            has_open_brace = "{" in buf

        if not has_open_brace:            # '{' 独占一行
            while i < len(lines) and "{" not in lines[i]:
                i += 1
            i += 1
        body, i = _parse_body(lines, i)
        mod["entries"].append(Entry(name, params, body))
    return mod


def render_module(mod):
    out = [f".version {mod['version']}",
           f".target {mod['target']}",
           f".address_size {mod['addr_size']}"]
    out.extend(mod["header"])
    for e in mod["entries"]:
        out.append(f".visible .entry {e.name}({e.params}) {{")
        for x in e.body:
            out.append(x.render() if isinstance(x, (Inst, Label)) else x)
        out.append("}")
    return "\n".join(out) + "\n"
