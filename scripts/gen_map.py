"""Generate MAP.md — a token-cheap navigation index of the codebase.

For every .py under src/ and scripts/, emit its purpose (module docstring),
line count, and top-level symbols (functions, classes, ALLCAPS constants) with
1-indexed line numbers + compact signatures.  An agent reads one ~250-line file
to locate any symbol, then `Read(path, offset, limit)` only the slice it needs.

Deterministic and timestamp-free: regenerating an unchanged tree rewrites an
identical MAP.md (no spurious diff).  Run: `uv run python scripts/gen_map.py`.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["src/rehab_sci", "scripts"]
DOC_MAX = 80  # truncate one-line summaries


def _sig(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    a = node.args
    names = [arg.arg for arg in [*a.posonlyargs, *a.args]]
    if a.vararg:
        names.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        names.append("*")
    names += [arg.arg for arg in a.kwonlyargs]
    if a.kwarg:
        names.append("**" + a.kwarg.arg)
    return f"{node.name}({', '.join(names)})"


def _is_callback(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for d in node.decorator_list:
        target = d.func if isinstance(d, ast.Call) else d
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", None)
        if name in {"callback", "app.callback"}:
            return True
    return False


def _doc1(node: ast.AST) -> str:
    doc = ast.get_docstring(node)
    if not doc:
        return ""
    first = doc.strip().splitlines()[0].strip()
    return first[:DOC_MAX] + "…" if len(first) > DOC_MAX else first


def _symbols(tree: ast.Module) -> list[str]:
    rows: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            tag = " [callback]" if _is_callback(node) else ""
            doc = _doc1(node)
            rows.append(f"- L{node.lineno} `{_sig(node)}`{tag}" + (f" — {doc}" if doc else ""))
        elif isinstance(node, ast.ClassDef):
            methods = [
                n.name for n in node.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
            ]
            doc = _doc1(node)
            head = f"- L{node.lineno} `class {node.name}`" + (f" — {doc}" if doc else "")
            rows.append(head)
            if methods:
                rows.append(f"    methods: {', '.join(methods)}")
        else:  # ALLCAPS public module constants only
            targets = (
                [node.target]
                if isinstance(node, ast.AnnAssign)
                else node.targets
                if isinstance(node, ast.Assign)
                else []
            )
            for t in targets:
                if isinstance(t, ast.Name) and t.id.isupper() and not t.id.startswith("_"):
                    rows.append(f"- L{node.lineno} `{t.id}` (const)")
    return rows


def main() -> None:
    out: list[str] = [
        "# MAP.md — generated code map (do not edit by hand)",
        "",
        "Regenerate after structural changes: `uv run python scripts/gen_map.py`.",
        "Line numbers are 1-indexed — slice with `Read(path, offset, limit)` instead of",
        "reading whole files.  Sources: " + ", ".join(TARGETS) + ".",
        "",
    ]

    files: list[Path] = []
    for target in TARGETS:
        files += sorted((ROOT / target).rglob("*.py"))
    files = [f for f in files if "__pycache__" not in f.parts]

    total_lines = 0
    by_dir: dict[str, list[Path]] = {}
    for f in files:
        rel = f.relative_to(ROOT)
        by_dir.setdefault(str(rel.parent), []).append(f)

    for d in sorted(by_dir):
        out.append(f"## {d}")
        out.append("")
        for f in by_dir[d]:
            rel = f.relative_to(ROOT)
            text = f.read_text()
            nlines = text.count("\n") + (0 if text.endswith("\n") or not text else 1)
            total_lines += nlines
            tree = ast.parse(text)
            rows = _symbols(tree)
            mod_doc = _doc1(tree)
            out.append(f"### {rel.name} ({nlines} lines)")
            if mod_doc:
                out.append(mod_doc)
            out.extend(rows if rows else ["- (no top-level symbols)"])
            out.append("")

    out.insert(
        5,
        f"Index: {len(files)} files, {total_lines} source lines.",
    )
    (ROOT / "MAP.md").write_text("\n".join(out).rstrip() + "\n")
    print(f"wrote MAP.md: {len(files)} files, {total_lines} lines")


if __name__ == "__main__":
    main()
