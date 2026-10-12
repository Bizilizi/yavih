"""A small TOML reader for Python 3.10 environments without ``tomli`` (``yavih.config``'s last fallback).

Python 3.11+ uses the stdlib ``tomllib`` and an installed ``tomli`` is preferred over this module; it exists so a
``pip install --no-deps yavih`` into a 3.10 model env (no new dependency) can still read the site
config. It covers what a yavih config uses: ``[table]`` / ``[a.b]`` / ``[[array]]`` headers, bare, quoted and
dotted keys, basic and literal strings (single- and multi-line), integers (``_`` separators, ``0x``/``0o``/``0b``),
floats (``inf``/``nan``), booleans, arrays (multi-line, trailing comma) and inline tables. Dates and times are
not supported (a config has no use for them) and are reported as a decode error. A malformed document raises
``TOMLDecodeError`` (a ``ValueError``) naming the line. stdlib only.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["loads", "load", "TOMLDecodeError"]


class TOMLDecodeError(ValueError):
    """Malformed TOML (the name and base class of ``tomllib.TOMLDecodeError``)."""


_BARE = re.compile(r"[A-Za-z0-9_-]+")
_ESC = {"b": "\b", "t": "\t", "n": "\n", "f": "\f", "r": "\r", '"': '"', "\\": "\\"}
_INT = re.compile(r"[+-]?(0|[1-9](_?[0-9])*)$")
_FLOAT = re.compile(r"[+-]?(0|[1-9](_?[0-9])*)(\.[0-9](_?[0-9])*)?([eE][+-]?[0-9](_?[0-9])*)?$")


class _Parser:
    def __init__(self, text: str) -> None:
        self.s = text.replace("\r\n", "\n")
        self.i = 0

    # -- low level --------------------------------------------------------------------------
    def err(self, msg: str) -> TOMLDecodeError:
        line = self.s.count("\n", 0, self.i) + 1
        return TOMLDecodeError(f"{msg} (line {line})")

    def peek(self, n: int = 1) -> str:
        return self.s[self.i:self.i + n]

    def ws(self) -> None:
        while self.i < len(self.s) and self.s[self.i] in " \t":
            self.i += 1

    def ws_nl(self) -> None:
        """Whitespace, newlines and comments (inside arrays)."""
        while self.i < len(self.s):
            c = self.s[self.i]
            if c in " \t\n":
                self.i += 1
            elif c == "#":
                self.skip_comment()
            else:
                break

    def skip_comment(self) -> None:
        j = self.s.find("\n", self.i)
        self.i = len(self.s) if j < 0 else j

    def eol(self) -> None:
        self.ws()
        if self.peek() == "#":
            self.skip_comment()
        if self.i < len(self.s):
            if self.s[self.i] != "\n":
                raise self.err(f"expected end of line, found {self.s[self.i]!r}")
            self.i += 1

    # -- keys -------------------------------------------------------------------------------
    def key(self) -> list[str]:
        parts: list[str] = []
        while True:
            self.ws()
            c = self.peek()
            if c == '"':
                parts.append(self.basic_string())
            elif c == "'":
                parts.append(self.literal_string())
            else:
                m = _BARE.match(self.s, self.i)
                if not m:
                    raise self.err("expected a key")
                parts.append(m.group(0))
                self.i = m.end()
            self.ws()
            if self.peek() != ".":
                return parts
            self.i += 1

    # -- values -----------------------------------------------------------------------------
    def basic_string(self) -> str:
        if self.peek(3) == '"""':
            self.i += 3
            if self.peek() == "\n":
                self.i += 1
            return self._basic_body('"""', multiline=True)
        self.i += 1
        return self._basic_body('"', multiline=False)

    def _basic_body(self, end: str, *, multiline: bool) -> str:
        out: list[str] = []
        while True:
            if self.i >= len(self.s):
                raise self.err("unterminated string")
            if self.s.startswith(end, self.i):
                if multiline and self.s.startswith(end + '"', self.i):  # up to two quotes before the closing triple
                    out.append('"')
                    self.i += 1
                    continue
                self.i += len(end)
                return "".join(out)
            c = self.s[self.i]
            if c == "\n" and not multiline:
                raise self.err("newline in a basic string")
            if c == "\\":
                self.i += 1
                e = self.peek()
                if multiline and e in " \t\n":  # line-ending backslash: trim the newline and leading whitespace
                    self.ws()
                    if self.peek() != "\n":
                        raise self.err("invalid escape")
                    self.ws_nl()
                    continue
                if e in _ESC:
                    out.append(_ESC[e])
                    self.i += 1
                elif e in ("u", "U"):
                    n = 4 if e == "u" else 8
                    hexs = self.s[self.i + 1:self.i + 1 + n]
                    if len(hexs) != n or not re.fullmatch(r"[0-9A-Fa-f]+", hexs):
                        raise self.err("invalid unicode escape")
                    out.append(chr(int(hexs, 16)))
                    self.i += 1 + n
                else:
                    raise self.err(f"invalid escape \\{e}")
                continue
            out.append(c)
            self.i += 1

    def literal_string(self) -> str:
        if self.peek(3) == "'''":
            self.i += 3
            if self.peek() == "\n":
                self.i += 1
            j = self.s.find("'''", self.i)
            if j < 0:
                raise self.err("unterminated string")
            while self.s.startswith("'", j + 3) and j + 3 < len(self.s):
                j += 1
            out = self.s[self.i:j]
            self.i = j + 3
            return out
        self.i += 1
        j = self.s.find("'", self.i)
        nl = self.s.find("\n", self.i)
        if j < 0 or (0 <= nl < j):
            raise self.err("unterminated string")
        out = self.s[self.i:j]
        self.i = j + 1
        return out

    def value(self) -> Any:
        self.ws()
        c = self.peek()
        if c == '"':
            return self.basic_string()
        if c == "'":
            return self.literal_string()
        if c == "[":
            return self.array()
        if c == "{":
            return self.inline_table()
        m = re.compile(r"[^\s,\]\}#]+").match(self.s, self.i)
        if not m:
            raise self.err("expected a value")
        tok = m.group(0)
        self.i = m.end()
        if tok == "true":
            return True
        if tok == "false":
            return False
        if tok in ("inf", "+inf", "-inf", "nan", "+nan", "-nan"):
            return float(tok)
        for prefix, base in (("0x", 16), ("0o", 8), ("0b", 2)):
            if tok.startswith(prefix):
                try:
                    return int(tok[2:].replace("_", ""), base)
                except ValueError:
                    raise self.err(f"invalid integer {tok!r}") from None
        if _INT.match(tok):
            return int(tok.replace("_", ""))
        if _FLOAT.match(tok):
            return float(tok.replace("_", ""))
        raise self.err(f"unsupported value {tok!r} (dates/times are not supported by yavih's fallback reader)")

    def array(self) -> list:
        self.i += 1
        out: list = []
        while True:
            self.ws_nl()
            if self.peek() == "]":
                self.i += 1
                return out
            out.append(self.value())
            self.ws_nl()
            c = self.peek()
            if c == ",":
                self.i += 1
            elif c == "]":
                self.i += 1
                return out
            else:
                raise self.err("expected ',' or ']' in an array")

    def inline_table(self) -> dict:
        self.i += 1
        out: dict = {}
        self.ws()
        if self.peek() == "}":
            self.i += 1
            return out
        while True:
            k = self.key()
            if self.peek() != "=":
                raise self.err("expected '=' after a key")
            self.i += 1
            _assign(out, k, self.value(), self)
            self.ws()
            c = self.peek()
            if c == ",":
                self.i += 1
            elif c == "}":
                self.i += 1
                return out
            else:
                raise self.err("expected ',' or '}' in an inline table")

    # -- document ---------------------------------------------------------------------------
    def document(self) -> dict:
        root: dict = {}
        cur = root
        defined: set[tuple[str, ...]] = set()
        while True:
            self.ws_nl()
            if self.i >= len(self.s):
                return root
            if self.peek(2) == "[[":
                self.i += 2
                path = self.key()
                if self.peek(2) != "]]":
                    raise self.err("expected ']]'")
                self.i += 2
                parent = _walk(root, path[:-1], self)
                arr = parent.setdefault(path[-1], [])
                if not isinstance(arr, list):
                    raise self.err(f"{'.'.join(path)} is not an array of tables")
                cur = {}
                arr.append(cur)
            elif self.peek() == "[":
                self.i += 1
                path = self.key()
                if self.peek() != "]":
                    raise self.err("expected ']'")
                self.i += 1
                if tuple(path) in defined:
                    raise self.err(f"table [{'.'.join(path)}] defined twice")
                defined.add(tuple(path))
                cur = _walk(root, path, self)
            else:
                k = self.key()
                if self.peek() != "=":
                    raise self.err("expected '=' after a key")
                self.i += 1
                _assign(cur, k, self.value(), self)
            self.eol()


def _walk(d: dict, path: list[str], p: _Parser) -> dict:
    for part in path:
        nxt = d.setdefault(part, {})
        if isinstance(nxt, list):  # [a.b] after [[a]]: the last element
            nxt = nxt[-1]
        if not isinstance(nxt, dict):
            raise p.err(f"key {part!r} is not a table")
        d = nxt
    return d


def _assign(d: dict, key: list[str], value: Any, p: _Parser) -> None:
    target = _walk(d, key[:-1], p)
    if key[-1] in target:
        raise p.err(f"key {'.'.join(key)!r} defined twice")
    target[key[-1]] = value


def loads(text: str) -> dict[str, Any]:
    """Parse a TOML document (the subset in the module docstring) into a dict."""
    return _Parser(text).document()


def load(fh: Any) -> dict[str, Any]:
    """``tomllib.load`` signature: a binary file object."""
    data = fh.read()
    if isinstance(data, bytes):
        try:
            data = data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise TOMLDecodeError(f"not UTF-8: {e}") from e
    return loads(data)
