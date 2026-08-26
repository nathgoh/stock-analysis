import html
import re

_BLOCK_RE: re.Pattern[str] = re.compile(
    r"<script\b[^>]*>.*?</script>",
    re.IGNORECASE | re.DOTALL,
)


_TAG_RE: re.Pattern[str] = re.compile(
    r"""
    <                       # opening angle bracket
    (?:                     # non-capturing group of allowed tag names
        a\s*[^>]*?
      | /a
      | a
      | p\s*[^>]*?
      | /p
      | p
      | br\s*/?
      | strong\s*[^>]*?
      | /strong
      | strong
      | em\s*[^>]*?
      | /em
      | em
      | b\s*[^>]*?
      | /b
      | b
      | i\s*[^>]*?
      | /i
      | i
      | u\s*[^>]*?
      | /u
      | u
      | h[1-6]\s*[^>]*?
      | /h[1-6]
      | h[1-6]
      | ul\s*[^>]*?
      | /ul
      | ul
      | ol\s*[^>]*?
      | /ol
      | ol
      | li\s*[^>]*?
      | /li
      | li
      | div\s*[^>]*?
      | /div
      | div
      | span\s*[^>]*?
      | /span
      | span
      | blockquote\s*[^>]*?
      | /blockquote
      | blockquote
      | code\s*[^>]*?
      | /code
      | code
      | pre\s*[^>]*?
      | /pre
      | pre
      | img\s*[^>]*?
      | img
      | hr\s*/?
      | hr
      | figure\s*[^>]*?
      | /figure
      | figure
    )
    >
    """,
    re.IGNORECASE | re.VERBOSE,
)


_MARKDOWN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),  # images
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),  # links
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),  # headings
    (re.compile(r"\*{1,3}([^*]+)\*{1,3}"), r"\1"),  # bold/italic
    (re.compile(r"_{1,3}([^_]+)_{1,3}"), r"\1"),  # bold/italic
    (re.compile(r"~~([^~]+)~~"), r"\1"),  # strikethrough
    (re.compile(r"`{1,3}([^`]+)`{1,3}"), r"\1"),  # inline/block code
    (re.compile(r"^>\s+", re.MULTILINE), ""),  # blockquotes
    (re.compile(r"^[-*+]\s+", re.MULTILINE), ""),  # unordered list markers
    (re.compile(r"^\d+\.\s+", re.MULTILINE), ""),  # ordered list markers
    (re.compile(r"^\s*[-*_]{3,}\s*$", re.MULTILINE), ""),  # horizontal rules
    (re.compile(r"^(\s*\|.*\|\s*)$", re.MULTILINE), ""),  # table rows
]


_WHITESPACE_RE: re.Pattern[str] = re.compile(r"[ \t]+")
_BLANK_LINE_RE: re.Pattern[str] = re.compile(r"\n{3,}")


def strip_html(text: str) -> str:
    """
    Remove HTML tags from text and unescape HTML entities.
    """

    if not text:
        return ""
    text = _BLOCK_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return text


def strip_markdown(text: str) -> str:
    """
    Remove common Markdown formatting from text.
    """

    if not text:
        return ""
    for pattern, replacement in _MARKDOWN_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def clean_text(text: str) -> str:
    """
    Strip HTML and Markdown, then normalize whitespace.
    """

    if not text:
        return ""
    text = strip_html(text)
    text = strip_markdown(text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINE_RE.sub("\n\n", text)
    return text.strip()
