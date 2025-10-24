from dataclasses import dataclass
import math
import re
import string
import unicodedata


@dataclass
class SecurityConfig:
    """Configuration for security validation."""

    max_expression_length: int = 500
    max_nesting_depth: int = 10
    max_variable_name_length: int = 50
    max_numeric_value: float = 1e12
    max_function_calls: int = 20
    max_complexity_score: int = 100


class SecurityError(ValueError):
    """Raised when a security violation is detected in an expression."""

    def __init__(self, message: str):
        super().__init__(message)


class ExpressionSecurityValidator:
    """
    Comprehensive security validator for mathematical expressions.

    Validates expressions against a wide range of security threats for both
    Python and Rust evaluation environments.
    """

    # Mathematical operators that are always allowed
    ALLOWED_OPERATORS: set[str] = {
        "+",
        "-",
        "*",
        "/",
        "^",
        "**",
        "%",
        "(",
        ")",
        ".",
        ",",
        " ",
        "\t",
        "\n",
        "\r",
    }

    # Safe mathematical functions supported by evalexpr (case-insensitive matching)
    SAFE_FUNCTIONS: set[str] = {
        # Trigonometric functions
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        # Exponential and logarithmic functions
        "exp",
        "ln",
        "log",
        "log2",
        "log10",
        # Power and root functions
        "sqrt",
        "cbrt",
        "pow",
        "hypot",
        # Rounding and absolute value functions
        "abs",
        "floor",
        "ceil",
        "round",
        # Min/max and conditional
        "max",
        "min",
        "if",
        # Type checking (evalexpr specific)
        "typeof",
        # Note: 'pi' and 'e' are constants, not functions
    }

    # Python-specific dangerous patterns
    PYTHON_DANGEROUS_PATTERNS: set[str] = {
        # Built-ins and imports
        "import",
        "__import__",
        "eval",
        "exec",
        "execfile",
        "compile",
        "open",
        "file",
        "raw_input",
        "print",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "hasattr",
        "delattr",
        "isinstance",
        "issubclass",
        "callable",
        "type",
        "super",
        "reload",
        "help",
        "id",
        "hash",
        "repr",
        "str",
        # System and OS access
        "os",
        "sys",
        "subprocess",
        "platform",
        "shutil",
        "environ",
        "system",
        "popen",
        "spawn",
        "fork",
        "path",
        "listdir",
        "walk",
        "mkdir",
        "rmdir",
        "remove",
        "unlink",
        "rename",
        "copy",
        "move",
        # Network and I/O
        "socket",
        "urllib",
        "http",
        "requests",
        "ftplib",
        "smtplib",
        "telnetlib",
        "connect",
        "bind",
        "listen",
        "accept",
        "read",
        "write",
        "flush",
        "close",
        "seek",
        "tell",
        # Dangerous Python features
        "class",
        "def",
        "lambda",
        "yield",
        "async",
        "await",
        "with",
        "try",
        "except",
        "finally",
        "raise",
        "assert",
        "del",
        "global",
        "nonlocal",
        "pass",
        "break",
        "continue",
        # Serialization and code objects
        "pickle",
        "marshal",
        "shelve",
        "dill",
        "joblib",
        "loads",
        "dumps",
        "load",
        "dump",
        "compile",
        "code",
        "frame",
        "traceback",
        # Format strings and templates
        "format",
        "template",
        "formatter",
        # Memory and garbage collection
        "gc",
        "weakref",
        "ctypes",
        "array",
        "buffer",
        "memoryview",
        # Metaclasses and descriptors
        "metaclass",
        "__new__",
        "__init__",
        "__call__",
        "__getattribute__",
        # Dynamic execution
        "property",
        "staticmethod",
        "classmethod",
        "decorator",
    }

    # Rust-specific dangerous patterns
    RUST_DANGEROUS_PATTERNS: set[str] = {
        # Unsafe operations
        "unsafe",
        "transmute",
        "from_raw",
        "raw_parts",
        "raw",
        "parts",
        "as_ptr",
        "as_mut_ptr",
        # Note: "offset" removed - it's a common mathematical term
        "add",
        "sub",
        "wrapping",
        "unchecked",
        "wrapping_add",
        "wrapping_sub",
        "wrapping_mul",
        "unchecked_add",
        "unchecked_sub",
        "unchecked_mul",
        # System calls and FFI
        "libc",
        "winapi",
        "syscall",
        "extern",
        "asm",
        "llvm_asm",
        "link",
        "no_mangle",
        "repr",
        "packed",
        # Memory manipulation
        "alloc",
        "dealloc",
        "realloc",
        "layout",
        "global_alloc",
        "ptr",
        "null",
        "dangling",
        "alignment",
        "slice_from_raw_parts",
        "slice_from_raw_parts_mut",
        # Concurrency primitives that could be dangerous
        "spawn",
        "thread",
        "mutex",
        "atomic",
        "channel",
        "sync",
        "send",
        "static",
        "lazy",
        "lazy_static",
        # Macros and code generation
        "macro_rules",
        "proc_macro",
        "derive",
        "cfg",
        "include",
        "include_str",
        "include_bytes",
        "concat",
        "stringify",
        # I/O and filesystem
        "fs",
        "file",
        "read",
        "write",
        "open",
        "create",
        "std",
        "env",
        "process",
        "command",
        "output",
        # Network operations
        "net",
        "tcp",
        "udp",
        "socket",
        "connect",
        "bind",
        "listen",
        # Foreign function interface
        "ffi",
        "c_void",
        "c_char",
        "c_int",
        "cstring",
        "cstr",
        # Panic and error handling that could be exploited
        "panic",
        "unreachable",
        "todo",
        "unimplemented",
        # Type system manipulation
        "any",
        "typeid",
        "downcast",
        "phantom",
        "marker",
    }

    # Combined dangerous patterns
    DANGEROUS_PATTERNS: set[str] = PYTHON_DANGEROUS_PATTERNS | RUST_DANGEROUS_PATTERNS

    # Suspicious keywords that should not appear in variable names
    SUSPICIOUS_KEYWORDS: set[str] = {
        "eval",
        "exec",
        "import",
        "system",
        "unsafe",
        "transmute",
        "admin",
        "root",
        "sudo",
        "password",
        "secret",
        "key",
        "token",
        "auth",
        "login",
        "user",
        "exploit",
        "hack",
        "inject",
        "payload",
        "shell",
        "cmd",
        "command",
    }

    # Regex patterns for advanced threat detection
    _dunder_regex: re.Pattern[str] = re.compile(r"__\w+__")
    _hex_escape_regex: re.Pattern[str] = re.compile(r"\\x[0-9a-fA-F]{2}")
    _unicode_escape_regex: re.Pattern[str] = re.compile(r"\\u[0-9a-fA-F]{4}")
    _octal_escape_regex: re.Pattern[str] = re.compile(r"\\[0-7]{1,3}")
    _base64_regex: re.Pattern[str] = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
    _url_regex: re.Pattern[str] = re.compile(r"https?://[^\s]+")
    _ip_regex: re.Pattern[str] = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")
    _format_string_regex: re.Pattern[str] = re.compile(r"\{[^}]*\}")
    _percent_format_regex: re.Pattern[str] = re.compile(r"%[sdiouxXeEfFgGcr%]")

    # Zero-width and invisible Unicode characters
    DANGEROUS_UNICODE_CATEGORIES: set[str] = {
        "Cf",  # Other, format
        "Mn",  # Mark, nonspacing
        "Me",  # Mark, enclosing
    }

    # Homograph attack character mappings (lookalike characters)
    HOMOGRAPH_MAP: dict[str, str] = {
        # Cyrillic to Latin lookalikes
        "а": "a",
        "е": "e",
        "о": "o",
        "р": "p",
        "с": "c",
        "х": "x",
        "у": "y",
        "А": "A",
        "В": "B",
        "С": "C",
        "Е": "E",
        "Н": "H",
        "К": "K",
        "М": "M",
        "О": "O",
        "Р": "P",
        "Т": "T",
        "У": "Y",
        "Х": "X",
        # Greek to Latin lookalikes
        "α": "a",
        "β": "b",
        "γ": "g",
        "δ": "d",
        "ε": "e",
        "ζ": "z",
        "η": "h",
        "θ": "th",
        "ι": "i",
        "κ": "k",
        "λ": "l",
        "μ": "m",
        "ν": "n",
        "ξ": "x",
        "ο": "o",
        "π": "p",
        "ρ": "r",
        "σ": "s",
        "τ": "t",
        "υ": "u",
        "φ": "f",
        "χ": "ch",
        "ψ": "ps",
        "ω": "w",
    }

    def __init__(self, config: SecurityConfig | None = None):
        """Initialize the validator with the given configuration."""
        self.config: SecurityConfig = config or SecurityConfig()

    def validate_expression(self, expression: str) -> None:
        """
        Comprehensive validation of an expression for security concerns.

        Parameters
        ----------
        expression: str
            The mathematical expression to validate

        Raises
        ------
        SecurityError
            If the expression contains security violations
        """
        if not expression or not expression.strip():
            raise SecurityError("Empty expression")

        # Basic checks
        self._check_length(expression)
        self._check_encoding_attacks(expression)
        self._check_unicode_attacks(expression)

        # Content validation (dangerous patterns first since they ignore special chars)
        self._check_dangerous_patterns(expression)
        self._check_allowed_characters(expression)

        # Structure validation
        self._check_nesting_depth(expression)
        self._check_balanced_delimiters(expression)

        # Advanced threat detection
        self._check_advanced_threats(expression)

        # Tokenization and semantic analysis
        tokens = self._tokenize_expression(expression)
        self._validate_tokens(tokens)

        # Strict mode additional checks
        self._strict_validation(tokens, expression)

    def _check_length(self, expression: str) -> None:
        """Check if expression length is within limits."""
        if len(expression) > self.config.max_expression_length:
            raise SecurityError(
                (
                    f"Expression too long: {len(expression)} characters "
                    f"(maximum: {self.config.max_expression_length})"
                )
            )

    def _check_encoding_attacks(self, expression: str) -> None:
        """Check for encoding-based attacks."""
        if self._hex_escape_regex.search(expression):
            raise SecurityError(
                "Encoding attack detected: Hex escape sequences not allowed"
            )

        if self._unicode_escape_regex.search(expression):
            raise SecurityError(
                "Encoding attack detected: Unicode escape sequences not allowed"
            )

        if self._octal_escape_regex.search(expression):
            raise SecurityError(
                "Encoding attack detected: Octal escape sequences not allowed"
            )

        if self._base64_regex.search(expression):
            raise SecurityError(
                "Encoding attack detected: Suspected base64 encoded content"
            )

        if self._url_regex.search(expression):
            raise SecurityError(
                "Encoding attack detected: URLs not allowed in expressions"
            )

        if self._ip_regex.search(expression):
            raise SecurityError(
                "Encoding attack detected: IP addresses not allowed in expressions"
            )

    def _check_unicode_attacks(self, expression: str) -> None:
        """Check for Unicode-based attacks."""
        for i, char in enumerate(expression):
            category = unicodedata.category(char)
            if category in self.DANGEROUS_UNICODE_CATEGORIES:
                raise SecurityError(
                    (
                        f"Dangerous Unicode character at position {i}: "
                        f"'{char}' (category: {category})"
                    )
                )

            if char in self.HOMOGRAPH_MAP:
                raise SecurityError(
                    (
                        f"Suspicious Unicode character at position {i}: '{char}' "
                        f"(looks like '{self.HOMOGRAPH_MAP[char]}')"
                    )
                )

            # Check for zero-width characters
            if unicodedata.east_asian_width(char) == "F" and ord(char) in [
                0x200B,
                0x200C,
                0x200D,
                0xFEFF,
            ]:
                raise SecurityError(f"Zero-width character detected at position {i}")

    def _check_nesting_depth(self, expression: str) -> None:
        """Check parentheses and bracket nesting depth."""
        depth = 0
        max_depth = 0

        for char in expression:
            if char == "(":
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == ")":
                depth -= 1
                if depth < 0:
                    raise SecurityError("Unmatched parentheses")
        if depth != 0:
            raise SecurityError("Unmatched parentheses")

        if max_depth > self.config.max_nesting_depth:
            raise SecurityError(
                (
                    f"Too many nested levels: {max_depth} "
                    f"(maximum: {self.config.max_nesting_depth})"
                )
            )

    def _check_balanced_delimiters(self, expression: str) -> None:
        """Check for balanced delimiters and quotes."""
        quote_stack: list[str] = []
        for _, char in enumerate(expression):
            if char in ['"', "'", "`"]:
                if quote_stack and quote_stack[-1] == char:
                    _ = quote_stack.pop()
                else:
                    quote_stack.append(char)
        if quote_stack:
            raise SecurityError("Unmatched quotes detected")

    def _check_dangerous_patterns(self, expression: str) -> None:
        """Check for known dangerous patterns."""
        lower_expr = expression.lower()

        tokens: list[str] = re.findall(r"\w+", lower_expr)

        # Check individual tokens
        for token in tokens:
            if token in self.DANGEROUS_PATTERNS:
                raise SecurityError(f"Dangerous pattern detected: '{token}'")

        # Check for compound patterns (underscore-separated)
        compound_tokens: list[str] = re.findall(r"[a-z_]+[a-z0-9_]*", lower_expr)
        for compound in compound_tokens:
            if compound in self.DANGEROUS_PATTERNS:
                raise SecurityError(f"Dangerous pattern detected: '{compound}'")

            # Check if compound contains dangerous subpatterns
            parts = compound.split("_")
            for part in parts:
                if part and len(part) > 3 and part in self.DANGEROUS_PATTERNS:
                    raise SecurityError(
                        f"Suspicious variable name: '{compound}' contains '{part}'"
                    )

    def _check_allowed_characters(self, expression: str) -> None:
        """Check if expression contains only allowed characters."""
        allowed_chars = (
            set(string.ascii_letters)
            | set(string.digits)
            | self.ALLOWED_OPERATORS
            | {"_"}
        )
        for char in expression:
            if char not in allowed_chars:
                raise SecurityError(f"Disallowed character: '{char}'")
        # Check for malformed numbers
        self._check_malformed_numbers(expression)

    def _check_malformed_numbers(self, expression: str) -> None:
        """Check for malformed numeric values."""
        # First check if this contains valid hex/binary/octal numbers - if so,
        # skip the check
        has_valid_special_format = re.search(r"0[xXbBoO][0-9a-fA-F]+", expression)

        # Pattern for numbers followed by letters (excluding valid scientific
        # notation and special formats)
        if not has_valid_special_format and re.search(
            r"\d+[a-df-zA-DF-Z]+", expression
        ):
            raise SecurityError("Invalid numeric format: digits followed by letters")

        # Pattern for digits followed by 'e' and then non-numeric characters
        # (invalid scientific notation)
        if not has_valid_special_format and re.search(r"\d+[eE][a-zA-Z]", expression):
            raise SecurityError("Invalid numeric format: digits followed by letters")

        # Pattern for multiple decimal points (e.g., 3..14)
        if re.search(r"\d+\.\.", expression):
            raise SecurityError("Invalid numeric format: multiple decimal points")

        # Pattern for incomplete scientific notation (e.g., 1e, 1e1e1)
        if re.search(r"\d+e(?![+-]?\d)", expression, re.IGNORECASE):
            raise SecurityError(
                "Invalid numeric format: incomplete scientific notation"
            )

        # Pattern for scientific notation starting with 'e' (e.g., e10)
        if re.search(r"(?<!\w)e\d+", expression, re.IGNORECASE):
            raise SecurityError(
                "Invalid numeric format: scientific notation without base"
            )

        # Pattern for multiple 'e' in scientific notation (e.g., 1e1e1)
        if re.search(r"\d+e[+-]?\d+e", expression, re.IGNORECASE):
            raise SecurityError("Invalid numeric format: multiple exponents")

    def _check_advanced_threats(self, expression: str) -> None:
        """Check for advanced attack patterns."""
        # Format string attacks
        if self._format_string_regex.search(
            expression
        ) or self._percent_format_regex.search(expression):
            raise SecurityError("Format string patterns detected")

        # Check for too many function calls
        function_call_count = expression.count("(")
        if function_call_count > self.config.max_function_calls:
            raise SecurityError(
                (
                    f"Too many function calls: {function_call_count} "
                    f"(maximum: {self.config.max_function_calls})"
                )
            )

        # Check for repeated patterns that might indicate attack attempts
        if len(set(expression)) < len(expression) * 0.1 and len(expression) > 200:
            raise SecurityError("Suspicious repetitive pattern")

        # Check for very long identifiers (potential buffer overflow attempts)
        tokens: list[str] = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", expression)
        for token in tokens:
            if len(token) > self.config.max_variable_name_length * 2:
                raise SecurityError(
                    f"Variable name too long: '{token[:50]}...' (length: {len(token)})"
                )

    def _tokenize_expression(self, expression: str) -> list[str]:
        """Tokenize the expression into components."""
        token_regex = (
            r"[a-zA-Z_][a-zA-Z0-9_]*"  # Identifiers
            r"|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?"  # Numbers
            r"|\.\d+(?:[eE][+-]?\d+)?"  # Numbers starting with .
            r"|0[xX][0-9a-fA-F]+"  # Hexadecimal numbers
            r"|0[bB][01]+"  # Binary numbers
            r"|0[oO][0-7]+"  # Octal numbers
            r"|[\+\-\*/\^\(\)\[\]\.,<>=!&|?:;%]"  # Operators
            r"|['\"][^'\"]*['\"]"  # Quoted strings
        )
        matches: list[str] = re.findall(token_regex, expression)
        return [token for token in matches if token.strip()]

    def _validate_tokens(self, tokens: list[str]) -> None:
        """Validate individual tokens."""
        function_call_count = 0
        for token in tokens:
            if not token.strip():
                continue
            # Skip operators
            if token in self.ALLOWED_OPERATORS:
                continue
            # Validate numbers
            if self._is_number(token):
                self._validate_numeric_value(token)
                continue
            # Check for quoted strings (not allowed in math expressions)
            if (token.startswith('"') and token.endswith('"')) or (
                token.startswith("'") and token.endswith("'")
            ):
                raise SecurityError(f"Quoted strings not allowed: {token}")
            # Check function names
            if token.lower() in self.SAFE_FUNCTIONS:
                function_call_count += 1
                if function_call_count > self.config.max_function_calls:
                    raise SecurityError(
                        (
                            f"Too many function calls: {function_call_count} "
                            f"(maximum: {self.config.max_function_calls})"
                        )
                    )
                continue
            # Check if it's a valid variable name
            if self._is_valid_variable_name(token):
                continue
            # If we get here, the token is suspicious
            raise SecurityError(f"Suspicious token: '{token}'")

    def _validate_numeric_value(self, token: str) -> None:
        """Validate that numeric values are within reasonable bounds."""
        try:
            value = float(token)
            if abs(value) > self.config.max_numeric_value:
                raise SecurityError(
                    (
                        f"Numeric value too large: {value} "
                        f"(maximum: {self.config.max_numeric_value})"
                    )
                )
            if math.isnan(value) or math.isinf(value):
                raise SecurityError(f"Invalid numeric value: {token}")
        except SecurityError:
            raise
        except (ValueError, OverflowError):
            raise SecurityError(f"Invalid numeric token: {token}")

    def _strict_validation(self, tokens: list[str], expression: str) -> None:
        """Additional validation for strict mode."""
        for token in tokens:
            if self._is_number(token) or token in self.ALLOWED_OPERATORS:
                continue
            if token.lower() in self.SAFE_FUNCTIONS:
                continue
            # Check for dunder methods
            if self._dunder_regex.search(token):
                raise SecurityError(f"Suspicious dunder pattern: '{token}'")
            # Split token and check for suspicious substrings
            substrings = re.split(r"[_\-.]|(?=[A-Z])", token.lower())
            for substring in substrings:
                if substring and substring in self.SUSPICIOUS_KEYWORDS:
                    raise SecurityError(
                        f"Suspicious keyword '{substring}' in identifier: '{token}'"
                    )
            # Check variable name length
            if len(token) > self.config.max_variable_name_length:
                raise SecurityError(
                    (
                        f"Variable name too long: '{token}' "
                        f"(maximum: {self.config.max_variable_name_length})"
                    )
                )

        # Check for potential code injection through mathematical operations
        self._check_mathematical_injection(expression)

    def _check_mathematical_injection(self, expression: str) -> None:
        """Check for attempts to inject code through mathematical constructs."""
        # Look for suspicious mathematical patterns that could be code
        suspicious_math_patterns = [
            r"\d+\s*[eE]\s*[a-zA-Z]",  # Numbers followed by 'e' and letters
            r"[a-zA-Z]+\s*\d+\s*[a-zA-Z]+",  # Mixed alphanumeric that's not valid math
            r"\.[a-zA-Z]+\s*\(",  # Method calls
            r"[a-zA-Z]+\s*\[[^\]]*\]",  # Array/dict access
        ]

        for pattern in suspicious_math_patterns:
            if re.search(pattern, expression):
                raise SecurityError("Suspicious mathematical pattern detected")

    def _is_number(self, token: str) -> bool:
        """Check if a token represents a number."""
        try:
            _ = float(token)
            return True
        except ValueError:
            # Try other number formats
            try:
                if token.startswith(("0x", "0X")):
                    _ = int(token, 16)
                    return True
                elif token.startswith(("0b", "0B")):
                    _ = int(token, 2)
                    return True
                elif token.startswith(("0o", "0O")):
                    _ = int(token, 8)
                    return True
            except ValueError:
                pass
            return False

    def _is_valid_variable_name(self, token: str) -> bool:
        """Check if a token is a valid variable name."""
        if not token:
            return False

        # Must start with letter or underscore
        if not (token[0].isalpha() or token[0] == "_"):
            return False

        # Must contain only alphanumeric characters and underscores
        return all(c.isalnum() or c == "_" for c in token)


def validate_expression_security(
    expression: str, config: SecurityConfig | None = None
) -> None:
    """
    Validate an expression for security concerns.

    This is the main entry point for comprehensive expression security validation
    that protects against both Python and Rust-specific attack vectors.

    Parameters
    ----------
    expression: str
        The mathematical expression to validate
    config: SecurityConfig
        Optional custom configuration

    Raises
    ------
    SecurityError
        If the expression contains security violations with detailed information
        about the violation type and context

    Examples
    --------
    >>> validate_expression_security("beta * S * I / N")  # OK
    >>> validate_expression_security("__import__('os')")  # Raises SecurityError
    >>> validate_expression_security("unsafe { *ptr }")  # Raises SecurityError
    >>> validate_expression_security("α * β")  # Raises SecurityError (unicode)
    """
    validator = ExpressionSecurityValidator(config)
    validator.validate_expression(expression)


def get_expression_variables(expression: str) -> list[str]:
    """
    Extracts variable names from a mathematical expression.

    This function identifies sequences of characters that represent variable names,
    ignoring numbers, operators, and known mathematical functions.

    Parameters
    ----------
    expression : str
        The mathematical expression.

    Returns
    -------
    list[str]
        A list of unique variable names found in the expression.
    """
    # Regex to find identifiers (variables)
    identifier_regex = r"[a-zA-Z_][a-zA-Z0-9_]*"

    # Find all potential identifiers
    all_identifiers: list[str] = re.findall(identifier_regex, expression)

    # Filter out known safe functions and constants
    safe_tokens = ExpressionSecurityValidator.SAFE_FUNCTIONS | {"pi", "e"}
    variables: list[str] = []
    for var in all_identifiers:
        if var.lower() not in safe_tokens and not var.isdigit():
            variables.append(var)

    return list(set(variables))
