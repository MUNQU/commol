import pytest

from commol.utils.security import (
    validate_expression_security,
    SecurityError,
    SecurityConfig,
    ExpressionSecurityValidator,
)


class TestBasicValidation:
    """Test basic validation functionality and valid expressions."""

    def test_valid_mathematical_expressions(self):
        """Test that valid mathematical expressions pass validation."""
        valid_expressions = [
            # Basic variables and numbers
            "beta",
            "gamma",
            "0.5",
            "42",
            "3.14159",
            # Basic arithmetic
            "beta * gamma",
            "S + I + R",
            "N - S",
            "beta * S * I / N",
            "gamma * I",
            # Mathematical functions
            "sin(step)",
            "cos(2 * pi * step / 365)",
            "exp(-decay_rate * step)",
            "log(population_size)",
            "sqrt(variance)",
            "abs(difference)",
            "max(a, b)",
            "min(susceptible, exposed)",
            "pow(base, exponent)",
            "floor(continuous_value)",
            "ceil(fractional_value)",
            "round(precise_value)",
            # Complex expressions
            "(S + I + R) / N",
            "beta * (1 - vaccination_rate)",
            "1 / (1 + exp(-steepness * (step - midpoint)))",
            "beta * (1 + 0.1 * sin(2 * pi * step / 365))",
        ]
        for expr in valid_expressions:
            validate_expression_security(expr)

    def test_numeric_formats(self):
        """Test various numeric format validations."""
        valid_numbers = [
            "0",
            "42",
            "3.14",
            "2.71828",
            "0.001",
            "1000000",
            "1e-6",
            "1E+3",
            "2.5e-10",
            ".5",
            "5.",
            "0xFF",
            "0b1010",
            "0o777",
        ]
        for num in valid_numbers:
            validate_expression_security(num)

    def test_safe_mathematical_functions(self):
        """Test all safe mathematical functions are allowed."""
        safe_functions = [
            # Trigonometric
            "sin(x)",
            "cos(x)",
            "tan(x)",
            "asin(x)",
            "acos(x)",
            "atan(x)",
            "atan2(y, x)",
            "sinh(x)",
            "cosh(x)",
            "tanh(x)",
            # Exponential and logarithmic
            "exp(x)",
            "log(x)",
            "ln(x)",
            "log2(x)",
            "log10(x)",
            # Power and root
            "sqrt(x)",
            "pow(x, y)",
            "abs(x)",
            # Rounding
            "floor(x)",
            "ceil(x)",
            "round(x)",
            "trunc(x)",
            # Min/max
            "max(a, b)",
            "min(a, b)",
            # Constants
            "pi",
            "e",
        ]
        for func in safe_functions:
            validate_expression_security(func)

    def test_tokenization(self):
        """Test expression tokenization."""
        validator = ExpressionSecurityValidator()
        # Test basic tokenization
        tokens = validator._tokenize_expression("beta * S + gamma")  # pyright: ignore[reportPrivateUsage]  # noqa: E501
        assert "beta" in tokens
        assert "*" in tokens
        assert "S" in tokens
        assert "+" in tokens
        assert "gamma" in tokens
        # Test function tokenization
        tokens = validator._tokenize_expression("sin(2 * pi * t)")  # pyright: ignore[reportPrivateUsage]  # noqa: E501
        assert "sin" in tokens
        assert "2" in tokens
        assert "pi" in tokens
        assert "t" in tokens


class TestPythonSecurityThreats:
    """Test Python-specific security threat detection."""

    def test_dangerous_builtin_functions(self):
        """Test that dangerous Python builtin functions are blocked."""
        dangerous_builtins = [
            "__import__('os')",
            "eval('malicious_code')",
            "exec('dangerous_command')",
            "compile('code', 'file', 'exec')",
            "open('/etc/passwd')",
            "globals()",
            "locals()",
            "getattr(obj, 'attr')",
            "setattr(obj, 'attr', 'value')",
            "hasattr(obj, 'attr')",
            "delattr(obj, 'attr')",
            "isinstance(obj, type)",
            "callable(function)",
        ]
        for expr in dangerous_builtins:
            with pytest.raises(SecurityError, match="Dangerous pattern detected"):
                validate_expression_security(expr)

    def test_system_access_patterns(self):
        """Test that Python system access patterns are blocked."""
        system_patterns = [
            "os.system('command')",
            "subprocess.call(['ls'])",
            "platform.system()",
            "sys.exit(1)",
            "shutil.rmtree('/')",
            "environ['PATH']",
        ]
        for expr in system_patterns:
            with pytest.raises(SecurityError):
                validate_expression_security(expr)

    def test_network_access_patterns(self):
        """Test that Python network access patterns are blocked."""
        network_patterns = [
            "socket.connect(('host', 80))",
            "urllib.request.urlopen('http://evil.com')",
            "requests.get('http://malicious.com')",
            "http.client.HTTPConnection('evil.com')",
            "ftplib.FTP('ftp.evil.com')",
        ]
        for expr in network_patterns:
            with pytest.raises(SecurityError):
                validate_expression_security(expr)

    def test_python_language_constructs(self):
        """Test that Python language constructs are blocked."""
        language_constructs = [
            "class MyClass: pass",
            "def function(): pass",
            "lambda x: x",
            "yield value",
            "async def func(): pass",
            "await coroutine",
            "with context: pass",
            "try: operation()",
            "except Exception: handle()",
            "finally: cleanup()",
            "raise ValueError('error')",
            "assert condition",
            "del variable",
            "global var",
            "nonlocal var",
            "import math",
            "from os import path",
        ]
        for expr in language_constructs:
            with pytest.raises(SecurityError):
                validate_expression_security(expr)

    def test_serialization_patterns(self):
        """Test that serialization-related patterns are blocked."""
        serialization_patterns = [
            "pickle.loads(data)",
            "marshal.load(file)",
            "shelve.open('file')",
            "joblib.load('model')",
        ]
        for expr in serialization_patterns:
            with pytest.raises(SecurityError):
                validate_expression_security(expr)

    def test_dunder_method_access(self):
        """Test that dunder method access is blocked."""
        dunder_patterns = [
            "__init__",
            "__call__",
            "__getattribute__",
            "__dict__",
            "__class__",
            "__bases__",
            "__module__",
            "obj.__dict__",
            "instance.__class__.__name__",
        ]
        for expr in dunder_patterns:
            with pytest.raises(SecurityError):
                validate_expression_security(expr)


class TestRustSecurityThreats:
    """Test Rust-specific security threat detection."""

    def test_unsafe_operations(self):
        """Test that Rust unsafe operations are blocked."""
        unsafe_patterns = [
            "unsafe { code }",
            "transmute(value)",
            "from_raw(ptr)",
            "raw_parts(slice)",
            "as_ptr()",
            "as_mut_ptr()",
            "wrapping_add(value)",
            "unchecked_add(value)",
        ]
        for expr in unsafe_patterns:
            with pytest.raises(SecurityError, match="Dangerous pattern detected"):
                validate_expression_security(expr)

    def test_system_calls_and_ffi(self):
        """Test that Rust system calls and FFI patterns are blocked."""
        system_patterns = [
            "libc::system(command)",
            "winapi::um::processthreadsapi::CreateProcess",
            "syscall(number, args)",
            'extern "C" { fn dangerous(); }',
            'asm!("mov rax, rbx")',
            'llvm_asm!("nop")',
        ]
        for expr in system_patterns:
            with pytest.raises(SecurityError):
                validate_expression_security(expr)

    def test_memory_operations(self):
        """Test that Rust memory manipulation patterns are blocked."""
        memory_patterns = [
            "alloc::alloc(layout)",
            "alloc::dealloc(ptr, layout)",
            "std::ptr::null()",
            "std::ptr::dangling()",
            "global_alloc::alloc(layout)",
        ]
        for expr in memory_patterns:
            with pytest.raises(SecurityError):
                validate_expression_security(expr)

    def test_concurrency_patterns(self):
        """Test that potentially dangerous Rust concurrency patterns are blocked."""
        concurrency_patterns = [
            "thread::spawn(closure)",
            "std::sync::Mutex::new(data)",
            "std::sync::atomic::AtomicBool::new(false)",
            "lazy_static! { static ref VAR: Type = value; }",
        ]
        for expr in concurrency_patterns:
            with pytest.raises(SecurityError):
                validate_expression_security(expr)

    def test_process_operations(self):
        """Test that Rust process operations are blocked."""
        process_patterns = [
            'std::process::Command::new("ls")',
            "std::process::exit(1)",
            'std::env::var("PATH")',
            'std::env::set_var("VAR", "value")',
        ]
        for expr in process_patterns:
            with pytest.raises(SecurityError):
                validate_expression_security(expr)


class TestAdvancedAttackVectors:
    """Test advanced attack vector detection."""

    def test_encoding_attacks(self):
        """Test detection of encoding-based attacks."""
        encoding_attacks = [
            "beta + \\x41",  # Hex escape sequences
            "\\x48\\x65\\x6c\\x6c\\x6f",
            "beta + \\u0041",  # Unicode escape sequences
            "\\u0048\\u0065\\u006c\\u006c\\u006f",
            "aGVsbG8gd29ybGQgdGhpcyBpcyBhIGxvbmcgc3RyaW5n",  # Base64-like
            "VGhpcyBpcyBhIHN1c3BpY2lvdXMgcGF5bG9hZA==",
        ]
        for expr in encoding_attacks:
            with pytest.raises(SecurityError, match="Encoding attack detected"):
                validate_expression_security(expr)

    def test_unicode_attacks(self):
        """Test detection of Unicode-based attacks."""
        unicode_attacks = [
            "а * β",  # Cyrillic 'a' (homograph attack)
            "е + γ",  # Cyrillic 'e'
            "οmega",  # Greek omicron instead of 'o'
            "var\u200b",  # Zero-width space
            "test\u200c",  # Zero-width non-joiner
        ]
        for expr in unicode_attacks:
            with pytest.raises(SecurityError):
                validate_expression_security(expr)

    def test_format_string_attacks(self):
        """Test detection of format string attacks."""
        format_attacks = [
            "rate * {0}",
            "beta + {variable}",
            "gamma * %s",
            "delta / %d",
            "%x + alpha",
            "value.format(args)",
        ]
        for expr in format_attacks:
            with pytest.raises(SecurityError):
                validate_expression_security(expr)

    def test_network_references(self):
        """Test detection of network references."""
        network_refs = [
            "http://evil.com/payload",
            "https://malicious.example.org",
            "ftp://dangerous.com",
            "192.168.1.1",
            "10.0.0.1",
            "127.0.0.1",
        ]
        for expr in network_refs:
            with pytest.raises(SecurityError):
                validate_expression_security(expr)

    def test_mathematical_injection(self):
        """Test detection of mathematical injection attempts."""
        math_injection_attacks = [
            "123eVAL",  # Suspicious 'e' followed by letters
            "456EXEC",  # Suspicious numeric followed by dangerous keyword
            "789sys",  # Number followed by system keyword
        ]
        for expr in math_injection_attacks:
            with pytest.raises(SecurityError):
                validate_expression_security(expr)

    def test_repetitive_patterns(self):
        """Test detection of repetitive patterns that could indicate attacks."""
        repetitive_attacks = [
            "abc_xyz_" * 50,  # Very repetitive pattern
            "A" * 200,  # Single character repetition
            "def" * 100,  # Keyword repetition
        ]
        for expr in repetitive_attacks:
            with pytest.raises(SecurityError):
                validate_expression_security(expr)


class TestExpressionLimitsAndComplexity:
    """Test expression limits and complexity validation."""

    def test_expression_length_limits(self):
        """Test expression length validation."""
        config = SecurityConfig(max_expression_length=100)
        # Valid length
        short_expr = "beta * gamma"
        validate_expression_security(short_expr, config)
        # Too long
        long_expr = "x + " * 50  # Creates expression longer than 100 chars
        with pytest.raises(SecurityError, match="Expression too long"):
            validate_expression_security(long_expr, config)

    def test_nesting_depth_limits(self):
        """Test nesting depth validation."""
        config = SecurityConfig(max_nesting_depth=5)
        # Valid nesting
        valid_nested = "((((x))))"  # 4 levels
        validate_expression_security(valid_nested, config)
        # Too deep
        deep_nested = "(" * 8 + "x" + ")" * 8  # 8 levels
        with pytest.raises(SecurityError, match="Too many nested"):
            validate_expression_security(deep_nested, config)

    def test_variable_name_length_limits(self):
        """Test variable name length limits."""
        config = SecurityConfig(max_variable_name_length=20)
        # Valid variable name
        validate_expression_security("short_name", config)
        # Too long variable name
        long_var = "very_long_variable_name_that_exceeds_limit"
        with pytest.raises(SecurityError, match="Variable name too long"):
            validate_expression_security(long_var, config)

    def test_function_call_limits(self):
        """Test function call count limits."""
        # Create expression with many function calls
        many_functions = " + ".join([f"sin(x{i})" for i in range(30)])
        with pytest.raises(SecurityError, match="Too many function calls"):
            validate_expression_security(many_functions)

    def test_balanced_delimiters(self):
        """Test balanced delimiter validation."""
        unbalanced_expressions = [
            "(x + y",  # Missing closing paren
            "x + y)",  # Missing opening paren
            "((x + y)",  # Unmatched opening paren
            "(x + y))",  # Unmatched closing paren
            ")x + y(",  # Reversed parens
            "sin(x + cos(y)",  # Unbalanced in function call
        ]
        for expr in unbalanced_expressions:
            with pytest.raises(SecurityError, match="Unmatched"):
                validate_expression_security(expr)

    def test_invalid_numeric_values(self):
        """Test validation of invalid numeric formats."""
        invalid_numbers = [
            "3..14",  # Double decimal point
            "1e",  # Incomplete scientific notation
            "e10",  # Missing base in scientific notation
            "1e1e1",  # Multiple 'e' characters
            "123abc",  # Mixed alphanumeric
        ]
        for num in invalid_numbers:
            with pytest.raises(SecurityError):
                validate_expression_security(num)


class TestDisallowedCharacters:
    """Test disallowed character detection."""

    def test_disallowed_special_characters(self):
        """Test that disallowed special characters are rejected."""
        disallowed_chars = [
            "beta;gamma",  # Semicolon
            "alpha&beta",  # Ampersand
            "gamma|delta",  # Pipe
            "value`backdoor`",  # Backtick
            "param$injection",  # Dollar sign
            "var#comment",  # Hash symbol
            "string'quote",  # Single quote
            'string"quote',  # Double quote
            "list[index]",  # Square brackets
            "dict{key: value}",  # Curly braces
            "value @ matrix",  # At symbol
            "~variable",  # Tilde
        ]
        for expr in disallowed_chars:
            with pytest.raises(SecurityError, match="Disallowed character"):
                validate_expression_security(expr)

    def test_allowed_characters(self):
        """Test that allowed characters are accepted."""
        allowed_expressions = [
            "beta_gamma",  # Underscore
            "alpha123",  # Alphanumeric
            "value + 2.5",  # Decimal point in numbers
            "func(arg1, arg2)",  # Parentheses and comma
            "x * y / z",  # Basic operators
            "rate**2",  # Double asterisk (power)
            "a % b",  # Modulo operator
        ]
        for expr in allowed_expressions:
            validate_expression_security(expr)


class TestStrictModeValidation:
    """Test strict mode validation features."""

    def test_strict_mode_variable_patterns(self):
        """Test strict mode variable name pattern detection."""
        suspicious_names = [
            "eval_something",
            "something_exec",
            "import_data",
            "system_call",
            "exec_command",
            "eval_expression",
            "file_open",
            "socket_connect",
        ]
        for name in suspicious_names:
            with pytest.raises(SecurityError, match="Suspicious variable name"):
                validate_expression_security(name)

    def test_strict_mode_numeric_limits(self):
        """Test strict mode numeric value limits."""
        # Very large numbers that might be problematic
        large_numbers = [
            "1e50",
            "999999999999999",
            "-1e100",
        ]
        for num in large_numbers:
            with pytest.raises(SecurityError, match="Numeric value too large"):
                validate_expression_security(num)
