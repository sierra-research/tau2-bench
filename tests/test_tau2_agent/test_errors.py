"""Unit tests for tau2_agent error types."""

import pytest

from tau2_agent.errors import ErrorCode, EvaluationError


class TestErrorCode:
    """Tests for ErrorCode enum."""

    def test_error_codes_are_strings(self):
        """ErrorCode values should be string-serializable."""
        assert ErrorCode.MISSING_HEADER.value == "MISSING_HEADER"
        assert ErrorCode.INVALID_AUTH.value == "INVALID_AUTH"
        assert ErrorCode.USER_LLM_AUTH_FAILED.value == "USER_LLM_AUTH_FAILED"
        assert ErrorCode.LIMIT_EXCEEDED.value == "LIMIT_EXCEEDED"
        assert ErrorCode.EVALUATION_FAILED.value == "EVALUATION_FAILED"

    def test_all_error_codes_defined(self):
        """All expected error codes should be defined."""
        expected_codes = {
            "MISSING_HEADER",
            "INVALID_AUTH",
            "USER_LLM_AUTH_FAILED",
            "LIMIT_EXCEEDED",
            "EVALUATION_FAILED",
        }
        actual_codes = {code.value for code in ErrorCode}
        assert actual_codes == expected_codes


class TestEvaluationError:
    """Tests for EvaluationError dataclass."""

    def test_to_dict_without_details(self):
        """to_dict should return error and code without details."""
        error = EvaluationError(
            code=ErrorCode.MISSING_HEADER,
            message="Missing X-User-LLM-Model header",
        )
        result = error.to_dict()

        assert result == {
            "error": "Missing X-User-LLM-Model header",
            "code": "MISSING_HEADER",
        }
        assert "details" not in result

    def test_to_dict_with_details(self):
        """to_dict should include details when provided."""
        error = EvaluationError(
            code=ErrorCode.LIMIT_EXCEEDED,
            message="num_tasks must be between 1 and 30",
            details={"num_tasks": 50, "max_tasks": 30},
        )
        result = error.to_dict()

        assert result == {
            "error": "num_tasks must be between 1 and 30",
            "code": "LIMIT_EXCEEDED",
            "details": {"num_tasks": 50, "max_tasks": 30},
        }

    def test_to_dict_with_empty_details(self):
        """to_dict should include empty details dict when explicitly provided."""
        error = EvaluationError(
            code=ErrorCode.EVALUATION_FAILED,
            message="Evaluation failed",
            details={},
        )
        result = error.to_dict()

        assert result == {
            "error": "Evaluation failed",
            "code": "EVALUATION_FAILED",
            "details": {},
        }

    def test_to_dict_with_none_details(self):
        """to_dict should not include details when None."""
        error = EvaluationError(
            code=ErrorCode.INVALID_AUTH,
            message="Invalid authorization",
            details=None,
        )
        result = error.to_dict()

        assert "details" not in result

    def test_str_without_details(self):
        """String representation without details."""
        error = EvaluationError(
            code=ErrorCode.USER_LLM_AUTH_FAILED,
            message="User LLM authentication failed",
        )
        assert str(error) == "[USER_LLM_AUTH_FAILED] User LLM authentication failed"

    def test_str_with_details(self):
        """String representation with details."""
        error = EvaluationError(
            code=ErrorCode.LIMIT_EXCEEDED,
            message="num_tasks exceeded",
            details={"value": 100},
        )
        assert "[LIMIT_EXCEEDED]" in str(error)
        assert "num_tasks exceeded" in str(error)
        assert "{'value': 100}" in str(error)

    @pytest.mark.parametrize(
        "code,message",
        [
            (ErrorCode.MISSING_HEADER, "Missing X-User-LLM-Model header"),
            (ErrorCode.MISSING_HEADER, "Missing X-User-LLM-API-Key header"),
            (ErrorCode.INVALID_AUTH, "Invalid authorization"),
            (ErrorCode.USER_LLM_AUTH_FAILED, "User LLM authentication failed"),
            (ErrorCode.LIMIT_EXCEEDED, "num_tasks must be between 1 and 30"),
            (ErrorCode.LIMIT_EXCEEDED, "num_trials must be between 1 and 3"),
            (ErrorCode.EVALUATION_FAILED, "Evaluation failed: connection timeout"),
        ],
    )
    def test_common_error_scenarios(self, code: ErrorCode, message: str):
        """Test common error scenarios are representable."""
        error = EvaluationError(code=code, message=message)
        result = error.to_dict()

        assert result["error"] == message
        assert result["code"] == code.value
