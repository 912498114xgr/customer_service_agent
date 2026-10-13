import pytest

from customer_care.api import create_app


def test_api_reports_optional_dependency_requirement_when_fastapi_missing() -> None:
    try:
        import fastapi  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="uv sync --extra web"):
            create_app()
