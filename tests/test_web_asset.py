from pathlib import Path


def test_customer_workbench_static_asset_exists() -> None:
    path = Path(__file__).parents[1] / "customer_care" / "web" / "index.html"

    content = path.read_text(encoding="utf-8")

    assert "/v1/chat" in content
    assert "knowledge_sources" in content
