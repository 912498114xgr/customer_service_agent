"""本地客服订单 MCP Server，只提供只读演示数据。"""

from mcp.server.fastmcp import FastMCP

server = FastMCP("customer-care-order-tools")


@server.tool()
def get_order_status(order_id: str) -> dict[str, str]:
    """查询订单当前状态。order_id 必须由用户请求或已确认上下文提供。"""
    if not order_id.strip():
        raise ValueError("order_id 不能为空")
    return {"order_id": order_id, "status": "delivered"}


@server.tool()
def get_refund_status(order_id: str) -> dict[str, str | int]:
    """查询订单退款状态和预计到账时间，不会发起退款。"""
    if not order_id.strip():
        raise ValueError("order_id 不能为空")
    return {"order_id": order_id, "status": "processing", "estimated_days": 3}


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
