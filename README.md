# COMP9414-Tutorials
Traditional approach: Cross your fingers 🤞
system_prompt = "You are a helpful assistant. Please follow these 47 rules..."

# Parlant approach: Ensured compliance ✅
await agent.create_guideline(
    condition="Customer asks about refunds",
    action="Check order status first to see if eligible",
    tools=[check_order_status],
)
