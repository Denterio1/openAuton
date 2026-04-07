from tools.web_search import create_web_search_tool

tool = create_web_search_tool()
result = tool("transformer architecture", max_results=3)
print("Status:", result["status"])
print("Number of results:", result["num_results"])
print("Formatted:\n", result["formatted"][:500])