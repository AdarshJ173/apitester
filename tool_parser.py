"""
Universal Tool Parser for Text-Based AI Responses
Enables tool use with ANY AI provider by parsing structured commands from text
"""

import re
import json
from typing import Any, Dict, List, Optional, Tuple


class ToolParser:
    """Parse tool commands from AI text responses"""

    # Patterns to detect tool invocations in text
    PATTERNS = {
        "create_file": [
            r"(?:🔧\s*)?create_file\s*\(\s*\{[^}]+\}\s*\)",
            r"(?:🔧\s*)?create_file:\s*\{[^}]+\}",
            r'(?:create|write)\s+file:\s*(["\']?)(.+?)\1\s*with\s+content:\s*(.+)',
            r"```tool\s*\n?create_file\s*\n?([^`]+)```",
        ],
        "read_file": [
            r"(?:🔧\s*)?read_file\s*\(\s*\{[^}]+\}\s*\)",
            r"(?:🔧\s*)?read_file:\s*\{[^}]+\}",
            r'(?:read|show)\s+file:\s*(["\']?)(.+?)\1',
            r"```tool\s*\n?read_file\s*\n?([^`]+)```",
        ],
        "update_file": [
            r"(?:🔧\s*)?update_file\s*\(\s*\{[^}]+\}\s*\)",
            r"(?:🔧\s*)?update_file:\s*\{[^}]+\}",
            r'(?:update|modify)\s+file:\s*(["\']?)(.+?)\1',
            r"```tool\s*\n?update_file\s*\n?([^`]+)```",
        ],
        "delete_file": [
            r"(?:🔧\s*)?delete_file\s*\(\s*\{[^}]+\}\s*\)",
            r"(?:🔧\s*)?delete_file:\s*\{[^}]+\}",
            r'(?:delete|remove)\s+file:\s*(["\']?)(.+?)\1',
            r"```tool\s*\n?delete_file\s*\n?([^`]+)```",
        ],
        "list_directory": [
            r"(?:🔧\s*)?list_directory\s*\(\s*\{[^}]*\}\s*\)",
            r"(?:🔧\s*)?list_directory:\s*\{[^}]*\}",
            r'(?:list|show)\s+(?:files?|directory|dir)(?:\s+in)?:?\s*(["\']?)([^\n\']*)\1',
            r"```tool\s*\n?list_directory\s*\n?([^`]+)```",
        ],
        "execute_command": [
            r"(?:🔧\s*)?execute_command\s*\(\s*\{[^}]+\}\s*\)",
            r"(?:🔧\s*)?execute_command:\s*\{[^}]+\}",
            r'(?:run|execute)\s+command:\s*(["\']?)(.+?)\1',
            r"```tool\s*\n?execute_command\s*\n?([^`]+)```",
        ],
    }

    @classmethod
    def parse_text_response(cls, text: str) -> Tuple[str, List[Dict]]:
        """
        Parse AI text response for tool commands

        Returns:
            (cleaned_response, list_of_tool_calls)
            Each tool_call is {"tool": str, "arguments": dict}
        """
        if not text:
            return text, []

        tool_calls = []
        cleaned_text = text

        for tool_name, patterns in cls.PATTERNS.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE | re.DOTALL))

                for match in matches:
                    try:
                        # Try to extract JSON arguments
                        args = cls._extract_arguments(match.group(0), tool_name)
                        if args:
                            tool_calls.append(
                                {
                                    "tool": tool_name,
                                    "arguments": args,
                                    "match_text": match.group(0),
                                }
                            )
                            # Remove the tool call from cleaned text
                            cleaned_text = cleaned_text.replace(match.group(0), "")
                    except Exception:
                        continue

        # Clean up extra whitespace
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text.strip())

        return cleaned_text, tool_calls

    @classmethod
    def _extract_arguments(cls, match_text: str, tool_name: str) -> Optional[Dict]:
        """Extract arguments from matched text"""

        # Try to find JSON object
        json_match = re.search(
            r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", match_text, re.DOTALL
        )
        if json_match:
            try:
                args = json.loads(json_match.group(0))
                # Normalize path for file operations
                if "path" in args and tool_name in [
                    "create_file",
                    "read_file",
                    "update_file",
                    "delete_file",
                ]:
                    args["path"] = cls._normalize_path(args["path"])
                return args
            except json.JSONDecodeError:
                pass

        # Fallback: extract based on tool type
        return cls._fallback_extraction(match_text, tool_name)

    @classmethod
    def _normalize_path(cls, path: str) -> str:
        """Ensure path starts with workspace/ for file operations"""
        if not path:
            return "workspace/"
        path = path.strip()
        # Don't modify absolute paths or paths that already have workspace/
        if (
            path.startswith("workspace/")
            or path.startswith("/")
            or path.startswith("\\")
        ):
            return path
        # Add workspace/ prefix
        return f"workspace/{path}"

    @classmethod
    def _fallback_extraction(cls, text: str, tool_name: str) -> Optional[Dict]:
        """Extract arguments using regex when JSON parsing fails"""

        if tool_name == "create_file":
            path_match = re.search(r'["\']?path["\']?\s*:\s*["\']([^"\']+)["\']', text)
            content_match = re.search(
                r'["\']?content["\']?\s*:\s*["\']([^"\']*)["\']', text, re.DOTALL
            )
            if path_match:
                return {
                    "path": cls._normalize_path(path_match.group(1)),
                    "content": content_match.group(1) if content_match else "",
                }

        elif tool_name in ["read_file", "update_file", "delete_file"]:
            path_match = re.search(r'["\']?path["\']?\s*:\s*["\']([^"\']+)["\']', text)
            if path_match:
                args: Dict[str, Any] = {
                    "path": cls._normalize_path(path_match.group(1))
                }
                if tool_name == "read_file":
                    start_match = re.search(r'["\']?start_line["\']?\s*:\s*(\d+)', text)
                    end_match = re.search(r'["\']?end_line["\']?\s*:\s*(-?\d+)', text)
                    if start_match:
                        args["start_line"] = int(start_match.group(1))
                    if end_match:
                        args["end_line"] = int(end_match.group(1))
                elif tool_name == "update_file":
                    content_match = re.search(
                        r'["\']?content["\']?\s*:\s*["\']([^"\']*)["\']',
                        text,
                        re.DOTALL,
                    )
                    if content_match:
                        args["content"] = content_match.group(1)
                return args

        elif tool_name == "list_directory":
            path_match = re.search(r'["\']?path["\']?\s*:\s*["\']([^"\']*)["\']', text)
            return {"path": path_match.group(1) if path_match else "."}

        elif tool_name == "execute_command":
            cmd_match = re.search(
                r'["\']?command["\']?\s*:\s*["\']([^"\']+)["\']', text
            )
            if cmd_match:
                return {"command": cmd_match.group(1)}

        return None

    @classmethod
    def create_tool_system_prompt(cls) -> str:
        """Create a system prompt that instructs the AI to use tools"""
        return """You are an AI assistant with access to file system tools.

AVAILABLE TOOLS:
1. create_file - Create a new file with content
   Usage: create_file({"path": "workspace/file.txt", "content": "file content"})

2. read_file - Read content from a file
   Usage: read_file({"path": "workspace/file.txt"})

3. update_file - Update existing file content
   Usage: update_file({"path": "workspace/file.txt", "content": "new content"})

4. delete_file - Delete a file
   Usage: delete_file({"path": "workspace/file.txt"})

5. list_directory - List directory contents
   Usage: list_directory({"path": "."})

6. execute_command - Execute safe shell commands
   Usage: execute_command({"command": "ls -la"})

CRITICAL RULES:
1. ALL file paths MUST start with "workspace/" - NO EXCEPTIONS
2. When creating a file, remember the EXACT path you used
3. When deleting/updating, use the SAME path you used to create it
4. Always output the tool call EXACTLY in the format shown above
5. After tool calls, provide your response to the user

Example - Creating then deleting:
User: Create a file called test.txt
create_file({"path": "workspace/test.txt", "content": "Hello"})
User: Delete it
delete_file({"path": "workspace/test.txt"})

BAD EXAMPLES (these will fail):
create_file({"path": "test.txt", ...})  # Missing workspace/
delete_file({"path": "test.txt"})       # Missing workspace/"""


class UniversalToolExecutor:
    """Execute tools parsed from text for any AI provider"""

    def __init__(self, execute_tool_func):
        self.execute_tool = execute_tool_func

    def process_response(self, text: str) -> Tuple[str, List[Dict]]:
        """
        Process AI response: parse tools, execute them, return results

        Returns:
            (response_text, execution_results)
        """
        from agent_tools import execute_tool

        cleaned_text, tool_calls = ToolParser.parse_text_response(text)
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call["tool"]
            arguments = tool_call["arguments"]

            # Execute the tool
            result = execute_tool(tool_name, arguments)
            results.append(
                {"tool": tool_name, "arguments": arguments, "result": result}
            )

            # Print execution status (use ASCII-safe characters for Windows)
            if result.get("success"):
                print(f"[OK] {tool_name}")
            else:
                print(f"[ERR] {tool_name}: {result.get('error')}")

        return cleaned_text, results
