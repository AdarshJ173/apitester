"""
Tests for tool_parser.py module
"""

import json
import pytest
from unittest.mock import patch, MagicMock


class TestToolParserPatterns:
    """Tests for ToolParser PATTERNS dictionary"""

    def test_patterns_has_all_tools(self):
        """Test PATTERNS includes all tool types"""
        from tool_parser import ToolParser

        expected_tools = [
            "create_file",
            "read_file",
            "update_file",
            "delete_file",
            "list_directory",
            "execute_command",
        ]

        for tool in expected_tools:
            assert tool in ToolParser.PATTERNS, f"Missing pattern for {tool}"

    def test_patterns_are_lists(self):
        """Test each tool has a list of patterns"""
        from tool_parser import ToolParser

        for tool_name, patterns in ToolParser.PATTERNS.items():
            assert isinstance(patterns, list), f"{tool_name} patterns should be a list"
            assert len(patterns) > 0, f"{tool_name} should have at least one pattern"


class TestToolParserParseTextResponse:
    """Tests for ToolParser.parse_text_response method"""

    def test_empty_text_returns_empty(self):
        """Test parsing empty text returns empty list"""
        from tool_parser import ToolParser

        cleaned, tools = ToolParser.parse_text_response("")

        assert cleaned == ""
        assert tools == []

    def test_text_without_tools(self):
        """Test parsing text without tool calls"""
        from tool_parser import ToolParser

        text = "This is just a regular response"
        cleaned, tools = ToolParser.parse_text_response(text)

        assert "This is just a regular response" in cleaned
        assert tools == []

    def test_parse_create_file_tool(self):
        """Test parsing create_file tool call"""
        from tool_parser import ToolParser

        text = 'create_file({"path": "workspace/test.txt", "content": "hello"})'
        cleaned, tools = ToolParser.parse_text_response(text)

        assert len(tools) == 1
        assert tools[0]["tool"] == "create_file"
        assert tools[0]["arguments"]["path"] == "workspace/test.txt"
        assert tools[0]["arguments"]["content"] == "hello"

    def test_parse_read_file_tool(self):
        """Test parsing read_file tool call"""
        from tool_parser import ToolParser

        text = 'read_file({"path": "workspace/test.txt"})'
        cleaned, tools = ToolParser.parse_text_response(text)

        assert len(tools) == 1
        assert tools[0]["tool"] == "read_file"
        assert tools[0]["arguments"]["path"] == "workspace/test.txt"

    def test_parse_multiple_tools(self):
        """Test parsing multiple tool calls"""
        from tool_parser import ToolParser

        text = """First create a file: create_file({"path": "workspace/a.txt", "content": "A"})
                  Then read it: read_file({"path": "workspace/a.txt"})"""
        cleaned, tools = ToolParser.parse_text_response(text)

        assert len(tools) == 2
        assert tools[0]["tool"] == "create_file"
        assert tools[1]["tool"] == "read_file"

    def test_removes_tool_calls_from_cleaned_text(self):
        """Test tool calls are removed from cleaned response"""
        from tool_parser import ToolParser

        text = 'Hello create_file({"path": "test.txt", "content": "hi"}) World'
        cleaned, tools = ToolParser.parse_text_response(text)

        assert "create_file" not in cleaned
        assert "Hello" in cleaned
        assert "World" in cleaned

    def test_handles_invalid_json_gracefully(self):
        """Test handling of invalid JSON in tool calls"""
        from tool_parser import ToolParser

        text = 'create_file({"path": "test.txt", invalid})'
        cleaned, tools = ToolParser.parse_text_response(text)

        # Should not crash, may or may not extract the tool
        assert isinstance(tools, list)


class TestToolParserExtractArguments:
    """Tests for ToolParser._extract_arguments method"""

    def test_extract_json_arguments(self):
        """Test extracting JSON arguments"""
        from tool_parser import ToolParser

        match_text = 'create_file({"path": "test.txt", "content": "hello"})'
        result = ToolParser._extract_arguments(match_text, "create_file")

        assert result is not None
        assert result["path"] == "test.txt"
        assert result["content"] == "hello"

    def test_extract_arguments_with_workspace_prefix(self):
        """Test path gets workspace prefix added"""
        from tool_parser import ToolParser

        match_text = 'create_file({"path": "test.txt", "content": "hello"})'
        result = ToolParser._extract_arguments(match_text, "create_file")

        assert result["path"] == "workspace/test.txt"

    def test_extract_arguments_preserves_existing_workspace_prefix(self):
        """Test path with existing workspace prefix is preserved"""
        from tool_parser import ToolParser

        match_text = 'create_file({"path": "workspace/test.txt", "content": "hello"})'
        result = ToolParser._extract_arguments(match_text, "create_file")

        assert result["path"] == "workspace/test.txt"

    def test_extract_arguments_preserves_absolute_paths(self):
        """Test absolute paths are preserved"""
        from tool_parser import ToolParser

        match_text = 'create_file({"path": "/absolute/path.txt", "content": "hello"})'
        result = ToolParser._extract_arguments(match_text, "create_file")

        assert result["path"] == "/absolute/path.txt"


class TestToolParserFallbackExtraction:
    """Tests for ToolParser._fallback_extraction method"""

    def test_fallback_create_file(self):
        """Test fallback extraction for create_file"""
        from tool_parser import ToolParser

        text = 'path: "test.txt", content: "hello world"'
        result = ToolParser._fallback_extraction(text, "create_file")

        assert result is not None
        assert result["path"] == "workspace/test.txt"
        assert result["content"] == "hello world"

    def test_fallback_read_file_with_line_range(self):
        """Test fallback extraction for read_file with line range"""
        from tool_parser import ToolParser

        text = 'path: "test.txt", start_line: 5, end_line: 10'
        result = ToolParser._fallback_extraction(text, "read_file")

        assert result is not None
        assert result["path"] == "workspace/test.txt"
        assert result["start_line"] == 5
        assert result["end_line"] == 10

    def test_fallback_update_file(self):
        """Test fallback extraction for update_file"""
        from tool_parser import ToolParser

        text = 'path: "test.txt", content: "new content"'
        result = ToolParser._fallback_extraction(text, "update_file")

        assert result is not None
        assert result["path"] == "workspace/test.txt"
        assert result["content"] == "new content"

    def test_fallback_list_directory(self):
        """Test fallback extraction for list_directory"""
        from tool_parser import ToolParser

        text = 'path: "workspace/"'
        result = ToolParser._fallback_extraction(text, "list_directory")

        assert result is not None
        assert result["path"] == "workspace/"

    def test_fallback_list_directory_default(self):
        """Test fallback extraction for list_directory defaults to current dir"""
        from tool_parser import ToolParser

        text = "no path here"
        result = ToolParser._fallback_extraction(text, "list_directory")

        assert result is not None
        assert result["path"] == "."

    def test_fallback_execute_command(self):
        """Test fallback extraction for execute_command"""
        from tool_parser import ToolParser

        text = 'command: "ls -la"'
        result = ToolParser._fallback_extraction(text, "execute_command")

        assert result is not None
        assert result["command"] == "ls -la"

    def test_fallback_unknown_tool(self):
        """Test fallback extraction for unknown tool returns None"""
        from tool_parser import ToolParser

        text = "some arguments"
        result = ToolParser._fallback_extraction(text, "unknown_tool")

        assert result is None


class TestToolParserNormalizePath:
    """Tests for ToolParser._normalize_path method"""

    def test_adds_workspace_prefix(self):
        """Test adds workspace/ prefix to relative paths"""
        from tool_parser import ToolParser

        result = ToolParser._normalize_path("test.txt")
        assert result == "workspace/test.txt"

    def test_preserves_workspace_prefix(self):
        """Test preserves existing workspace/ prefix"""
        from tool_parser import ToolParser

        result = ToolParser._normalize_path("workspace/test.txt")
        assert result == "workspace/test.txt"

    def test_preserves_absolute_unix_path(self):
        """Test preserves absolute Unix paths"""
        from tool_parser import ToolParser

        result = ToolParser._normalize_path("/home/user/test.txt")
        assert result == "/home/user/test.txt"

    def test_preserves_absolute_windows_path(self):
        """Test preserves absolute Windows paths"""
        from tool_parser import ToolParser

        result = ToolParser._normalize_path("C:\\Users\\test.txt")
        assert result == "C:\\Users\\test.txt"

    def test_handles_empty_path(self):
        """Test handles empty path"""
        from tool_parser import ToolParser

        result = ToolParser._normalize_path("")
        assert result == "workspace/"

    def test_handles_whitespace(self):
        """Test handles paths with whitespace"""
        from tool_parser import ToolParser

        result = ToolParser._normalize_path("  test.txt  ")
        assert result == "workspace/test.txt"


class TestToolParserCreateToolSystemPrompt:
    """Tests for ToolParser.create_tool_system_prompt method"""

    def test_returns_string(self):
        """Test returns a string"""
        from tool_parser import ToolParser

        prompt = ToolParser.create_tool_system_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_includes_all_tools(self):
        """Test prompt includes all available tools"""
        from tool_parser import ToolParser

        prompt = ToolParser.create_tool_system_prompt()

        assert "create_file" in prompt
        assert "read_file" in prompt
        assert "update_file" in prompt
        assert "delete_file" in prompt
        assert "list_directory" in prompt
        assert "execute_command" in prompt

    def test_includes_critical_rules(self):
        """Test prompt includes critical usage rules"""
        from tool_parser import ToolParser

        prompt = ToolParser.create_tool_system_prompt()

        assert "workspace/" in prompt
        assert "CRITICAL RULES" in prompt


class TestUniversalToolExecutor:
    """Tests for UniversalToolExecutor class"""

    def test_init_requires_execute_func(self):
        """Test initialization requires execute_tool function"""
        from tool_parser import UniversalToolExecutor

        mock_func = MagicMock()
        executor = UniversalToolExecutor(mock_func)

        assert executor.execute_tool == mock_func

    def test_process_response_no_tools(self):
        """Test processing response without tools"""
        from tool_parser import UniversalToolExecutor

        mock_execute = MagicMock()
        executor = UniversalToolExecutor(mock_execute)

        text = "Just a regular response"
        cleaned, results = executor.process_response(text)

        assert "Just a regular response" in cleaned
        assert results == []
        mock_execute.assert_not_called()

    def test_process_response_with_tools(self):
        """Test processing response with tool calls"""
        from tool_parser import UniversalToolExecutor

        mock_execute = MagicMock(return_value={"success": True})
        executor = UniversalToolExecutor(mock_execute)

        text = 'create_file({"path": "test.txt", "content": "hello"})'
        cleaned, results = executor.process_response(text)

        assert len(results) == 1
        assert results[0]["tool"] == "create_file"
        assert results[0]["result"]["success"] is True
        mock_execute.assert_called_once()

    def test_process_response_multiple_tools(self):
        """Test processing response with multiple tool calls"""
        from tool_parser import UniversalToolExecutor

        mock_execute = MagicMock(return_value={"success": True})
        executor = UniversalToolExecutor(mock_execute)

        text = """create_file({"path": "a.txt", "content": "A"})
                  read_file({"path": "a.txt"})"""
        cleaned, results = executor.process_response(text)

        assert len(results) == 2
        assert mock_execute.call_count == 2

    def test_process_response_tool_failure(self):
        """Test processing response when tool execution fails"""
        from tool_parser import UniversalToolExecutor

        mock_execute = MagicMock(return_value={"success": False, "error": "Failed"})
        executor = UniversalToolExecutor(mock_execute)

        text = 'create_file({"path": "test.txt", "content": "hello"})'
        cleaned, results = executor.process_response(text)

        assert len(results) == 1
        assert results[0]["result"]["success"] is False
        assert results[0]["result"]["error"] == "Failed"

    @patch("builtins.print")
    def test_process_response_prints_success(self, mock_print):
        """Test processing response prints success status"""
        from tool_parser import UniversalToolExecutor

        mock_execute = MagicMock(return_value={"success": True})
        executor = UniversalToolExecutor(mock_execute)

        text = 'create_file({"path": "test.txt", "content": "hello"})'
        executor.process_response(text)

        # Should print success status
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("OK" in call or "create_file" in call for call in print_calls)

    @patch("builtins.print")
    def test_process_response_prints_failure(self, mock_print):
        """Test processing response prints failure status"""
        from tool_parser import UniversalToolExecutor

        mock_execute = MagicMock(return_value={"success": False, "error": "Failed"})
        executor = UniversalToolExecutor(mock_execute)

        text = 'create_file({"path": "test.txt", "content": "hello"})'
        executor.process_response(text)

        # Should print error status
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("ERR" in call or "Failed" in call for call in print_calls)
