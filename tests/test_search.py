"""Exercise the real MCP SDK, provider dispatcher and LangGraph result consumer."""

import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx

import config
import memory
import orchestrator
from agents.search import execute_search
from utils.parallel_search import ENDPOINT, USER_AGENT, ParallelSearchClient


ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://docs.python.org/3/library/asyncio-task.html?language=en#timeouts"
SEARCH_DATA = {"results": [{
    "url": SOURCE, "title": "Coroutines and Tasks",
    "excerpts": ["asyncio.timeout bounds the time spent awaiting an operation."],
}]}
REAL_ASYNC_CLIENT = httpx.AsyncClient


class SearchConfigTests(unittest.TestCase):
    def resolve(self, values, saved=""):
        # Copy the actual configuration into an isolated installation with its own
        # .env; neither ambient environment keys nor a developer's .env can leak in.
        with tempfile.TemporaryDirectory() as directory:
            shutil.copy(ROOT / "config.py", directory)
            Path(directory, ".env").write_text(saved)
            env = {"PATH": os.environ.get("PATH", ""), **values}
            return subprocess.run(
                [sys.executable, "-c", "import config; print(config.get_search_provider())"],
                cwd=directory, env=env, capture_output=True, text=True,
            )

    def test_fresh_install_and_explicit_choices(self):
        for values, expected in [
            ({}, "parallel"),
            ({"SONAR_API_KEY": "saved-key"}, "sonar"),
            ({"SONAR_API_KEY": "saved-key", "SEARCH_PROVIDER": " Parallel "}, "parallel"),
            ({"SEARCH_PROVIDER": "sonar"}, "sonar"),
            ({"SEARCH_PROVIDER": "none", "SONAR_API_KEY": "saved-key"}, "none"),
            ({"SEARCH_PROVIDER": ""}, "parallel"),
        ]:
            with self.subTest(values=values):
                result = self.resolve(values)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), expected)

    def test_saved_dotenv_and_environment_override(self):
        saved = "SONAR_API_KEY=saved-key\nSEARCH_PROVIDER=sonar\n"
        self.assertEqual(self.resolve({}, saved).stdout.strip(), "sonar")
        self.assertEqual(self.resolve({"SEARCH_PROVIDER": "parallel"}, saved).stdout.strip(), "parallel")
        self.assertEqual(self.resolve({"SEARCH_PROVIDER": "none"}, saved).stdout.strip(), "none")

    def test_invalid_choice_does_not_fall_back(self):
        result = self.resolve({"SEARCH_PROVIDER": "typo"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SEARCH_PROVIDER must be", result.stderr)


class SearchRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.clients = []
        self.mode = "structured"
        self.data = SEARCH_DATA
        self.call_started = asyncio.Event()
        self.patches = [
            patch.object(config, "SEARCH_PROVIDER", ""),
            patch.object(config, "SONAR_API_KEY", ""),
            patch.object(config, "OPENROUTER_API_KEY", "fixture-key"),
            patch("httpx.AsyncClient", self.client_factory),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    async def asyncTearDown(self):
        self.assertTrue(all(client.is_closed for client in self.clients))

    def client_factory(self, **kwargs):
        client = REAL_ASYNC_CLIENT(transport=httpx.MockTransport(self.handle), **kwargs)
        self.clients.append(client)
        return client

    async def handle(self, request):
        self.requests.append(request)
        message = json.loads(request.content)
        if request.url.host == "api.perplexity.ai":
            self.assertEqual(str(request.url), "https://api.perplexity.ai/chat/completions")
            self.assertEqual(request.headers["Authorization"], "Bearer saved-sonar-key")
            self.assertEqual(message["model"], "sonar-pro")
            if self.mode == "sonar_error":
                return httpx.Response(401, text="Invalid Sonar credentials")
            return httpx.Response(200, json={"choices": [{"message": {"content": "Sonar answer"}}], "citations": [SOURCE]})
        if request.url.host == "openrouter.ai":
            self.assertIn("asyncio.timeout", message["messages"][-1]["content"])
            self.assertIn(SOURCE, message["messages"][-1]["content"])
            return httpx.Response(200, json={"choices": [{"message": {"content": "Use asyncio.timeout [1]."}}]})

        self.assertEqual(str(request.url), ENDPOINT)
        self.assertEqual(request.headers["User-Agent"], USER_AGENT)
        self.assertNotIn("Authorization", request.headers)
        method = message["method"]
        if method == "initialize":
            result = {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "search-fixture", "version": "1.0"},
            }
        elif method == "notifications/initialized" or method == "notifications/cancelled":
            return httpx.Response(202)
        elif method == "tools/list":
            result = {"tools": [{"name": "web_search", "inputSchema": {"type": "object"}}]}
        elif method == "tools/call":
            self.call_started.set()
            self.assertEqual(message["params"]["name"], "web_search")
            if self.mode == "wait":
                await asyncio.Event().wait()
            if self.mode == "http_error":
                return httpx.Response(403, text="Search forbidden")
            if self.mode == "network_timeout":
                raise httpx.ReadTimeout("fixture read timed out", request=request)
            if self.mode == "rpc_error":
                return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32603, "message": "Search unavailable"}})
            if self.mode == "quota":
                result = {"isError": True, "content": [{"type": "text", "text": "Free search quota reached"}]}
            elif self.mode == "text":
                result = {"content": [{"type": "text", "text": json.dumps(self.data)}]}
            elif self.mode == "malformed_text":
                result = {"content": [{"type": "text", "text": "not JSON"}]}
            else:
                result = {"structuredContent": self.data, "content": []}
        else:
            self.fail("Unexpected MCP method: " + method)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})

    async def test_default_search_and_synthesis_through_langgraph(self):
        # Script the planner only. Routing, search selection, MCP transport,
        # synthesis prompt construction and final citation consumption are real.
        plan = [{"type": "search", "description": "Python asyncio timeout documentation", "dependencies": [], "status": "pending", "result": None}]
        with tempfile.TemporaryDirectory() as directory:
            mem = memory.MemoryManager(str(Path(directory, "agent.db")))
            with patch.object(orchestrator, "_memory_manager", mem), patch.object(orchestrator.planner, "plan_tasks", return_value=plan):
                response = await orchestrator.run_agent("Find asyncio timeout documentation", "a" * 32)
        self.assertIn("asyncio.timeout", response["response"])
        self.assertEqual(response["citations"], [SOURCE])
        calls = [json.loads(r.content) for r in self.requests if r.url.host == "search.parallel.ai"]
        self.assertIn("initialize", [r["method"] for r in calls])
        self.assertIn("tools/list", [r["method"] for r in calls])
        arguments = next(r["params"]["arguments"] for r in calls if r["method"] == "tools/call")
        self.assertEqual(arguments["session_id"], "a" * 32)
        self.assertEqual(arguments["search_queries"], [plan[0]["description"]])

    async def test_context_and_text_result(self):
        self.mode = "text"
        response = await execute_search("Python asyncio timeout", "Use official Python documentation")
        self.assertTrue(response["success"])
        self.assertEqual(response["citations"], [SOURCE])
        request = next(json.loads(r.content) for r in self.requests if json.loads(r.content).get("method") == "tools/call")
        self.assertIn("Use official Python documentation", request["params"]["arguments"]["objective"])

    async def test_configured_sonar_and_explicit_parallel(self):
        with patch.object(config, "SONAR_API_KEY", "saved-sonar-key"):
            response = await execute_search("Python asyncio timeout", "Official docs")
            self.assertEqual(response["content"], "Sonar answer")
            self.assertEqual(len(self.requests), 1)
            self.assertEqual(json.loads(self.requests[0].content)["messages"][0]["content"], "Official docs")
            with patch.object(config, "SEARCH_PROVIDER", "parallel"):
                self.assertTrue((await execute_search("Python asyncio timeout"))["success"])
        self.assertTrue(any(r.url.host == "search.parallel.ai" for r in self.requests))

    async def test_disabled_invalid_and_missing_sonar_key_never_connect(self):
        for provider in ("none", "unknown", "sonar"):
            with self.subTest(provider=provider), patch.object(config, "SEARCH_PROVIDER", provider):
                response = await execute_search("Python asyncio timeout")
                self.assertFalse(response["success"])
        self.assertEqual(self.requests, [])

    async def test_configured_sonar_failure_does_not_fall_back(self):
        self.mode = "sonar_error"
        with patch.object(config, "SONAR_API_KEY", "saved-sonar-key"):
            response = await execute_search("Python asyncio timeout")
        self.assertFalse(response["success"])
        self.assertIn("401", response["content"])
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.requests[0].url.host, "api.perplexity.ai")

    async def test_errors_do_not_fall_back(self):
        for mode in ("quota", "rpc_error", "http_error", "network_timeout", "malformed_text"):
            with self.subTest(mode=mode):
                self.mode = mode
                with patch.object(config, "SEARCH_PROVIDER", "parallel"), patch.object(config, "SONAR_API_KEY", "saved-sonar-key"):
                    response = await execute_search("Python asyncio timeout")
                self.assertFalse(response["success"])
                self.assertEqual(response["citations"], [])
                self.assertNotIn("ExceptionGroup:", response["content"])
                if mode == "network_timeout":
                    self.assertIn("timed out", response["content"])
        self.assertTrue(all(r.url.host == "search.parallel.ai" for r in self.requests))

    async def test_total_timeout_and_cancellation_close_connections(self):
        self.mode = "wait"
        with patch.object(ParallelSearchClient, "_TIMEOUT_SECONDS", 0.05):
            response = await execute_search("Python asyncio timeout")
        self.assertFalse(response["success"])
        self.call_started.clear()
        task = asyncio.create_task(execute_search("Python asyncio timeout"))
        await asyncio.wait_for(self.call_started.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_urls_and_duplicate_citations(self):
        self.data = {"results": SEARCH_DATA["results"] * 2 + [
            {"url": "javascript:alert(1)", "excerpts": ["invalid"]},
            {"url": "https://user:password@example.com", "excerpts": ["invalid"]},
            {"url": "https://[broken", "excerpts": ["invalid"]},
            {"url": "https://example.com", "excerpts": "not a list"},
        ]}
        response = await execute_search("Python asyncio timeout")
        self.assertEqual(response["citations"], [SOURCE])
        self.assertNotIn("invalid", response["content"])

    async def test_empty_and_malformed_results(self):
        self.data = {"results": []}
        response = await execute_search("Python asyncio timeout")
        self.assertEqual(response, {"content": "No web search results found.", "citations": [], "success": True})
        self.data = {"unexpected": []}
        self.assertFalse((await execute_search("Python asyncio timeout"))["success"])

    async def test_empty_query_and_invalid_session_do_not_connect(self):
        self.assertFalse((await execute_search(" "))["success"])
        self.assertFalse((await execute_search("Python asyncio timeout", session_id="x" * 101))["success"])
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()
