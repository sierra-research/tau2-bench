"""Retrieval capability MixIns for the banking_knowledge domain.

Each MixIn provides a single retrieval tool via @is_tool methods.
The ToolKitType metaclass collects these tools through MRO when
mixed into a concrete toolkit class.

MixIns define no __init__ — they expect the concrete class to set
the required attributes (e.g., _kb_pipeline, _grep_pipeline, _sandbox)
during its own __init__.
"""

from tau2.environment.toolkit import ToolKitType, ToolType, is_tool


class KBSearchMixin(metaclass=ToolKitType):
    """MixIn that provides the KB_search tool.

    Expects ``self._kb_pipeline`` (a RetrievalPipeline) to be set by the
    concrete class before any tool calls.
    """

    @is_tool(ToolType.READ)
    def KB_search(self, query: str) -> str:
        """Search the knowledge base for relevant documents.

        Args:
            query: The search query to find relevant documents

        Returns:
            Relevant document excerpts matching the query
        """
        # TODO: clean up knowledge retrieval pipelines to return structure results
        retrieval_result = self._kb_pipeline.retrieve(query, return_timing=True)
        results = retrieval_result.results
        timing = retrieval_result.timing

        if not results:
            output = "No relevant documents found."
            output += f"\n\n[Timing: retrieval={timing.retrieval_ms:.0f}ms"
            if timing.postprocessing_ms > 0:
                output += f", reranking={timing.postprocessing_ms:.0f}ms"
            output += f", total={timing.total_ms:.0f}ms]"
            return output

        formatted = []
        for i, (doc_id, score) in enumerate(results, 1):
            title = self._kb_pipeline.get_document_title(doc_id) or "Untitled"
            content = self._kb_pipeline.get_document_content(doc_id) or ""
            formatted.append(
                f"{i}. {title}\n"
                f"   ID: {doc_id}\n"
                f"   Score: {score:.4f}\n"
                f"   Content: {content}\n"
            )

        output = "\n".join(formatted)
        output += f"\n\n[Timing: retrieval={timing.retrieval_ms:.0f}ms"
        if timing.postprocessing_ms > 0:
            output += f", reranking={timing.postprocessing_ms:.0f}ms"
        output += f", total={timing.total_ms:.0f}ms]"
        return output


class GrepMixin(metaclass=ToolKitType):
    """MixIn that provides the grep tool.

    Expects ``self._grep_pipeline`` (a RetrievalPipeline) to be set.
    """

    @is_tool(ToolType.READ)
    def grep(self, pattern: str) -> str:
        """Search for a regex pattern in all knowledge base documents.

        Returns documents ranked by number of matches, with full content.

        Args:
            pattern: The regex pattern to search for (e.g., 'credit.*card', 'fee|charge')

        Returns:
            Matching documents ranked by relevance (match count)
        """
        results = self._grep_pipeline.retrieve(pattern)

        if not results:
            return f"No matches found for pattern: {pattern}"

        formatted = []
        for i, (doc_id, score) in enumerate(results, 1):
            title = self._grep_pipeline.get_document_title(doc_id) or "Untitled"
            content = self._grep_pipeline.get_document_content(doc_id) or ""
            formatted.append(
                f"{i}. {title}\n"
                f"   ID: {doc_id}\n"
                f"   Score: {score:.4f}\n"
                f"   Content: {content}\n"
            )

        return "\n".join(formatted)


class ShellMixin(metaclass=ToolKitType):
    """MixIn that provides the shell tool.

    Expects ``self._sandbox`` (a SandboxManager) to be set.
    """

    @is_tool(ToolType.READ)
    def shell(self, command: str) -> str:
        """Execute a shell command in the knowledge base directory.

        Use standard Unix utilities to explore and search the knowledge base files.
        Common commands: ls, cat, grep, head, tail, find, wc, awk, sed, etc.

        Args:
            command: The shell command to execute (e.g., "ls -la", "grep -r 'credit card' .", "cat INDEX.txt")

        Returns:
            The command output (stdout) or error message
        """
        if self._sandbox is None:
            return "Error: Sandbox not initialized"

        ret_code, stdout, stderr = self._sandbox.run_command(command)

        if ret_code != 0:
            # grep returns 1 when no matches (not an error)
            if ret_code == 1 and "grep" in command and not stderr:
                return "No matches found."
            if stderr:
                return f"Error (exit code {ret_code}): {stderr}"
            return f"Command failed with exit code {ret_code}"

        return stdout if stdout else "(no output)"


class RewriteContextMixin(metaclass=ToolKitType):
    """MixIn that provides the rewrite_context tool for summarization."""

    @is_tool(ToolType.WRITE, mutates_state=False)
    def rewrite_context(self, new_context: str) -> str:
        """Replace your working context with a new summary or condensed version.

        Use this after searching to summarize findings, extract key points,
        or condense information for easier reference later in the conversation.

        Args:
            new_context: The summarized or rewritten context to store

        Returns:
            The context you provided, for reference
        """
        return f"Context updated:\n\n{new_context}"
