"""
llm/client.py — The ONLY file that imports the Azure/OpenAI SDK.

Reuses the same design as SQL Agent's LLMClient with two additions:

  NEW: generate_cheap()
    Uses cheap_model (gpt-4o-mini) via Chat Completions API.
    Why Chat Completions and not Responses API?
      - Per-page summarization doesn't need tool calling or conversation chaining
      - Chat Completions is simpler, universally supported, and returns text directly
      - Responses API overhead (conversation state, tool dispatch) isn't needed here
      - Using the right API for the right task is good engineering

  NEW: embed()
    Uses the Embeddings API (completely separate from both above).
    Returns a vector of floats representing the semantic meaning of the input text.
    Used in Phase 2 for relevance filtering: embed the query, embed each summary,
    keep only the top-k summaries by cosine similarity.

    Why embeddings for filtering and not just truncation?
      Truncation keeps the first N summaries by order.
      Embedding similarity keeps the N most semantically relevant summaries.
      If round 3 research found the most relevant source, truncation would drop it.
      Similarity keeps it. Quality difference is significant in practice.

THREE API SHAPES:
  client.generate()       → Responses API → smart_model → full Response object
  client.generate_cheap() → Chat Completions API → cheap_model → plain string
  client.embed()          → Embeddings API → embedding_model → list of floats

USAGE:
  from llm.client import LLMClient
  client = LLMClient()

  # Smart: planning, reflection, synthesis
  response = client.generate(input=[{"role": "user", "content": "Plan research for..."}])
  print(response.output_text)

  # Cheap: per-page summarization
  summary = client.generate_cheap("Summarize this page: ...")
  print(summary)  # plain string, no Response object

  # Embed: relevance filtering
  vector = client.embed("What are the latest battery breakthroughs?")
  print(len(vector))  # 1536 for text-embedding-3-small
"""

from openai import AzureOpenAI, AsyncAzureOpenAI, OpenAI, AsyncOpenAI
from openai.types.responses import Response

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from config import settings


class LLMClient:
    """
    Thin wrapper around Azure AI Foundry → three OpenAI API surfaces.

    Same two auth paths as SQL Agent (API key vs managed identity).
    Extended with generate_cheap() and embed() for research workloads.
    """

    def __init__(self) -> None:
        if settings.foundry_api_key:
            # Path A: API key auth
            self._client: OpenAI = AzureOpenAI(
                api_key=settings.foundry_api_key,
                azure_endpoint=settings.foundry_endpoint,
                api_version=settings.api_version,
            )
            self._async_client: AsyncOpenAI = AsyncAzureOpenAI(
                api_key=settings.foundry_api_key,
                azure_endpoint=settings.foundry_endpoint,
                api_version=settings.api_version,
            )
        else:
            # Path B: Managed identity / az login
            project_client = AIProjectClient(
                endpoint=settings.foundry_endpoint,
                credential=DefaultAzureCredential(),
            )
            self._client = project_client.get_openai_client()
            self._async_client = AsyncAzureOpenAI(
                azure_endpoint=settings.foundry_endpoint,
                api_version=settings.api_version,
            )

        self._smart_model = settings.smart_model
        self._cheap_model = settings.cheap_model
        self._embedding_model = settings.embedding_model

    # ── Smart model — Responses API ────────────────────────────────────────────

    def generate(
        self,
        input: str | list[dict],
        *,
        tools: list[dict] | None = None,
        previous_response_id: str | None = None,
        temperature: float | None = None,
    ) -> Response:
        """
        Call smart_model via Responses API.

        Used for: query decomposition, reflection, outline generation,
        section writing, confidence assessment — anything needing high quality
        or tool calling.

        Returns the full Response object (same as SQL Agent).
        Access text via response.output_text, tokens via response.usage.
        """
        kwargs: dict = {"model": self._smart_model, "input": input}

        if temperature is not None:
            kwargs["temperature"] = temperature
        if tools is not None:
            kwargs["tools"] = tools
        if previous_response_id is not None:
            kwargs["previous_response_id"] = previous_response_id

        return self._client.responses.create(**kwargs)

    async def generate_async(
        self,
        input: str | list[dict],
        *,
        tools: list[dict] | None = None,
        previous_response_id: str | None = None,
        temperature: float | None = None,
    ) -> Response:
        """Async version of generate(). Use inside async functions."""
        kwargs: dict = {"model": self._smart_model, "input": input}

        if temperature is not None:
            kwargs["temperature"] = temperature
        if tools is not None:
            kwargs["tools"] = tools
        if previous_response_id is not None:
            kwargs["previous_response_id"] = previous_response_id

        return await self._async_client.responses.create(**kwargs)

    # ── Cheap model — Chat Completions API ─────────────────────────────────────

    def generate_cheap(self, prompt: str) -> str:
        """
        Call cheap_model via Chat Completions API. Returns plain text string.

        Used for: per-page summarization — the high-volume step where quality
        requirements are lower but call count is 50-100x per run.

        Why Chat Completions instead of Responses API?
          Page summarization is stateless (no tool calls, no conversation chain).
          Chat Completions is simpler, cheaper to call, and works with every
          model including gpt-4o-mini. Responses API overhead isn't needed here.

        Returns: the model's text response as a plain string.
        Raises on API error — caller should handle with try/except.
        """
        response = self._client.chat.completions.create(
            model=self._cheap_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=settings.max_summary_tokens,
        )
        return response.choices[0].message.content or ""

    async def generate_cheap_async(self, prompt: str) -> str:
        """Async version of generate_cheap(). Use for parallel page summarization."""
        response = await self._async_client.chat.completions.create(
            model=self._cheap_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=settings.max_summary_tokens,
        )
        return response.choices[0].message.content or ""

    # ── Embedding model — Embeddings API ───────────────────────────────────────

    def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """
        Embed text using text-embedding-3-small.

        Takes a string or list of strings.
        Returns a single vector (list[float]) or list of vectors.

        A vector is a list of 1536 floats (for text-embedding-3-small).
        Each float represents one dimension of the text's semantic meaning.
        Texts with similar meaning will have vectors that point in similar
        directions — measured by cosine similarity.

        This is used in Phase 2 to filter 50-100 summaries down to the
        top-k most relevant before synthesis.

        Usage:
            # Single text
            v = client.embed("battery technology breakthroughs")
            # v is a list of 1536 floats

            # Multiple texts (batched — one API call)
            vectors = client.embed(["text1", "text2", "text3"])
            # vectors is a list of 3 lists of 1536 floats each
        """
        response = self._client.embeddings.create(
            model=self._embedding_model,
            input=text,
        )
        if isinstance(text, str):
            return response.data[0].embedding
        return [item.embedding for item in response.data]

    async def embed_async(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """Async version of embed(). Use for parallel embedding during context filtering."""
        response = await self._async_client.embeddings.create(
            model=self._embedding_model,
            input=text,
        )
        if isinstance(text, str):
            return response.data[0].embedding
        return [item.embedding for item in response.data]
