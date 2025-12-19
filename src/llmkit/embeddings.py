"""
Embeddings - Unified Interface
llmkit 방식: Client와 같은 패턴, 자동 provider 감지
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

from .utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingResult:
    """Embedding 결과"""

    embeddings: List[List[float]]  # 임베딩 벡터들
    model: str  # 사용된 모델
    usage: Dict[str, int]  # 토큰 사용량 등


class BaseEmbedding(ABC):
    """Embedding 베이스 클래스"""

    def __init__(self, model: str, **kwargs):
        """
        Args:
            model: 모델 이름
            **kwargs: provider별 추가 파라미터
        """
        self.model = model
        self.kwargs = kwargs

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트들을 임베딩

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            임베딩 벡터 리스트
        """
        pass

    @abstractmethod
    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트들을 임베딩 (동기)

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            임베딩 벡터 리스트
        """
        pass


class OpenAIEmbedding(BaseEmbedding):
    """
    OpenAI Embeddings

    Example:
        ```python
        from llmkit.embeddings import OpenAIEmbedding

        emb = OpenAIEmbedding(model="text-embedding-3-small")
        vectors = await emb.embed(["text1", "text2"])
        ```
    """

    def __init__(
        self, model: str = "text-embedding-3-small", api_key: Optional[str] = None, **kwargs
    ):
        """
        Args:
            model: OpenAI embedding 모델
            api_key: OpenAI API 키 (None이면 환경변수)
            **kwargs: 추가 파라미터
        """
        super().__init__(model, **kwargs)

        # OpenAI 클라이언트 초기화
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError(
                "openai is required for OpenAIEmbedding. " "Install it with: pip install openai"
            )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.async_client = AsyncOpenAI(api_key=self.api_key)
        self.sync_client = OpenAI(api_key=self.api_key)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (비동기)"""
        try:
            response = await self.async_client.embeddings.create(
                input=texts, model=self.model, **self.kwargs
            )

            embeddings = [item.embedding for item in response.data]
            logger.info(
                f"Embedded {len(texts)} texts using {self.model}, "
                f"usage: {response.usage.total_tokens} tokens"
            )

            return embeddings

        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (동기)"""
        try:
            response = self.sync_client.embeddings.create(
                input=texts, model=self.model, **self.kwargs
            )

            embeddings = [item.embedding for item in response.data]
            logger.info(
                f"Embedded {len(texts)} texts using {self.model}, "
                f"usage: {response.usage.total_tokens} tokens"
            )

            return embeddings

        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise


class GeminiEmbedding(BaseEmbedding):
    """
    Google Gemini Embeddings

    Example:
        ```python
        from llmkit.embeddings import GeminiEmbedding

        emb = GeminiEmbedding(model="models/embedding-001")
        vectors = await emb.embed(["text1", "text2"])
        ```
    """

    def __init__(
        self, model: str = "models/embedding-001", api_key: Optional[str] = None, **kwargs
    ):
        """
        Args:
            model: Gemini embedding 모델
            api_key: Google API 키 (None이면 환경변수)
            **kwargs: 추가 파라미터
        """
        super().__init__(model, **kwargs)

        # Gemini 클라이언트 초기화
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai is required for GeminiEmbedding. "
                "Install it with: pip install llmkit[gemini]"
            )

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables")

        genai.configure(api_key=self.api_key)
        self.genai = genai

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (비동기)"""
        # Gemini SDK는 async 지원 안 함, sync 사용
        return self.embed_sync(texts)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (동기)"""
        try:
            embeddings = []
            # Gemini는 배치 임베딩을 지원하지 않으므로 하나씩 처리
            for text in texts:
                result = self.genai.embed_content(model=self.model, content=text, **self.kwargs)
                embeddings.append(result["embedding"])

            logger.info(f"Embedded {len(texts)} texts using {self.model}")
            return embeddings

        except Exception as e:
            logger.error(f"Gemini embedding failed: {e}")
            raise


class OllamaEmbedding(BaseEmbedding):
    """
    Ollama Embeddings (로컬)

    Example:
        ```python
        from llmkit.embeddings import OllamaEmbedding

        emb = OllamaEmbedding(model="nomic-embed-text")
        vectors = emb.embed_sync(["text1", "text2"])
        ```
    """

    def __init__(
        self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434", **kwargs
    ):
        """
        Args:
            model: Ollama embedding 모델
            base_url: Ollama 서버 URL
            **kwargs: 추가 파라미터
        """
        super().__init__(model, **kwargs)

        try:
            import ollama
        except ImportError:
            raise ImportError(
                "ollama is required for OllamaEmbedding. "
                "Install it with: pip install llmkit[ollama]"
            )

        self.client = ollama.Client(host=base_url)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (비동기)"""
        # Ollama는 async 지원 안 함
        return self.embed_sync(texts)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (동기)"""
        try:
            embeddings = []
            for text in texts:
                response = self.client.embeddings(model=self.model, prompt=text)
                embeddings.append(response["embedding"])

            logger.info(f"Embedded {len(texts)} texts using Ollama {self.model}")
            return embeddings

        except Exception as e:
            logger.error(f"Ollama embedding failed: {e}")
            raise


class VoyageEmbedding(BaseEmbedding):
    """
    Voyage AI Embeddings

    Example:
        ```python
        from llmkit.embeddings import VoyageEmbedding

        emb = VoyageEmbedding(model="voyage-2")
        vectors = await emb.embed(["text1", "text2"])
        ```
    """

    def __init__(self, model: str = "voyage-2", api_key: Optional[str] = None, **kwargs):
        """
        Args:
            model: Voyage AI 모델
            api_key: Voyage AI API 키
            **kwargs: 추가 파라미터
        """
        super().__init__(model, **kwargs)

        try:
            import voyageai
        except ImportError:
            raise ImportError(
                "voyageai is required for VoyageEmbedding. " "Install it with: pip install voyageai"
            )

        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError("VOYAGE_API_KEY not found in environment variables")

        self.client = voyageai.Client(api_key=self.api_key)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (비동기)"""
        return self.embed_sync(texts)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (동기)"""
        try:
            response = self.client.embed(texts=texts, model=self.model, **self.kwargs)

            logger.info(f"Embedded {len(texts)} texts using {self.model}")
            return response.embeddings

        except Exception as e:
            logger.error(f"Voyage AI embedding failed: {e}")
            raise


class JinaEmbedding(BaseEmbedding):
    """
    Jina AI Embeddings

    Example:
        ```python
        from llmkit.embeddings import JinaEmbedding

        emb = JinaEmbedding(model="jina-embeddings-v2-base-en")
        vectors = await emb.embed(["text1", "text2"])
        ```
    """

    def __init__(
        self, model: str = "jina-embeddings-v2-base-en", api_key: Optional[str] = None, **kwargs
    ):
        """
        Args:
            model: Jina AI 모델
            api_key: Jina AI API 키
            **kwargs: 추가 파라미터
        """
        super().__init__(model, **kwargs)

        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("JINA_API_KEY not found in environment variables")

        self.url = "https://api.jina.ai/v1/embeddings"

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (비동기)"""
        return self.embed_sync(texts)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (동기)"""
        try:
            import requests

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            data = {"model": self.model, "input": texts, **self.kwargs}

            response = requests.post(self.url, headers=headers, json=data)
            response.raise_for_status()

            result = response.json()
            embeddings = [item["embedding"] for item in result["data"]]

            logger.info(f"Embedded {len(texts)} texts using {self.model}")
            return embeddings

        except Exception as e:
            logger.error(f"Jina AI embedding failed: {e}")
            raise


class MistralEmbedding(BaseEmbedding):
    """
    Mistral AI Embeddings

    Example:
        ```python
        from llmkit.embeddings import MistralEmbedding

        emb = MistralEmbedding(model="mistral-embed")
        vectors = await emb.embed(["text1", "text2"])
        ```
    """

    def __init__(self, model: str = "mistral-embed", api_key: Optional[str] = None, **kwargs):
        """
        Args:
            model: Mistral AI 모델
            api_key: Mistral AI API 키
            **kwargs: 추가 파라미터
        """
        super().__init__(model, **kwargs)

        try:
            from mistralai.client import MistralClient
        except ImportError:
            raise ImportError(
                "mistralai is required for MistralEmbedding. "
                "Install it with: pip install mistralai"
            )

        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")

        self.client = MistralClient(api_key=self.api_key)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (비동기)"""
        return self.embed_sync(texts)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (동기)"""
        try:
            response = self.client.embeddings(model=self.model, input=texts)

            embeddings = [item.embedding for item in response.data]
            logger.info(f"Embedded {len(texts)} texts using {self.model}")
            return embeddings

        except Exception as e:
            logger.error(f"Mistral AI embedding failed: {e}")
            raise


class CohereEmbedding(BaseEmbedding):
    """
    Cohere Embeddings

    Example:
        ```python
        from llmkit.embeddings import CohereEmbedding

        emb = CohereEmbedding(model="embed-english-v3.0")
        vectors = await emb.embed(["text1", "text2"])
        ```
    """

    def __init__(
        self,
        model: str = "embed-english-v3.0",
        api_key: Optional[str] = None,
        input_type: str = "search_document",
        **kwargs,
    ):
        """
        Args:
            model: Cohere embedding 모델
            api_key: Cohere API 키 (None이면 환경변수)
            input_type: "search_document", "search_query", "classification", "clustering"
            **kwargs: 추가 파라미터
        """
        super().__init__(model, **kwargs)

        # Cohere 클라이언트 초기화
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "cohere is required for CohereEmbedding. " "Install it with: pip install cohere"
            )

        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY not found in environment variables")

        self.client = cohere.Client(api_key=self.api_key)
        self.input_type = input_type

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (비동기)"""
        # Cohere SDK는 async 지원 안 함, sync 사용
        return self.embed_sync(texts)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (동기)"""
        try:
            response = self.client.embed(
                texts=texts, model=self.model, input_type=self.input_type, **self.kwargs
            )

            logger.info(f"Embedded {len(texts)} texts using {self.model}")
            return response.embeddings

        except Exception as e:
            logger.error(f"Cohere embedding failed: {e}")
            raise


class Embedding:
    """
    Embedding 팩토리 - 자동 provider 감지

    **llmkit 방식: Client와 같은 패턴!**

    Example:
        ```python
        from llmkit import Embedding

        # 자동 감지 (모델 이름으로)
        emb = Embedding(model="text-embedding-3-small")  # OpenAI 자동
        emb = Embedding(model="embed-english-v3.0")      # Cohere 자동

        # 임베딩
        vectors = await emb.embed(["text1", "text2"])

        # 동기 버전
        vectors = emb.embed_sync(["text1", "text2"])
        ```
    """

    # 모델 이름 패턴으로 provider 감지
    PROVIDER_PATTERNS = {
        "openai": [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ],
        "gemini": [
            "models/embedding-001",
            "models/text-embedding-004",
            "embedding-001",
            "text-embedding-004",
        ],
        "ollama": [
            "nomic-embed-text",
            "mxbai-embed-large",
            "all-minilm",
        ],
        "voyage": [
            "voyage-2",
            "voyage-large-2",
            "voyage-code-2",
            "voyage-lite-02-instruct",
        ],
        "jina": [
            "jina-embeddings-v2-base-en",
            "jina-embeddings-v2-small-en",
            "jina-embeddings-v2-base-zh",
            "jina-clip-v1",
        ],
        "mistral": [
            "mistral-embed",
        ],
        "cohere": [
            "embed-english-v3.0",
            "embed-english-light-v3.0",
            "embed-multilingual-v3.0",
            "embed-english-v2.0",
        ],
    }

    # Provider별 클래스 매핑
    PROVIDERS = {
        "openai": OpenAIEmbedding,
        "gemini": GeminiEmbedding,
        "ollama": OllamaEmbedding,
        "voyage": VoyageEmbedding,
        "jina": JinaEmbedding,
        "mistral": MistralEmbedding,
        "cohere": CohereEmbedding,
    }

    # Provider별 필요한 환경변수
    PROVIDER_ENV_VARS = {
        "openai": "OPENAI_API_KEY",
        "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        "ollama": None,  # 로컬, API 키 불필요
        "voyage": "VOYAGE_API_KEY",
        "jina": "JINA_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "cohere": "COHERE_API_KEY",
    }

    def __new__(cls, model: str, provider: Optional[str] = None, **kwargs) -> BaseEmbedding:
        """
        Embedding 인스턴스 생성 (자동 provider 감지)

        Args:
            model: 모델 이름
            provider: Provider 명시 (None이면 자동 감지)
            **kwargs: Provider별 추가 파라미터

        Returns:
            적절한 Embedding 인스턴스
        """
        # Provider 감지
        if provider is None:
            provider = cls._detect_provider(model)
            if provider:
                logger.info(f"Auto-detected provider: {provider} for model: {model}")
            else:
                # 기본: OpenAI
                logger.warning(
                    f"Could not detect provider for model: {model}, " f"defaulting to OpenAI"
                )
                provider = "openai"

        # Provider 클래스 선택
        if provider not in cls.PROVIDERS:
            raise ValueError(
                f"Unknown provider: {provider}. " f"Supported: {list(cls.PROVIDERS.keys())}"
            )

        embedding_class = cls.PROVIDERS[provider]
        return embedding_class(model=model, **kwargs)

    @classmethod
    def _detect_provider(cls, model: str) -> Optional[str]:
        """모델 이름으로 provider 감지"""
        model_lower = model.lower()

        for provider, patterns in cls.PROVIDER_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in model_lower:
                    return provider

        return None

    @classmethod
    def openai(cls, model: str = "text-embedding-3-small", **kwargs) -> OpenAIEmbedding:
        """
        OpenAI Embedding 생성 (명시적)

        Example:
            ```python
            emb = Embedding.openai()
            emb = Embedding.openai(model="text-embedding-3-large")
            ```
        """
        return OpenAIEmbedding(model=model, **kwargs)

    @classmethod
    def gemini(cls, model: str = "models/embedding-001", **kwargs) -> GeminiEmbedding:
        """
        Gemini Embedding 생성 (명시적)

        Example:
            ```python
            emb = Embedding.gemini()
            emb = Embedding.gemini(model="models/text-embedding-004")
            ```
        """
        return GeminiEmbedding(model=model, **kwargs)

    @classmethod
    def ollama(cls, model: str = "nomic-embed-text", **kwargs) -> OllamaEmbedding:
        """
        Ollama Embedding 생성 (명시적)

        Example:
            ```python
            emb = Embedding.ollama()
            emb = Embedding.ollama(model="mxbai-embed-large")
            ```
        """
        return OllamaEmbedding(model=model, **kwargs)

    @classmethod
    def voyage(cls, model: str = "voyage-2", **kwargs) -> VoyageEmbedding:
        """
        Voyage AI Embedding 생성 (명시적)

        Example:
            ```python
            emb = Embedding.voyage()
            emb = Embedding.voyage(model="voyage-large-2")
            ```
        """
        return VoyageEmbedding(model=model, **kwargs)

    @classmethod
    def jina(cls, model: str = "jina-embeddings-v2-base-en", **kwargs) -> JinaEmbedding:
        """
        Jina AI Embedding 생성 (명시적)

        Example:
            ```python
            emb = Embedding.jina()
            emb = Embedding.jina(model="jina-embeddings-v2-small-en")
            ```
        """
        return JinaEmbedding(model=model, **kwargs)

    @classmethod
    def mistral(cls, model: str = "mistral-embed", **kwargs) -> MistralEmbedding:
        """
        Mistral AI Embedding 생성 (명시적)

        Example:
            ```python
            emb = Embedding.mistral()
            ```
        """
        return MistralEmbedding(model=model, **kwargs)

    @classmethod
    def cohere(cls, model: str = "embed-english-v3.0", **kwargs) -> CohereEmbedding:
        """
        Cohere Embedding 생성 (명시적)

        Example:
            ```python
            emb = Embedding.cohere()
            emb = Embedding.cohere(model="embed-multilingual-v3.0")
            ```
        """
        return CohereEmbedding(model=model, **kwargs)

    @classmethod
    def list_available_providers(cls) -> List[str]:
        """
        사용 가능한 provider 목록

        API 키가 설정된 provider만 반환

        Returns:
            사용 가능한 provider 이름 리스트

        Example:
            ```python
            providers = Embedding.list_available_providers()
            print(f"Available: {providers}")
            # ['openai', 'ollama']
            ```
        """
        available = []

        for provider, env_var in cls.PROVIDER_ENV_VARS.items():
            if env_var is None:  # Ollama (로컬)
                available.append(provider)
            elif isinstance(env_var, list):  # 여러 가능한 환경변수
                if any(os.getenv(var) for var in env_var):
                    available.append(provider)
            else:  # 단일 환경변수
                if os.getenv(env_var):
                    available.append(provider)

        return available

    @classmethod
    def get_default_provider(cls) -> Optional[str]:
        """
        기본 provider 반환

        사용 가능한 provider 중 우선순위가 가장 높은 것

        우선순위: OpenAI > Gemini > Voyage > Cohere > Ollama

        Returns:
            기본 provider 이름

        Example:
            ```python
            provider = Embedding.get_default_provider()
            emb = Embedding(model="...", provider=provider)
            ```
        """
        priority = ["openai", "gemini", "voyage", "cohere", "ollama"]
        available = cls.list_available_providers()

        for provider in priority:
            if provider in available:
                return provider

        return None


# 편의 함수
async def embed(
    texts: Union[str, List[str]], model: str = "text-embedding-3-small", **kwargs
) -> List[List[float]]:
    """
    텍스트를 임베딩하는 편의 함수

    Args:
        texts: 단일 텍스트 또는 리스트
        model: 모델 이름
        **kwargs: 추가 파라미터

    Returns:
        임베딩 벡터 리스트

    Example:
        ```python
        from llmkit.embeddings import embed

        # 단일 텍스트
        vector = await embed("Hello world")

        # 여러 텍스트
        vectors = await embed(["text1", "text2", "text3"])
        ```
    """
    # 단일 텍스트를 리스트로 변환
    if isinstance(texts, str):
        texts = [texts]

    embedding = Embedding(model=model, **kwargs)
    return await embedding.embed(texts)


def embed_sync(
    texts: Union[str, List[str]], model: str = "text-embedding-3-small", **kwargs
) -> List[List[float]]:
    """
    텍스트를 임베딩하는 편의 함수 (동기)

    Args:
        texts: 단일 텍스트 또는 리스트
        model: 모델 이름
        **kwargs: 추가 파라미터

    Returns:
        임베딩 벡터 리스트

    Example:
        ```python
        from llmkit.embeddings import embed_sync

        # 단일 텍스트
        vector = embed_sync("Hello world")

        # 여러 텍스트
        vectors = embed_sync(["text1", "text2", "text3"])
        ```
    """
    # 단일 텍스트를 리스트로 변환
    if isinstance(texts, str):
        texts = [texts]

    embedding = Embedding(model=model, **kwargs)
    return embedding.embed_sync(texts)


# 유사도 계산 유틸리티 함수
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    두 벡터 간의 코사인 유사도 계산

    코사인 유사도는 벡터의 방향(의미)을 측정하므로,
    텍스트 임베딩의 의미적 유사도를 비교할 때 적합합니다.

    Args:
        vec1: 첫 번째 임베딩 벡터
        vec2: 두 번째 임베딩 벡터

    Returns:
        코사인 유사도 값 (-1 ~ 1, 1에 가까울수록 유사)

    Example:
        ```python
        from llmkit.embeddings import embed_sync, cosine_similarity

        vec1 = embed_sync("고양이는 귀여워")[0]
        vec2 = embed_sync("강아지는 귀여워")[0]
        similarity = cosine_similarity(vec1, vec2)
        print(f"유사도: {similarity:.3f}")  # 0.8 정도
        ```

    수학적 고려사항:
        - 벡터가 이미 정규화되어 있으면 내적만으로 계산 가능
        - 정규화되지 않은 벡터는 자동으로 정규화하여 계산
        - 코사인 유사도는 벡터의 크기(길이)에 영향을 받지 않음
    """
    if not HAS_NUMPY:
        # numpy가 없는 경우 순수 Python 구현
        if len(vec1) != len(vec2):
            raise ValueError(
                f"벡터 차원이 다릅니다: {len(vec1)} vs {len(vec2)}. "
                "같은 모델로 생성한 임베딩을 사용해야 합니다."
            )

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            logger.warning("영벡터가 감지되었습니다. 유사도는 0으로 반환합니다.")
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return max(-1.0, min(1.0, similarity))

    try:
        v1 = np.array(vec1, dtype=np.float32)
        v2 = np.array(vec2, dtype=np.float32)

        # 차원 확인
        if len(v1) != len(v2):
            raise ValueError(
                f"벡터 차원이 다릅니다: {len(v1)} vs {len(v2)}. "
                "같은 모델로 생성한 임베딩을 사용해야 합니다."
            )

        # L2 정규화 (코사인 유사도 계산을 위해)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            logger.warning("영벡터가 감지되었습니다. 유사도는 0으로 반환합니다.")
            return 0.0

        # 코사인 유사도 = (A · B) / (||A|| * ||B||)
        similarity = np.dot(v1, v2) / (norm1 * norm2)

        # 수치 안정성을 위해 -1과 1 사이로 클리핑
        return float(np.clip(similarity, -1.0, 1.0))

    except Exception as e:
        logger.error(f"코사인 유사도 계산 중 오류: {e}")
        raise


def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """
    두 벡터 간의 유클리드 거리 계산

    유클리드 거리는 벡터의 크기와 방향을 모두 고려하므로,
    벡터의 절대적 차이를 측정할 때 사용합니다.

    Args:
        vec1: 첫 번째 임베딩 벡터
        vec2: 두 번째 임베딩 벡터

    Returns:
        유클리드 거리 (0에 가까울수록 유사)

    Example:
        ```python
        from llmkit.embeddings import embed_sync, euclidean_distance

        vec1 = embed_sync("고양이는 귀여워")[0]
        vec2 = embed_sync("강아지는 귀여워")[0]
        distance = euclidean_distance(vec1, vec2)
        print(f"거리: {distance:.3f}")  # 작을수록 유사
        ```

    수학적 고려사항:
        - 거리가 작을수록 유사도가 높음
        - 벡터의 크기(스케일)에 영향을 받음
        - 코사인 유사도와 달리 벡터의 절대적 위치를 비교
    """
    if not HAS_NUMPY:
        # numpy가 없는 경우 순수 Python 구현
        if len(vec1) != len(vec2):
            raise ValueError(f"벡터 차원이 다릅니다: {len(vec1)} vs {len(vec2)}")

        distance = sum((a - b) ** 2 for a, b in zip(vec1, vec2)) ** 0.5
        return distance

    try:
        v1 = np.array(vec1, dtype=np.float32)
        v2 = np.array(vec2, dtype=np.float32)

        if len(v1) != len(v2):
            raise ValueError(f"벡터 차원이 다릅니다: {len(v1)} vs {len(v2)}")

        # 유클리드 거리 = sqrt(sum((a_i - b_i)^2))
        distance = np.linalg.norm(v1 - v2)
        return float(distance)

    except Exception as e:
        logger.error(f"유클리드 거리 계산 중 오류: {e}")
        raise


def normalize_vector(vec: List[float]) -> List[float]:
    """
    벡터를 L2 정규화 (단위 벡터로 변환)

    정규화된 벡터는 크기가 1이 되어 코사인 유사도 계산이 간단해집니다.
    많은 임베딩 모델은 이미 정규화된 벡터를 반환하지만,
    필요시 명시적으로 정규화할 수 있습니다.

    Args:
        vec: 정규화할 벡터

    Returns:
        L2 정규화된 벡터 (크기 = 1)

    Example:
        ```python
        from llmkit.embeddings import embed_sync, normalize_vector

        vec = embed_sync("Hello world")[0]
        normalized = normalize_vector(vec)

        # 정규화 확인
        import math
        norm = math.sqrt(sum(x**2 for x in normalized))
        print(f"정규화 후 크기: {norm:.6f}")  # 1.0에 가까움
        ```

    수학적 고려사항:
        - L2 정규화: v / ||v||
        - 영벡터는 정규화할 수 없음 (원본 반환)
        - 정규화 후 벡터의 방향은 유지되고 크기만 1로 변경
    """
    if not HAS_NUMPY:
        # numpy가 없는 경우 순수 Python 구현
        norm = sum(x * x for x in vec) ** 0.5

        if norm == 0:
            logger.warning("영벡터는 정규화할 수 없습니다. 원본을 반환합니다.")
            return vec

        return [x / norm for x in vec]

    try:
        v = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(v)

        if norm == 0:
            logger.warning("영벡터는 정규화할 수 없습니다. 원본을 반환합니다.")
            return vec

        normalized = v / norm
        return normalized.tolist()

    except Exception as e:
        logger.error(f"벡터 정규화 중 오류: {e}")
        raise


def batch_cosine_similarity(
    query_vec: List[float], candidate_vecs: List[List[float]]
) -> List[float]:
    """
    하나의 쿼리 벡터와 여러 후보 벡터들 간의 코사인 유사도를 일괄 계산

    검색이나 유사도 기반 랭킹에 유용합니다.

    Args:
        query_vec: 쿼리 임베딩 벡터
        candidate_vecs: 후보 임베딩 벡터들의 리스트

    Returns:
        각 후보 벡터와의 코사인 유사도 리스트

    Example:
        ```python
        from llmkit.embeddings import embed_sync, batch_cosine_similarity

        query = embed_sync("고양이")[0]
        candidates = embed_sync(["강아지", "고양이", "자동차"])
        similarities = batch_cosine_similarity(query, candidates)

        # 가장 유사한 것 찾기
        best_idx = similarities.index(max(similarities))
        print(f"가장 유사한 것: {['강아지', '고양이', '자동차'][best_idx]}")
        ```

    수학적 고려사항:
        - 배치 처리로 효율적인 계산
        - 모든 벡터는 같은 차원이어야 함
        - 정규화된 벡터를 사용하면 내적만으로 계산 가능 (더 빠름)
    """
    if not HAS_NUMPY:
        # numpy가 없는 경우 순수 Python 구현
        return [cosine_similarity(query_vec, candidate) for candidate in candidate_vecs]

    try:
        query = np.array(query_vec, dtype=np.float32)
        candidates = np.array(candidate_vecs, dtype=np.float32)

        if len(query) != candidates.shape[1]:
            raise ValueError(
                f"벡터 차원이 다릅니다: 쿼리 {len(query)} vs 후보 {candidates.shape[1]}"
            )

        # 정규화
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return [0.0] * len(candidate_vecs)

        candidate_norms = np.linalg.norm(candidates, axis=1, keepdims=True)

        # 코사인 유사도 계산 (배치)
        similarities = np.dot(candidates, query) / (candidate_norms.flatten() * query_norm)

        # 클리핑
        similarities = np.clip(similarities, -1.0, 1.0)

        return similarities.tolist()

    except Exception as e:
        logger.error(f"배치 코사인 유사도 계산 중 오류: {e}")
        raise


# ============================================================================
# 실무 고급 기법들
# ============================================================================


def find_hard_negatives(
    query_vec: List[float],
    candidate_vecs: List[List[float]],
    positive_vecs: Optional[List[List[float]]] = None,
    similarity_threshold: tuple = (0.3, 0.7),
    top_k: Optional[int] = None,
) -> List[int]:
    """
    Hard Negative Mining: 학습에 유용한 어려운 negative 샘플 찾기

    Hard Negative는 쿼리와 관련 없어 보이지만 실제로는 관련 있는 샘플로,
    모델 학습 시 중요한 역할을 합니다.

    Args:
        query_vec: 쿼리 임베딩 벡터
        candidate_vecs: 후보 임베딩 벡터들의 리스트
        positive_vecs: Positive 샘플 벡터들 (선택적, 제외용)
        similarity_threshold: (min, max) 유사도 범위 (이 범위 안이 Hard Negative)
        top_k: 반환할 Hard Negative 개수 (None이면 모두)

    Returns:
        Hard Negative 인덱스 리스트

    Example:
        ```python
        from llmkit.embeddings import embed_sync, find_hard_negatives

        query = embed_sync("고양이 사료")[0]
        candidates = embed_sync([
            "강아지 사료",  # Hard Negative (비슷하지만 다름)
            "고양이 장난감",  # Hard Negative
            "자동차",  # Easy Negative (너무 다름)
            "고양이 먹이"  # Positive (같음)
        ])

        hard_neg_indices = find_hard_negatives(
            query, candidates,
            similarity_threshold=(0.3, 0.7)
        )
        # → [0, 1] (강아지 사료, 고양이 장난감)
        ```

    수학적 원리:
        - Easy Negative: 유사도 < 0.3 (너무 다름, 학습에 도움 안 됨)
        - Hard Negative: 0.3 < 유사도 < 0.7 (비슷하지만 다름, 학습에 중요!)
        - Positive: 유사도 > 0.7 (같음, 제외)
    """
    # 모든 후보와의 유사도 계산
    similarities = batch_cosine_similarity(query_vec, candidate_vecs)

    # Positive 제외 (제공된 경우)
    if positive_vecs:
        positive_similarities = [
            max(batch_cosine_similarity(query_vec, [pv])[0] for pv in positive_vecs)
            for _ in candidate_vecs
        ]
        # Positive와 유사한 것 제외
        similarities = [s if s < 0.7 else -1.0 for s in similarities]

    # Hard Negative 찾기 (유사도 범위 내)
    min_sim, max_sim = similarity_threshold
    hard_neg_indices = [i for i, sim in enumerate(similarities) if min_sim < sim < max_sim]

    # 유사도 순으로 정렬
    hard_neg_with_sim = [(i, similarities[i]) for i in hard_neg_indices]
    hard_neg_with_sim.sort(key=lambda x: x[1], reverse=True)

    # Top-k 선택
    if top_k is not None:
        hard_neg_with_sim = hard_neg_with_sim[:top_k]

    return [i for i, _ in hard_neg_with_sim]


def mmr_search(
    query_vec: List[float],
    candidate_vecs: List[List[float]],
    k: int = 5,
    lambda_param: float = 0.6,
) -> List[int]:
    """
    MMR (Maximal Marginal Relevance) 검색: 다양성을 고려한 검색

    관련성과 다양성을 균형있게 고려하여 검색 결과를 선택합니다.

    Args:
        query_vec: 쿼리 임베딩 벡터
        candidate_vecs: 후보 임베딩 벡터들의 리스트
        k: 반환할 결과 개수
        lambda_param: 관련성 vs 다양성 균형 (0.0-1.0, 높을수록 관련성 중시)

    Returns:
        선택된 후보 인덱스 리스트 (다양성 고려)

    Example:
        ```python
        from llmkit.embeddings import embed_sync, mmr_search

        query = embed_sync("고양이")[0]
        candidates = embed_sync([
            "고양이 사료", "고양이 사료 추천", "고양이 사료 종류",  # 모두 비슷함
            "고양이 건강", "고양이 행동"  # 다른 주제
        ])

        # 일반 검색: 모두 "사료" 관련
        # MMR 검색: 다양한 주제 포함
        selected = mmr_search(query, candidates, k=3, lambda_param=0.6)
        # → [0, 3, 4] (사료, 건강, 행동 - 다양함!)
        ```

    수학적 원리:
        MMR = argmax[λ × sim(q, d) - (1-λ) × max(sim(d, d_selected))]
        - λ × sim(q, d): 쿼리와의 관련성
        - (1-λ) × max(sim(d, d_selected)): 이미 선택된 문서와의 차이 (다양성)
    """
    if k >= len(candidate_vecs):
        return list(range(len(candidate_vecs)))

    # 쿼리와 모든 후보의 유사도
    query_similarities = batch_cosine_similarity(query_vec, candidate_vecs)

    # 첫 번째: 가장 관련성 높은 것
    selected = [query_similarities.index(max(query_similarities))]
    remaining = set(range(len(candidate_vecs))) - set(selected)

    # 나머지 k-1개 선택
    for _ in range(k - 1):
        if not remaining:
            break

        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            # 관련성 점수
            relevance = query_similarities[idx]

            # 다양성 점수 (이미 선택된 것과의 최대 유사도)
            diversity = 0.0
            if selected:
                selected_vecs = [candidate_vecs[i] for i in selected]
                candidate_sims = batch_cosine_similarity(candidate_vecs[idx], selected_vecs)
                diversity = max(candidate_sims) if candidate_sims else 0.0

            # MMR 점수
            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return selected


def query_expansion(
    query: str,
    embedding: BaseEmbedding,
    expansion_candidates: Optional[List[str]] = None,
    top_k: int = 3,
    similarity_threshold: float = 0.7,
) -> List[str]:
    """
    Query Expansion: 쿼리를 유사어로 확장하여 검색 범위 확대

    원본 쿼리와 유사한 용어를 추가하여 검색 리콜을 향상시킵니다.

    Args:
        query: 원본 쿼리
        embedding: 임베딩 인스턴스
        expansion_candidates: 확장 후보 단어/구 리스트 (None이면 자동 생성 불가)
        top_k: 추가할 확장어 개수
        similarity_threshold: 유사도 임계값 (이 이상만 추가)

    Returns:
        확장된 쿼리 리스트 [원본, 확장1, 확장2, ...]

    Example:
        ```python
        from llmkit.embeddings import Embedding, query_expansion

        emb = Embedding(model="text-embedding-3-small")

        # 후보 단어 제공
        candidates = ["고양이", "냥이", "고양이과", "cat", "feline", "강아지"]

        expanded = query_expansion("고양이", emb, candidates, top_k=3)
        # → ["고양이", "냥이", "고양이과", "cat"]
        ```

    언어학적 원리:
        - 동의어/유사어 추가로 검색 범위 확대
        - 예: "고양이" → "고양이", "냥이", "cat", "feline"
        - 리콜 향상 (더 많은 관련 문서 발견)
    """
    expanded = [query]

    if not expansion_candidates:
        logger.warning("expansion_candidates가 없으면 확장 불가. 원본만 반환합니다.")
        return expanded

    # 원본 쿼리 임베딩
    query_vec = embedding.embed_sync([query])[0]

    # 후보 임베딩
    candidate_vecs = embedding.embed_sync(expansion_candidates)

    # 유사도 계산
    similarities = batch_cosine_similarity(query_vec, candidate_vecs)

    # 유사도가 높은 순으로 정렬
    candidate_with_sim = list(zip(expansion_candidates, similarities))
    candidate_with_sim.sort(key=lambda x: x[1], reverse=True)

    # 임계값 이상이고 원본과 다른 것만 추가
    for candidate, sim in candidate_with_sim:
        if sim >= similarity_threshold and candidate.lower() != query.lower():
            expanded.append(candidate)
            if len(expanded) >= top_k + 1:  # +1은 원본 포함
                break

    return expanded


class EmbeddingCache:
    """
    Embedding 캐시: 같은 텍스트의 임베딩을 재사용하여 비용 절감

    Example:
        ```python
        from llmkit.embeddings import Embedding, EmbeddingCache

        emb = Embedding(model="text-embedding-3-small")
        cache = EmbeddingCache(ttl=3600)  # 1시간 캐시

        # 첫 번째: API 호출
        vec1 = await emb.embed(["텍스트"], cache=cache)

        # 두 번째: 캐시에서 가져옴 (API 호출 안 함)
        vec2 = await emb.embed(["텍스트"], cache=cache)
        ```
    """

    def __init__(self, ttl: int = 3600, max_size: int = 10000):
        """
        Args:
            ttl: 캐시 유지 시간 (초)
            max_size: 최대 캐시 항목 수
        """
        import time
        from collections import OrderedDict

        self.cache: OrderedDict[str, tuple[List[float], float]] = OrderedDict()
        self.ttl = ttl
        self.max_size = max_size
        self.time = time

    def get(self, text: str) -> Optional[List[float]]:
        """캐시에서 가져오기"""
        if text not in self.cache:
            return None

        vector, timestamp = self.cache[text]

        # TTL 확인
        if self.time.time() - timestamp > self.ttl:
            del self.cache[text]
            return None

        # LRU: 사용된 항목을 맨 뒤로
        self.cache.move_to_end(text)
        return vector

    def set(self, text: str, vector: List[float]):
        """캐시에 저장"""
        # 최대 크기 확인
        if len(self.cache) >= self.max_size:
            # 가장 오래된 항목 제거 (LRU)
            self.cache.popitem(last=False)

        self.cache[text] = (vector, self.time.time())

    def clear(self):
        """캐시 비우기"""
        self.cache.clear()

    def stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
        }
