import base64
import os
import tempfile
import subprocess
import logging
import httpx
from typing import Dict, Any, List
from urllib.parse import urlparse
import requests
from openai import OpenAI

from audio_evals.base import PromptStruct
from audio_evals.models.model import APIModel

logger = logging.getLogger(__name__)


class Gemini(APIModel):
    """
    Gemini model using via OpenAI API protocol (via third-party proxy like OneAPI).
    Supports audio input through OpenAI-compatible API.

    Environment variables:
    - GEMINI_API_KEY or OPENAI_API_KEY: API key for the service
    - GEMINI_API_BASE or OPENAI_API_BASE: Base URL for the API endpoint

    Usage:
    ```yaml
    gemini-2.5-pro:
      class: audio_evals.models.google_api.Gemini
      args:
        model_name: 'gemini-2.5-pro'
    ```
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-pro",
        api_key: str = None,
        base_url: str = None,
        sample_params: Dict[str, Any] = None,
    ):
        super().__init__(True, sample_params)
        self.model_name = model_name

        # Get API key and base URL from environment or parameters
        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if base_url is None:
            base_url = os.environ.get("GEMINI_API_BASE") or os.environ.get("OPENAI_API_BASE")

        if not api_key:
            raise ValueError(
                "not found GEMINI_API_KEY or OPENAI_API_KEY in your ENV"
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client(
                base_url=base_url,
                follow_redirects=True,
            ),
        )

    # Supported audio formats for conversion
    SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3"}

    def _is_url(self, path: str) -> bool:
        """Check if path is a URL."""
        return path.startswith(("http://", "https://"))

    def _get_audio_ext(self, audio_path: str) -> str:
        """Get audio file extension from file path or URL."""
        if self._is_url(audio_path):
            parsed = urlparse(audio_path)
            path = parsed.path
        else:
            path = audio_path
        return os.path.splitext(path)[1].lower()

    def _convert_to_wav(self, audio_data: bytes, original_ext: str) -> bytes:
        """Convert audio data to wav format using ffmpeg."""
        with tempfile.NamedTemporaryFile(
            suffix=original_ext, delete=False
        ) as input_file:
            input_file.write(audio_data)
            input_path = input_file.name

        output_path = input_path.rsplit(".", 1)[0] + ".wav"

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    input_path,
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    output_path,
                ],
                capture_output=True,
                check=True,
            )
            with open(output_path, "rb") as f:
                wav_data = f.read()
            return wav_data
        finally:
            # Clean up temp files
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_path):
                os.remove(output_path)

    def _load_audio_data(self, audio_path: str) -> bytes:
        """Load audio data from local file or URL."""
        if self._is_url(audio_path):
            response = requests.get(audio_path, timeout=30)
            response.raise_for_status()
            return response.content
        else:
            with open(audio_path, "rb") as audio_file:
                return audio_file.read()

    def _encode_audio(self, audio_path: str) -> tuple:
        """
        Encode audio file to base64 string. Supports both local files and URLs.
        Converts unsupported formats (non wav/mp3) to wav using ffmpeg.

        Returns:
            tuple: (base64_encoded_data, audio_format)
        """
        ext = self._get_audio_ext(audio_path)
        audio_data = self._load_audio_data(audio_path)

        # Convert to wav if format is not supported
        if ext not in self.SUPPORTED_AUDIO_FORMATS:
            logger.info(f"Converting audio from {ext} to wav format")
            audio_data = self._convert_to_wav(audio_data, ext)
            audio_format = "wav"
        else:
            audio_format = "wav" if ext == ".wav" else "mp3"

        return base64.b64encode(audio_data).decode("utf-8"), audio_format

    def _build_message_content(self, contents: List[Dict]) -> List[Dict]:
        """
        Build OpenAI message content from PromptStruct contents.
        Handles text and audio content types.
        """
        message_content = []
        for item in contents:
            content_type = item.get("type")
            value = item.get("value")

            if content_type == "text":
                message_content.append({"type": "text", "text": value})
            elif content_type == "audio":
                print("Get value:", value)
                audio_data, audio_format = self._encode_audio(value)
                print("audio_data")
                message_content.append(
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_data, "format": audio_format},
                    }
                )
        
        return message_content

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        """
        Perform inference with audio support using OpenAI API protocol.

        Args:
            prompt: PromptStruct containing messages with text/audio content
            **kwargs: Additional arguments passed to API

        Returns:
            str: Text response
        """
        messages = []
        for item in prompt:
            role = item["role"]
            contents = item.get("contents", [])

            # Handle simple text content (backward compatibility)
            if len(contents) == 1 and contents[0].get("type") == "text":
                messages.append({"role": role, "content": contents[0]["value"]})
            else:
                # Handle multimodal content
                message_content = self._build_message_content(contents)
                messages.append({"role": role, "content": message_content})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )

        print(response.choices[0].message.content)

        return response.choices[0].message.content
