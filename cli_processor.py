import os
import json
import requests
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

from audio import clip_segments_with_fade, get_audio_duration_ms
from transcribe import LocalWhisperTranscriber, Segment
from model_output import clean_and_parse_model_output

from dotenv import load_dotenv

load_dotenv()

import litellm

class CLIPodcastProcessor:
    def __init__(self, llm_mode: str = "local", local_model: str = "mixtral", gpt_model: str = "gpt-4o", context_mode: str = "stateless"):
        self.fade_ms = 500
        self.min_confidence = 0.86
        self.min_ad_segment_length_seconds = 3.0
        self.min_ad_segment_separation_seconds = 4.0
        self.llm_mode = llm_mode
        self.local_model = local_model
        self.gpt_model = gpt_model
        self.context_mode = context_mode

        self.transcriber = LocalWhisperTranscriber(self._make_logger(), "turbo")

        self.system_prompt_path = "config/system_prompt.txt"

        if self.llm_mode == "chatgpt":
            litellm.api_key = os.getenv("OPENAI_API_KEY")
            litellm.api_base = "https://api.openai.com/v1"
            if not litellm.api_key:
                raise RuntimeError("OpenAI API key is missing! Set it in .env or hardcode it.")
            self.assistant_id = os.getenv("OPENAI_ASSISTANT_ID") if self.context_mode == "assistant" else None

    def _make_logger(self):
        import logging
        logging.basicConfig(level=logging.WARNING)
        return logging.getLogger("cli")

    def process_file(self, file_path: Path, output_dir: Path) -> None:
        print(f"Generating Transcript. ({file_path})")

        cache_dir = Path(".cache_segments")
        cache_dir.mkdir(exist_ok=True)
        segments_path = cache_dir / f"{file_path.stem}.segments.json"

        if segments_path.exists():
            print("Loading Pre-Cached Transcript.")
            with open(segments_path, "r") as f:
                segments_data = json.load(f)
                segments = [Segment(**json.loads(s)) if isinstance(s, str) else Segment(**s) for s in segments_data]
        else:
            segments = self.transcriber.transcribe(str(file_path))
            self._write_segments_for_debug(segments, segments_path)
            print("Saving Cached Transcript.")

        system_prompt = self._load_text(self.system_prompt_path) if self.context_mode == "stateless" else None

        classified = self._classify_segments(segments, system_prompt)
        ad_segments = self._parse_ad_segments(classified, segments)

        duration_ms = get_audio_duration_ms(str(file_path))
        assert duration_ms is not None

        final_segments = self._merge_ad_segments(ad_segments, duration_ms)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_label = self.gpt_model if self.llm_mode == "chatgpt" else self.local_model
        new_filename = f"{file_path.stem} [adstrip-{model_label}-{timestamp}]{file_path.suffix}"
        output_path = output_dir / new_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        clip_segments_with_fade(
            in_path=str(file_path),
            ad_segments_ms=final_segments,
            fade_ms=self.fade_ms,
            out_path=str(output_path),
        )

        print(f"Processing Completed. File Saved: {output_path}")

    def _write_segments_for_debug(self, segments: List[Segment], path: Path):
        def round_timestamp(value: float) -> float:
            return round(value + 1e-8, 2)

        rounded_segments = []
        for s in segments:
            data = s.dict()
            data["start"] = round_timestamp(data["start"])
            data["end"] = round_timestamp(data["end"])
            rounded_segments.append(data)

        with open(path, "w") as f:
            json.dump(rounded_segments, f, indent=2)

    def _load_text(self, path: str) -> str:
        with open(path, "r") as f:
            return f.read()

    def _classify_segments(
        self,
        segments: List[Segment],
        system_prompt: str
    ) -> List[str]:
        print(f"Filtering Transcript. ({self.llm_mode} {self.gpt_model})")
        predictions = []
        chunk_size = 35

        def round_timestamp(value: float) -> float:
            return round(value + 1e-8, 2)

        if self.context_mode == "assistant":
            from litellm import create_thread
            thread = create_thread(custom_llm_provider="openai", messages=[])

        for i in range(0, len(segments), chunk_size):
            chunk = segments[i:i + chunk_size]
            transcript_text = "\n".join([f"[{round_timestamp(s.start)}] {s.text}" for s in chunk])

            user_prompt = transcript_text

            if self.llm_mode == "chatgpt":
                if self.context_mode == "stateless":
                    response = litellm.completion(
                        model=self.gpt_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    predictions.append(response.choices[0].message.content)
                elif self.context_mode == "assistant":
                    from litellm import add_message
                    add_message(thread_id=thread.id, role="user", content=user_prompt, custom_llm_provider="openai")
                    from litellm import run_thread
                    run = run_thread(thread_id=thread.id, assistant_id=self.assistant_id, custom_llm_provider="openai")
                    predictions.append(run['content'] if isinstance(run, dict) and 'content' in run else str(run))

            elif self.llm_mode == "local":
                prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
                payload = {
                    "model": self.local_model,
                    "prompt": prompt,
                    "stream": False
                }
                res = requests.post("http://localhost:11434/api/generate", json=payload)
                if res.ok:
                    predictions.append(res.json()["response"])
                else:
                    print(f"Error from local model: {res.text}")

        return predictions

    def _parse_ad_segments(
        self, identifications: List[str], segments: List[Segment]
    ) -> List[Tuple[float, float]]:
        all_ad_segments = []
        segment_dict = {s.start: s for s in segments}
        
        print(f"Parsing LLM Filter Suggestions.")

        for output in identifications:
            try:
                parsed_json = json.loads(output)
                parsed_json.setdefault("confidence", 0.0)
                if parsed_json["confidence"] > 0:
                    print(f"{self.llm_mode} response: {output}")
                #print(f"{self.llm_mode} response: {output}")
                parsed_json = json.loads(output)
                if "confidence" not in parsed_json:
                    parsed_json["confidence"] = 0.0
                parsed = clean_and_parse_model_output(json.dumps(parsed_json))

                if parsed.confidence < self.min_confidence:
                    continue

                for start in parsed.ad_segments:
                    if start in segment_dict:
                        end = segment_dict[start].end
                        all_ad_segments.append((start, end))
            except Exception as e:
                print(f"Error parsing model output: {e}\n{self.llm_mode} response: {output}")
        return all_ad_segments

    def _merge_ad_segments(
        self,
        ad_segments: List[Tuple[float, float]],
        duration_ms: int,
    ) -> List[Tuple[int, int]]:
        ad_segments = sorted(ad_segments)
        merged = []

        print(f"Re-encoding Audio File.")

        for start, end in ad_segments:
            if merged and start - merged[-1][1] <= self.min_ad_segment_separation_seconds:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))

        merged = [
            seg for seg in merged
            if seg[1] - seg[0] >= self.min_ad_segment_length_seconds
        ]

        if merged and (duration_ms / 1000 - merged[-1][1] < self.min_ad_segment_separation_seconds):
            merged[-1] = (merged[-1][0], duration_ms / 1000)

        return [(int(start * 1000), int(end * 1000)) for start, end in merged]
