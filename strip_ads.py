import argparse
from pathlib import Path
from cli_processor import CLIPodcastProcessor

def main():
    parser = argparse.ArgumentParser(description="Strip ads from podcast audio.")
    parser.add_argument("input", type=str, help="Input file or directory")
    parser.add_argument("--output", type=str, help="Output directory", default="output")
    parser.add_argument("--llm", type=str, choices=["local", "chatgpt"], default="chatgpt", help="LLM mode to use")
    parser.add_argument("--model", type=str, help="LLM model to use", default="gpt-4o")
    parser.add_argument("--context", type=str, choices=["stateless", "assistant"], default="stateless", help="Conversation context mode")

    args = parser.parse_args()
    processor = CLIPodcastProcessor(
        llm_mode=args.llm or "chatgpt",
        gpt_model=args.model or "gpt-4o",
        local_model=args.model or "mixtral",
        context_mode=args.context or "stateless"
    )

    input_path = Path(args.input)
    if input_path.is_file():
        processor.process_file(input_path, Path(args.output))
    else:
        for audio_file in input_path.glob("*.mp3"):
            processor.process_file(audio_file, Path(args.output))

if __name__ == "__main__":
    main()



