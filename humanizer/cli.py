"""
Command-line interface for the humanizer package.
"""
import argparse
import sys
from humanizer import TextHumanizer

def main():
    parser = argparse.ArgumentParser(description="Humanize text to make it more readable and natural.")
    parser.add_argument("input", nargs="?", help="Text to humanize (or use stdin if not provided)")
    parser.add_argument("-f", "--file", help="Input file containing text to humanize")
    parser.add_argument("-o", "--output", help="Output file to write humanized text to")
    parser.add_argument("--no-simplify", action="store_true", help="Disable vocabulary simplification")
    parser.add_argument("--no-grammar", action="store_true", help="Disable grammar correction")
    parser.add_argument("--threshold", type=int, default=8, help="Word length threshold for simplification")
    parser.add_argument("--style", help="Path to a trained style model (.pkl file)")
    parser.add_argument("--strength", type=float, default=0.7, help="Style strength (0.0-1.0)")
    
    args = parser.parse_args()
    
    # Initialize humanizer
    humanizer = TextHumanizer()
    
    # Load style model if provided
    if args.style:
        try:
            humanizer.load_style_model(args.style)
            print(f"Loaded style model from {args.style}", file=sys.stderr)
        except Exception as e:
            print(f"Error loading style model: {e}", file=sys.stderr)
    
    # Get input text
    input_text = ""
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                input_text = f.read()
        except Exception as e:
            print(f"Error reading input file: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.input:
        input_text = args.input
    else:
        # Read from stdin if no input argument or file
        try:
            input_text = sys.stdin.read()
        except KeyboardInterrupt:
            sys.exit(0)
    
    # Process text
    try:
        humanized_text = humanizer.humanize(
            input_text,
            simplify=not args.no_simplify,
            correct=not args.no_grammar,
            word_length_threshold=args.threshold,
            apply_style=bool(args.style),
            style_strength=args.strength
        )
        
        # Output result
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(humanized_text)
        else:
            print(humanized_text)
            
    except Exception as e:
        print(f"Error during humanization: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
