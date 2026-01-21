"""
TokenDetokenizer Utility

This script provides functionality to tokenize or detokenize text files using the specified tokenizer model.
It supports processing individual files as well as all files within a directory. The script can auto-detect
whether a file needs tokenization or detokenization based on its content or operate in a mode explicitly
specified by the user.

Usage:
    python tok_detok.py --input <path_to_file_or_directory> [--output <output_directory>] [--model_name <tokenizer_model_name>] [--mode <tokenize|detokenize>]

Arguments:
    --input: Mandatory. The path to the file or directory to process.
    --output: Optional. The directory or filename where the processed files will be saved. Defaults to the same directory as the input file(s).
    --model_name: Optional. The name of the tokenizer model to use. Defaults to 'xlm-roberta-base'.
    --mode: Optional. Sets the operation mode explicitly to 'tokenize' or 'detokenize'. If not specified, the script auto-detects based on file content.
    --keep_unk: Optional. If specified, the script will keep <unk> tokens when detokenizing. By default, <unk> tokens are removed.

Examples:
    Tokenize a single file with default model:
    python tok_detok.py --input ./data/input.txt --output ./data/output --mode tokenize

    Detokenize all files in a directory, using a specific model:
    python tok_detok.py --input ./data/input_dir --output ./data/output_dir --model_name bert-base-uncased --mode detokenize
"""

import os
import argparse
from transformers import AutoTokenizer


class TokenDetokenizer:
    """
    A class for tokenizing and detokenizing text files using a specified tokenizer model.
    """

    def __init__(self, model_name="xlm-roberta-base", keep_unk=False):
        """
        Initializes the TokenDetokenizer with a tokenizer model.

        :param model_name: Name of the tokenizer model to use.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.keep_unk = keep_unk

    def is_tokenized(self, file_path):
        """
        Checks if a file is already tokenized.

        :param file_path: Path to the file to check.
        :return: True if the file is tokenized, False otherwise.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            tokenized_lines = 0
            for i, line in enumerate(f):
                if i >= 5:
                    break
                if "<s>" in line or "</s>" in line or line.startswith("▁"):
                    tokenized_lines += 1
            if tokenized_lines >= 3:
                return True
        return False

    def tokenize(self, infile, outfile):
        """
        Tokenizes a file.

        :param infile: Path to the input file.
        :param outfile: Path to the output file.
        """
        print(f"Tokenizing {infile} to {outfile}...")
        with open(infile, "r", encoding="utf-8") as f_in, open(
            outfile, "w", encoding="utf-8"
        ) as f_out:
            for line in f_in:
                tokenized_ids = self.tokenizer(line.strip()).input_ids
                line_tok_list = self.tokenizer.convert_ids_to_tokens(
                    tokenized_ids, skip_special_tokens=True
                )
                line_tok = " ".join(line_tok_list) + "\n"
                f_out.write(line_tok)
        print(f"Tokenization complete.")

    def detokenize(self, infile, outfile):
        """
        Detokenizes a file, keeping all tokens except special ones, including <unk> if the --keep_unk flag is used.

        :param infile: Path to the input file.
        :param outfile: Path to the output file.
        """
        print(f"Detokenizing {infile} to {outfile}...")
        with open(infile, "r", encoding="utf-8") as f_in, open(
            outfile, "w", encoding="utf-8"
        ) as f_out:
            for line in f_in:
                token_ids = self.tokenizer.convert_tokens_to_ids(line.split())
                # Decode directly if keep_unk is True, as <unk> tokens are to be preserved
                if self.keep_unk:
                    detokenized_text = self.tokenizer.decode(
                        token_ids, skip_special_tokens=False
                    )
                    # Manually remove special tokens except <unk>
                    detokenized_text = detokenized_text.replace(
                        self.tokenizer.cls_token, ""
                    )
                    detokenized_text = detokenized_text.replace(
                        self.tokenizer.sep_token, ""
                    )
                    detokenized_text = detokenized_text.replace(
                        self.tokenizer.pad_token, ""
                    )
                    # Add more replacements here for other special tokens as necessary
                else:
                    # Skip all special tokens, including <unk>
                    detokenized_text = self.tokenizer.decode(
                        token_ids, skip_special_tokens=True
                    )
                f_out.write(
                    detokenized_text.strip() + "\n"
                )  # strip() removes leading/trailing spaces
        print(f"Detokenization complete.")

    def process_file(self, infile, outfile=None, mode=None):
        """
        Processes a file, either tokenizing or detokenizing it based on its current state or a specified mode.

        :param infile: Path to the input file.
        :param outfile: Path to the output file. If not specified, modifies the input file name accordingly or uses the specified filename.
        :param mode: Operation mode ('tokenize' or 'detokenize'). If not specified, auto-detects based on file content.
        """
        # Check if outfile is a directory. If so, construct a filename based on infile.
        if outfile and os.path.isdir(outfile):
            base_name = os.path.basename(infile)
            # Modify the filename based on the mode
            new_filename = base_name + (
                ".detok"
                if mode == "detokenize" or self.is_tokenized(infile)
                else ".tok"
            )
            outfile = os.path.join(outfile, new_filename)

        if not outfile:
            outfile = infile + (".detok" if self.is_tokenized(infile) else ".tok")

        if mode == "detokenize" or (mode is None and self.is_tokenized(infile)):
            self.detokenize(infile, outfile)
        else:
            self.tokenize(infile, outfile)

    def process_directory(self, indir, outdir=None, mode=None):
        """
        Processes all files in a directory, either tokenizing or detokenizing them based on their current state or a specified mode.

        :param indir: Path to the input directory.
        :param outdir: Path to the output directory. If not specified, uses the input directory.
        :param mode: Operation mode ('tokenize' or 'detokenize'). If not specified, auto-detects based on file content.
        """
        if not outdir:
            outdir = indir
        for filename in os.listdir(indir):
            full_path = os.path.join(indir, filename)
            if os.path.isfile(full_path):
                output_filename = filename + (
                    ".detok"
                    if self.is_tokenized(full_path) and mode != "tokenize"
                    else ".tok"
                )
                output_file_path = os.path.join(outdir, output_filename)
                self.process_file(full_path, output_file_path, mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize or detokenize files or directories using a specified tokenizer model."
    )
    parser.add_argument(
        "--input", help="Path to the file or directory to process.", required=True
    )
    parser.add_argument(
        "--output",
        help="Output directory or filename. If not specified, uses the same directory as the input file(s) or modifies the input filename.",
    )
    parser.add_argument(
        "--model_name", default="xlm-roberta-base", help="Model name for the tokenizer."
    )
    parser.add_argument(
        "--mode",
        choices=["tokenize", "detokenize"],
        help="Manually set the operation mode (tokenize or detokenize). If not specified, auto-detects based on file content.",
    )
    parser.add_argument(
        "--keep_unk",
        action="store_true",
        help="Keep <unk> tokens when detokenizing. By default, <unk> tokens are removed.",
    )
    args = parser.parse_args()

    print("Initializing TokenDetokenizer...")

    token_detokenizer = TokenDetokenizer(
        model_name=args.model_name, keep_unk=args.keep_unk
    )

    if os.path.isfile(args.input):
        outfile = args.output
        token_detokenizer.process_file(args.input, outfile, args.mode)
    elif os.path.isdir(args.input):
        if args.output and not os.path.exists(args.output):
            os.makedirs(args.output)
        token_detokenizer.process_directory(args.input, args.output, args.mode)
    else:
        print(f"The path {args.input} is neither a file nor a directory.")

