#!/usr/bin/env python3
"""
Find and replace duplicated descriptions in token description file.
Uses web search to find better descriptions for generic/duplicated ones.
"""

import json
from collections import defaultdict
from pathlib import Path
import argparse


def load_tokens(filepath):
    """Load token descriptions from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_duplicates(tokens):
    """
    Find all duplicated descriptions.
    Returns dict: {description: [list of (token, token_id, line_number)]}
    """
    description_map = defaultdict(list)

    for idx, token_entry in enumerate(tokens):
        desc = token_entry.get('description', '')
        token = token_entry.get('token', '')
        token_id = token_entry.get('token_id', '')

        # Calculate line number (assuming pretty-printed JSON with 6 lines per entry + header)
        # Line 1: [
        # Line 2-7: first entry
        # Line 8-13: second entry, etc.
        line_num = 2 + (idx * 6)  # approximate

        description_map[desc].append({
            'token': token,
            'token_id': token_id,
            'line_number': line_num,
            'index': idx
        })

    # Filter to only duplicates (2+ occurrences)
    duplicates = {desc: items for desc, items in description_map.items()
                  if len(items) > 1}

    return duplicates


def print_duplicates(duplicates):
    """Print duplicated descriptions with tokens and line numbers."""
    print("=" * 80)
    print("DUPLICATED DESCRIPTIONS")
    print("=" * 80)
    print(f"\nFound {len(duplicates)} duplicated descriptions\n")

    # Sort by number of occurrences (descending)
    sorted_dups = sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True)

    for desc, items in sorted_dups:
        print(f"\n{'─' * 80}")
        print(f"Description: \"{desc}\"")
        print(f"Occurrences: {len(items)}")
        print(f"{'─' * 80}")

        for item in items[:20]:  # Show max 20 examples
            print(f"  Line ~{item['line_number']:5d} | "
                  f"Token: {item['token']:20s} | "
                  f"ID: {item['token_id']}")

        if len(items) > 20:
            print(f"  ... and {len(items) - 20} more")

    print("\n" + "=" * 80)


def get_generic_descriptions(duplicates, min_count=10):
    """
    Get most generic/overused descriptions that need replacement.
    Returns list of (description, count, example_tokens)
    """
    generic = []

    for desc, items in duplicates.items():
        if len(items) >= min_count:
            example_tokens = [item['token'] for item in items[:5]]
            generic.append((desc, len(items), example_tokens))

    return sorted(generic, key=lambda x: x[1], reverse=True)


def generate_replacement_script(tokens, duplicates, output_file):
    """
    Generate a Python script to help replace duplicates using web search.
    """
    script_content = '''#!/usr/bin/env python3
"""
Interactive script to replace duplicated descriptions using web search.
Generated automatically by find_duplicate_descriptions.py
"""

import json
import sys
from pathlib import Path

# Web search will be done manually - this script helps organize the replacements


def load_tokens(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_tokens(tokens, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(tokens, f, ensure_ascii=False, indent=2)


def replace_description(tokens, token_id, new_description):
    """Replace description for a specific token_id."""
    for token_entry in tokens:
        if token_entry['token_id'] == token_id:
            token_entry['original_description'] = token_entry.get('description')
            token_entry['description'] = new_description
            return True
    return False


# Duplicated tokens that need better descriptions
DUPLICATES_TO_REPLACE = '''

    # Add duplicate groups
    script_content += "[\n"

    for desc, items in sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True):
        if len(items) >= 5:  # Only include frequent duplicates
            script_content += f"    # Description: '{desc}' ({len(items)} tokens)\n"
            script_content += "    [\n"
            for item in items[:10]:  # Max 10 per group
                script_content += f"        ({item['token_id']}, '{item['token']}'),\n"
            if len(items) > 10:
                script_content += f"        # ... and {len(items) - 10} more\n"
            script_content += "    ],\n\n"

    script_content += "]\n\n"

    script_content += '''

def main():
    input_file = "data/tokenizer/reviewed_new_token_description.json"
    output_file = "data/tokenizer/reviewed_new_token_description_fixed.json"

    print("Loading tokens...")
    tokens = load_tokens(input_file)

    print(f"\\nTotal tokens: {len(tokens)}")
    print(f"\\nDuplicate groups to process: {len(DUPLICATES_TO_REPLACE)}")

    # TODO: For each group, search web for proper descriptions
    # Example workflow:
    # 1. For each token in a duplicate group, search: "medical term {token} meaning"
    # 2. Update the description based on search results
    # 3. Call replace_description(tokens, token_id, new_description)

    print("\\nPlease manually update DUPLICATES_TO_REPLACE with new descriptions")
    print("Then uncomment the save code below")

    # save_tokens(tokens, output_file)
    # print(f"\\nSaved updated tokens to {output_file}")


if __name__ == "__main__":
    main()
'''

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(script_content)

    print(f"\nGenerated replacement helper script: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Find duplicated descriptions in token file"
    )
    parser.add_argument(
        '--input',
        default='data/tokenizer/reviewed_new_token_description.json',
        help='Input JSON file with token descriptions'
    )
    parser.add_argument(
        '--output-script',
        default='script/replace_duplicate_descriptions.py',
        help='Output Python script for replacements'
    )
    parser.add_argument(
        '--min-duplicates',
        type=int,
        default=5,
        help='Minimum occurrences to consider as duplicate (default: 5)'
    )
    parser.add_argument(
        '--show-all',
        action='store_true',
        help='Show all duplicates, not just generic ones'
    )

    args = parser.parse_args()

    print(f"Loading tokens from {args.input}...")
    tokens = load_tokens(args.input)
    print(f"Loaded {len(tokens)} tokens")

    print("\nFinding duplicates...")
    duplicates = find_duplicates(tokens)

    print_duplicates(duplicates)

    # Show most generic descriptions
    print("\n" + "=" * 80)
    print("MOST GENERIC DESCRIPTIONS (need replacement)")
    print("=" * 80)

    generic = get_generic_descriptions(duplicates, min_count=args.min_duplicates)

    for desc, count, examples in generic[:20]:
        print(f"\n'{desc}' - {count} tokens")
        print(f"  Examples: {', '.join(examples)}")

    # Generate replacement script
    generate_replacement_script(tokens, duplicates, args.output_script)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tokens: {len(tokens)}")
    print(f"Duplicated descriptions: {len(duplicates)}")
    print(f"Generic descriptions (≥{args.min_duplicates} occurrences): {len(generic)}")
    print(f"\nNext steps:")
    print(f"1. Review the output above")
    print(f"2. Edit {args.output_script} to add proper descriptions")
    print(f"3. Use web search for medical terms you're unsure about")
    print(f"4. Run the replacement script to update the file")


if __name__ == "__main__":
    main()
