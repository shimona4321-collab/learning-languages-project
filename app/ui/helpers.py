"""
Input helpers for the Language Exchange Matchmaking System.
"""

from __future__ import annotations


def input_int_in_range(prompt: str, min_val: int, max_val: int) -> int:
    """Get an integer input within the specified range."""
    while True:
        try:
            s = input(prompt).strip()
            val = int(s)
            if min_val <= val <= max_val:
                return val
            print(f"Please enter an integer between {min_val} and {max_val}.")
        except ValueError:
            print("Invalid input. Please enter an integer.")


def input_float(prompt: str) -> float:
    """Get a float input."""
    while True:
        try:
            return float(input(prompt).strip())
        except ValueError:
            print("Invalid input. Please enter a number.")
