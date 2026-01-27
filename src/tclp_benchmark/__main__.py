"""CLI entry point for TCLP Benchmark."""
import sys
from .cli.commands import cli


def main():
    """Main entry point."""
    return cli()


if __name__ == "__main__":
    sys.exit(main())
