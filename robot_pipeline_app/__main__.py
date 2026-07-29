from __future__ import annotations

from .cli_modes import main

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting safely.")
        raise SystemExit(1)
