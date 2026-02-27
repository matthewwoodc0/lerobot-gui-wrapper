# Backward-compatibility shim: all symbols have been merged into repo_utils.
# This file may be removed in a future cleanup; import from repo_utils directly.
from .repo_utils import (  # noqa: F401
    build_dataset_card_text,
    build_dataset_tag_upload_command,
    default_dataset_tags,
    safe_unlink,
    write_dataset_card_temp,
)
