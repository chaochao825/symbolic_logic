"""Pinned external authorities shared by CLI and evidence builders."""

DATASET_ORIGINS = {
    "ARC-AGI-1": "https://github.com/fchollet/ARC-AGI.git",
    "ARC-AGI-2": "https://github.com/arcprize/ARC-AGI-2.git",
}
DATASET_COMMITS = {
    "ARC-AGI-1": "399030444e0ab0cc8b4e199870fb20b863846f34",
    "ARC-AGI-2": "f3283f727488ad98fe575ea6a5ac981e4a188e49",
}
SPLIT_DIRECTORIES = {
    "public_training": "training",
    "public_evaluation": "evaluation",
}

ARC2_PUBLIC_TRAINING_AUDIT = {
    "available_file_count": 1000,
    "file_count": 1000,
    "selection_policy": "all_files",
    "selected_task_ids_sha256": "8f735c6d10f05654e5bb9e9eadab47becd50c185ef06101e14e83145368736c1",
    "file_manifest_sha256": "440600a22aee239e0a4720b531db20fd86c0fe58a9a1c25da93b606ecf7a39e8",
    "total_train_pairs": 3232,
    "total_test_pairs": 1076,
    "multi_test_task_count": 69,
    "test_outputs_present": 1076,
}
