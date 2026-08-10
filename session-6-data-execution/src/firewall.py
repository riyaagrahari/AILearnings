def consume_shard(manifest, expected_split="train"):
    if manifest["split"] != expected_split:
        raise PermissionError(
            f"training firewall blocked shard {manifest['shard_id']} split={manifest['split']}"
        )
    return True
