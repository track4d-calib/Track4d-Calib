"""Validate native file accesses against the declared calibration inputs."""

from pathlib import Path
import re

OPEN_RESULT = re.compile(r"=\s*\d+<(/[^>]+)>")
TRACE_PID = re.compile(r"^(?:\[pid\s+(\d+)\]|(\d+))\s+")
DIRECTORY_OPEN = re.compile(r",\s*(?:[A-Z0-9_]+\|)*O_DIRECTORY\s*(?:\||\)|,)")

def audit_external_results(
    trace: Path, run_root: Path, sources: list[Path], *, trusted_roots: tuple[Path, ...] = ()
) -> dict:
    """Audit resolved openat file descriptors, including native library IO."""

    forbidden = set()
    observed = set()
    source_rgb = 0
    source_directory_opens = 0
    source_directories = set()
    external_camera_files = set()
    external_dump_files = set()
    fresh_temporary_files = set()
    unfinished = {}
    unmatched_resumptions = []
    reassembled_calls = 0
    classified = {}
    declared_rgb = {
        image.resolve()
        for source in sources
        for image in (source / "images").glob("*/*.png")
        if image.is_file()
    }
    declared_intrinsics = {(source / "cameras/intri.yml").resolve() for source in sources}
    declared_directories = {
        directory.resolve()
        for source in sources
        for directory in [source / "images", *(source / "images").glob("*")]
        if directory.is_dir()
    }
    dump_suffixes = {".npy", ".npz", ".h5", ".hdf5", ".pkl", ".pickle", ".db", ".sqlite", ".bin"}
    for line in trace.open(encoding="utf-8", errors="replace"):
        if "openat" not in line:
            continue
        pid_match = TRACE_PID.match(line)
        pid = (pid_match.group(1) or pid_match.group(2)) if pid_match else "single"
        if "<unfinished ...>" in line:
            unfinished[pid] = line.split("<unfinished ...>", 1)[0]
            continue
        if "<... openat resumed>" in line:
            prefix = unfinished.pop(pid, None)
            if prefix is None:
                unmatched_resumptions.append(line.strip())
                continue
            line = prefix + line.split("<... openat resumed>", 1)[1]
            reassembled_calls += 1
        match = OPEN_RESULT.search(line)
        if match is None:
            continue
        path_text = match.group(1).removesuffix(" (deleted)")
        observed.add(path_text)
        if path_text.startswith("/tmp/") and "O_CREAT" in line and "O_EXCL" in line:
            # Check creation before the classification cache: it must not
            # retroactively erase a forbidden read earlier in this trace.
            fresh_temporary_files.add(path_text)
        if path_text in fresh_temporary_files:
            continue
        if DIRECTORY_OPEN.search(line):
            directory = Path(path_text)
            if directory in declared_directories:
                # Enumerating declared RGB folders is not reading pose/dump
                # content. Do not cache permission for later non-directory IO.
                source_directory_opens += 1
                source_directories.add(path_text)
                continue
        if path_text in classified:
            source_rgb += int(classified[path_text])
            continue
        path = Path(path_text)
        classified[path_text] = False
        if path.is_relative_to(run_root):
            continue
        allowed_rgb = path.suffix == ".png" and (
            path in declared_rgb
            or any(path.is_relative_to(source / "images") for source in sources)
        )
        allowed_intrinsics = path.name == "intri.yml" and path in declared_intrinsics
        if allowed_rgb:
            source_rgb += 1
            classified[path_text] = True
        if allowed_rgb or allowed_intrinsics:
            continue
        if path.name == "extri.yml":
            external_camera_files.add(str(path))


        if "experiments" in path.parts or path.name == "extri.yml":
            forbidden.add(str(path))
        if path.suffix in dump_suffixes and not any(path.is_relative_to(prefix) for prefix in trusted_roots):
            external_dump_files.add(str(path))
            forbidden.add(str(path))
    return {
        "trace": str(trace),
        "resolved_unique_file_count": len(observed),
        "source_RGB_open_calls": source_rgb,
        "declared_image_directory_opens": source_directory_opens,
        "declared_image_directories": sorted(source_directories),
        "external_extrinsics": sorted(external_camera_files),
        "external_feature_match_track_or_binary_dumps": sorted(external_dump_files),
        "exclusively_created_same_run_temporary_files": sorted(fresh_temporary_files),
        "unexpected_experiment_artifact_opens": sorted(forbidden),
        "reassembled_openat_calls": reassembled_calls,
        "unmatched_openat_resumptions": unmatched_resumptions,
        "unfinished_openat_pids": sorted(unfinished),
        "passed": not forbidden and not unmatched_resumptions and not unfinished,
        "scope": "successful resolved openat descriptors; declared RGB/K, explicit code/model/library roots and same-run O_CREAT|O_EXCL temporary files only",
    }
