import os
import re
from pathlib import Path

GSD_ROOT = Path(".gsd")
# Newer GSD versions keep milestone roadmaps under .gsd/phases; this repository's
# existing planning state still lives under .gsd/milestones. Search both, new first.
MILESTONES_ROOTS = (GSD_ROOT / "phases", GSD_ROOT / "milestones")
STATE_FILE = GSD_ROOT / "STATE.md"
MILESTONE_ID_RE = re.compile(r"\b(?P<milestone>M\d+(?:-[0-9A-Za-z]+)?)\b", re.IGNORECASE)
MILESTONE_HEADING_RE = re.compile(
    r"^#\s*(?P<milestone>M\d+(?:-[0-9A-Za-z]+)?)(?::|\s|$)",
    re.IGNORECASE | re.MULTILINE,
)
SLICE_LINE_RE = re.compile(
    r"^\s*-\s*\[(?P<mark>[ xX])\]\s*\*\*(?P<slice>S\d+):\s*(?P<title>[^*]+?)\*\*",
    re.MULTILINE,
)


def read_text(path):
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def output_value(value):
    return str(value).replace("%", "%25").replace("\n", "%0A").replace("\r", "%0D")


def write_outputs(**values):
    if "GITHUB_OUTPUT" not in os.environ:
        return
    output_path = os.environ["GITHUB_OUTPUT"]
    with open(output_path, "a", encoding="utf-8") as output:
        for key, value in values.items():
            output.write(f"{key}={output_value(value)}\n")


def normalize_milestone_id(value):
    match = MILESTONE_ID_RE.search(value or "")
    if not match:
        return (value or "").strip()
    raw = match.group("milestone")
    return f"M{raw[1:]}"


def milestone_key(value):
    return normalize_milestone_id(value).casefold()


def same_milestone(left, right):
    return milestone_key(left) == milestone_key(right)


def parse_state():
    state = read_text(STATE_FILE)
    milestone_match = re.search(r"^\*\*Active Milestone:\*\*\s*(?P<milestone>M[^\s:]+)", state, re.MULTILINE)
    slice_match = re.search(
        r"^\*\*Active Slice:\*\*\s*(?P<slice>S\d+)(?::\s*(?P<title>.+))?",
        state,
        re.MULTILINE,
    )
    completed = set(
        normalize_milestone_id(value)
        for value in re.findall(r"^\s*-\s*\u2705\s*\*\*(?P<milestone>M[^:*]+):", state, re.MULTILINE)
    )
    return {
        "milestone": normalize_milestone_id(milestone_match.group("milestone")) if milestone_match else "",
        "slice": slice_match.group("slice") if slice_match else "",
        "title": (slice_match.group("title") or "").strip() if slice_match else "",
        "completed_milestones": completed,
    }


def default_milestones_root():
    for root in MILESTONES_ROOTS:
        if root.exists():
            return root
    return MILESTONES_ROOTS[0]


def roadmap_files():
    files = []
    for root in MILESTONES_ROOTS:
        if root.exists():
            files.extend(sorted(root.rglob("*-ROADMAP.md")))
    return files


def roadmap_milestone_id(path):
    text = read_text(path)
    heading_match = MILESTONE_HEADING_RE.search(text)
    if heading_match:
        return normalize_milestone_id(heading_match.group("milestone"))
    return normalize_milestone_id(path.parent.name)


def roadmap_records():
    return [
        {
            "milestone": roadmap_milestone_id(path),
            "directory": path.parent,
            "roadmap": path,
        }
        for path in roadmap_files()
    ]


def find_roadmap_record(milestone_id):
    for record in roadmap_records():
        if same_milestone(record["milestone"], milestone_id):
            return record
    return None


def find_roadmap_record_for_directory(directory):
    for record in roadmap_records():
        if record["directory"] == directory:
            return record
    return None


def parse_roadmap(path):
    milestone_id = roadmap_milestone_id(path)
    slices = []
    for match in SLICE_LINE_RE.finditer(read_text(path)):
        slices.append(
            {
                "milestone": milestone_id,
                "slice": match.group("slice"),
                "title": match.group("title").strip(),
                "complete": match.group("mark").strip().lower() == "x",
                "roadmap": str(path),
            }
        )
    return slices


def find_roadmap_slice(milestone_id, slice_id, roadmap_path=None):
    if roadmap_path is not None:
        paths = [roadmap_path]
    else:
        paths = [record["roadmap"] for record in roadmap_records() if same_milestone(record["milestone"], milestone_id)]

    for path in paths:
        if not same_milestone(roadmap_milestone_id(path), milestone_id):
            continue
        for item in parse_roadmap(path):
            if item["slice"] == slice_id:
                return item
    return None


def roadmap_path_for_milestone(milestone_id):
    record = find_roadmap_record(milestone_id)
    if record is not None:
        return record["roadmap"]
    return None


def phase_directory_for_milestone(milestone_id, roadmap_path=None):
    if roadmap_path is not None:
        return roadmap_path.parent
    record = find_roadmap_record(milestone_id)
    if record is not None:
        return record["directory"]
    return default_milestones_root() / milestone_id


def slice_dir(milestone_id, slice_id, roadmap_path=None):
    return phase_directory_for_milestone(milestone_id, roadmap_path) / "slices" / slice_id


def flat_slice_prefix(roadmap_path, slice_id):
    if roadmap_path is None:
        return None
    roadmap_stem = roadmap_path.name.removesuffix("-ROADMAP.md")
    slice_number = slice_id.removeprefix("S").zfill(2)
    return roadmap_path.parent / f"{roadmap_stem}-{slice_number}"


def flat_slice_files(roadmap_path, slice_id, suffix_pattern):
    prefix = flat_slice_prefix(roadmap_path, slice_id)
    if prefix is None:
        return []
    return list(prefix.parent.glob(f"{prefix.name}-{suffix_pattern}"))


def completed_milestone(milestone_id, completed_milestones):
    return any(same_milestone(milestone_id, completed) for completed in completed_milestones)


def slice_artifact_dir(milestone_id, slice_id, roadmap_path=None):
    directory = slice_dir(milestone_id, slice_id, roadmap_path)
    if directory.exists():
        return directory
    return phase_directory_for_milestone(milestone_id, roadmap_path)


def slice_ready_for_implementation(milestone_id, slice_id, completed_milestones, roadmap_path=None):
    if completed_milestone(milestone_id, completed_milestones):
        return False

    roadmap_item = find_roadmap_slice(milestone_id, slice_id, roadmap_path)
    if roadmap_item is not None and roadmap_item["complete"]:
        return False

    directory = slice_dir(milestone_id, slice_id, roadmap_path)
    summary_file = directory / f"{slice_id}-SUMMARY.md"
    flat_summaries = flat_slice_files(roadmap_path, slice_id, "SUMMARY.md")

    if summary_file.exists() or flat_summaries:
        return False

    context_files = list(directory.glob("*CONTEXT*.md"))
    flat_context_files = flat_slice_files(roadmap_path, slice_id, "*CONTEXT*.md")
    return bool(context_files or flat_context_files)


def candidate_roadmap_records(active_milestone):
    records = []
    active_record = find_roadmap_record(active_milestone) if active_milestone else None
    if active_record is not None:
        records.append(active_record)
    records.extend(record for record in roadmap_records() if record not in records)
    return records


def select_implementation_slice():
    state = parse_state()
    completed_milestones = state["completed_milestones"]

    active_milestone = state["milestone"]
    active_slice = state["slice"]
    active_roadmap = roadmap_path_for_milestone(active_milestone)
    if (
        active_milestone
        and active_slice
        and slice_ready_for_implementation(active_milestone, active_slice, completed_milestones, active_roadmap)
    ):
        roadmap_item = find_roadmap_slice(active_milestone, active_slice, active_roadmap) or {}
        milestone_id = roadmap_item.get("milestone", active_milestone)
        return {
            "milestone_id": milestone_id,
            "slice_id": active_slice,
            "slice_title": state["title"] or roadmap_item.get("title", ""),
            "slice_dir": str(slice_artifact_dir(milestone_id, active_slice, active_roadmap)),
            "roadmap_file": roadmap_item.get("roadmap", str(active_roadmap or "")),
            "selection_reason": "active execution-ready slice in .gsd/STATE.md",
        }

    for record in candidate_roadmap_records(active_milestone):
        milestone_id = record["milestone"]
        roadmap_path = record["roadmap"]
        if completed_milestone(milestone_id, completed_milestones):
            continue
        for item in parse_roadmap(roadmap_path):
            if not item["complete"] and slice_ready_for_implementation(
                item["milestone"], item["slice"], completed_milestones, roadmap_path
            ):
                return {
                    "milestone_id": item["milestone"],
                    "slice_id": item["slice"],
                    "slice_title": item["title"],
                    "slice_dir": str(slice_artifact_dir(item["milestone"], item["slice"], roadmap_path)),
                    "roadmap_file": item["roadmap"],
                    "selection_reason": "first unchecked execution-ready slice",
                }

    slice_directories = sorted(
        path
        for root in MILESTONES_ROOTS
        if root.exists()
        for path in root.rglob("slices/S*")
        if path.is_dir()
    )
    for directory in slice_directories:
        phase_directory = directory.parents[1]
        record = find_roadmap_record_for_directory(phase_directory)
        milestone_id = record["milestone"] if record is not None else normalize_milestone_id(phase_directory.name)
        roadmap_path = record["roadmap"] if record is not None else None
        slice_id = directory.name
        if slice_ready_for_implementation(milestone_id, slice_id, completed_milestones, roadmap_path):
            return {
                "milestone_id": milestone_id,
                "slice_id": slice_id,
                "slice_title": "",
                "slice_dir": str(directory),
                "roadmap_file": str(roadmap_path or ""),
                "selection_reason": "slice directory with local task plans",
            }

    return None


def main():
    selected = select_implementation_slice()
    if not selected:
        write_outputs(
            has_incomplete="false",
            milestone_id="",
            slice_id="",
            slice_title="",
            slice_dir="",
            roadmap_file="",
            selection_reason="",
        )
        print("No execution-ready GSD slice found under .gsd.")
    else:
        write_outputs(has_incomplete="true", **selected)
        print(
            "Selected GSD slice ready for implementation: "
            f"{selected['milestone_id']}/{selected['slice_id']} "
            f"{selected['slice_title']} ({selected['selection_reason']})"
        )


if __name__ == "__main__":
    main()
