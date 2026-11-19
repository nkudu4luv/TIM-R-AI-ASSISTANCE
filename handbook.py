from pathlib import Path

DEFAULT_HANDBOOK_TEXT = """TIM-R ACADEMY SCHOOL HANDBOOK

1. ADMISSION POLICY

Tim-R Academy admits students into
primary and secondary school programmes.

Applicants must submit a completed
admission form, birth certificate,
previous academic records and two
passport photographs.

2. SCHOOL FEES

The annual tuition fee for secondary
school students is 250,000 naira.

The annual tuition fee for primary
school students is 180,000 naira.

School fees should be paid before
the beginning of each academic term.

3. ATTENDANCE POLICY

Students are expected to maintain
at least 75 percent attendance.

Students with attendance below
75 percent may be restricted from
participating in examinations,
subject to school management review.

4. SCHOOL RESUMPTION

The school resumes on Monday,
12 January 2027.

Students should arrive before
8:00 AM every school day.

5. EXAMINATION POLICY

Students participate in three major
assessment periods each academic year.

These include first-term examinations,
second-term examinations and
third-term examinations.

Students must complete all required
continuous assessment activities.
"""


def find_handbook_path(base_dir: Path):
    for candidate in (
        base_dir / "documents" / "school_handbook.txt",
        base_dir / "school_handbook.txt",
    ):
        if candidate.exists():
            return candidate
    return None


def split_document(text: str, chunk_size: int = 200, overlap: int = 50):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be zero or greater")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap

    return chunks
