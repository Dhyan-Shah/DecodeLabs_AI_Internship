# =============================================================================
#  Tech Stack Recommender — DecodeLabs | AI Project 3
#  Author  : DecodeLabs Trainee
#  Concept : Content-Based Filtering using TF-IDF + Cosine Similarity
# =============================================================================

import csv
import math


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA INGESTION
# Load the job-role dataset from raw_skills.csv
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(filepath: str) -> dict[str, list[str]]:
    """
    Reads raw_skills.csv and returns a dictionary:
      { "Data Scientist": ["Python", "SQL", ...], ... }
    """

    dataset = {}
    try:
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                role  = row["job_role"].strip()
                skills = [s.strip() for s in row["skills"].split()]
                dataset[role] = skills
    except FileNotFoundError:
        print(f"\n[ERROR] Dataset file not found: '{filepath}'")
        print("Make sure 'raw_skills.csv' is in the same folder as this script.\n")
        raise SystemExit(1)
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — VECTOR MAPPING  (build the shared vocabulary)
# Every unique skill across ALL roles becomes one dimension in our vector space.
# ─────────────────────────────────────────────────────────────────────────────

def build_vocabulary(dataset: dict[str, list[str]]) -> list[str]:
    """Returns a sorted list of every unique skill term in the dataset."""
    vocab = set()
    for skills in dataset.values():
        vocab.update(skills)
    return sorted(vocab)



# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — TF-IDF FEATURE EXTRACTION
#
# TF  (Term Frequency)        = count(term in doc) / total_terms_in_doc
# IDF (Inverse Doc Frequency) = log(total_docs / docs_containing_term)
# Weight                      = TF × IDF
#
# Why TF-IDF instead of raw binary 1/0?
#   Binary vectors treat every skill equally.  TF-IDF rewards skills that are
#   distinctive to a role and penalises skills that appear everywhere (like
#   "Git" or "Python") so they don't dominate the similarity score.
# ─────────────────────────────────────────────────────────────────────────────


def compute_tf(skills: list[str]) -> dict[str, float]:
    """Term Frequency: how often each skill appears relative to doc length."""
    tf = {}
    total = len(skills)
    for skill in skills:
        tf[skill] = tf.get(skill, 0) + 1
    return {skill: count / total for skill, count in tf.items()}


def compute_idf(dataset: dict[str, list[str]]) -> dict[str, float]:
    """
    Inverse Document Frequency: penalises skills common across many roles.
    Uses smoothed IDF to avoid division-by-zero edge cases.
    """
    total_docs = len(dataset)
    doc_freq   = {}

    for skills in dataset.values():
        unique_skills = set(skills)
        for skill in unique_skills:
            doc_freq[skill] = doc_freq.get(skill, 0) + 1

    idf = {}
    for skill, freq in doc_freq.items():
        idf[skill] = math.log((total_docs + 1) / (freq + 1)) + 1   # smoothed
    return idf


def compute_tfidf_vector(skills: list[str],
                        idf: dict[str, float],
                        vocabulary: list[str]) -> list[float]:
    """
    Converts a list of skill strings into a TF-IDF weighted numerical vector
    aligned to the shared vocabulary.
    """
    tf     = compute_tf(skills)
    vector = []
    for term in vocabulary:
        weight = tf.get(term, 0.0) * idf.get(term, 0.0)
        vector.append(weight)
    return vector


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — COSINE SIMILARITY (the Similarity Engine)
#
# cos(θ) = (A · B) / (||A|| × ||B||)
#
# Why Cosine and not Euclidean?
#   Euclidean distance is sensitive to vector magnitude — a role with 15 skills
#   would always seem "far" from a user who listed 3 skills, even if the
#   direction (pattern of interests) is identical.  Cosine similarity measures
#   ANGLE only, making it magnitude-invariant and perfect for sparse text data.
#
# Score interpretation:
#   1.0  → perfect alignment     |  0.0 → no shared characteristics
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Returns the cosine similarity between two equal-length vectors."""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a ** 2 for a in vec_a))
    mag_b = math.sqrt(sum(b ** 2 for b in vec_b))

    if mag_a == 0 or mag_b == 0:
        # Cold-start guard: zero vector means no data → similarity undefined
        return 0.0

    return dot_product / (mag_a * mag_b)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — THE 4-STEP RANKING PIPELINE
#   Step 1: Ingestion  → capture user skills
#   Step 2: Scoring    → compute cosine similarity for every role
#   Step 3: Sorting    → rank results highest → lowest
#   Step 4: Filtering  → return only the Top-N matches
# ─────────────────────────────────────────────────────────────────────────────

def recommend(user_skills: list[str],
              dataset:     dict[str, list[str]],
              vocabulary:  list[str],
              idf:         dict[str, float],
              top_n:       int = 3) -> list[tuple[str, float]]:
    
    """
    Full recommendation pipeline.

    Parameters
    ----------
    user_skills : list[str]  — skills entered by the user
    dataset     : dict       — job roles with their skill lists
    vocabulary  : list[str]  — shared feature space
    idf         : dict       — pre-computed IDF weights
    top_n       : int        — how many recommendations to return

    Returns
    -------
    List of (job_role, similarity_score) tuples, sorted best-first.
    """

    # Step 1 — Ingestion: vectorise the user profile
    user_vector = compute_tfidf_vector(user_skills, idf, vocabulary)

    # Cold-start guard: if every weight is zero the user gave unknown skills
    if all(w == 0 for w in user_vector):
        return []

    # Step 2 — Scoring: score every role against the user vector
    scored = []
    for role, skills in dataset.items():
        role_vector = compute_tfidf_vector(skills, idf, vocabulary)
        score       = cosine_similarity(user_vector, role_vector)
        scored.append((role, round(score, 4)))

    # Step 3 — Sorting: best scores first
    scored.sort(key=lambda x: x[1], reverse=True)

    # Step 4 — Filtering: truncate to Top-N to prevent choice overload
    return scored[:top_n]



# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — USER INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def get_user_skills(min_skills: int = 3) -> list[str]:
    """
    Prompts the user to enter their skills one by one.
    Enforces a minimum of 3 skills for sufficient data density (as per spec).
    """
    print("\n" + "═" * 60)
    print("   🤖  TECH STACK RECOMMENDER — DecodeLabs AI Project 3")
    print("═" * 60)
    print(f"\nEnter at least {min_skills} skills to get personalised job-role")
    print("recommendations.  Type 'done' when finished.\n")

    skills   = []
    examples = ["Python", "SQL", "Machine_Learning", "Docker", "React"]

    while True:
        prompt = f"  Skill {len(skills) + 1}: "
        entry  = input(prompt).strip()

        if entry.lower() == "done":
            if len(skills) < min_skills:
                print(f"\n  ⚠  Please enter at least {min_skills} skills "
                      f"(you have {len(skills)}).\n")
            else:
                break
        elif entry == "":
            print("  ⚠  Skill cannot be empty. Try again.")
        else:
            # Normalise: replace spaces with underscores to match vocabulary
            normalised = entry.replace(" ", "_").title()
            if normalised in skills:
                print(f"  ⚠  '{normalised}' already added.")
            else:
                skills.append(normalised)
                print(f"  ✔  Added: {normalised}")

                if len(skills) == min_skills:
                    print(f"\n  (Minimum met! Keep adding or type 'done'.)")
                    print(f"  Example skills: {', '.join(examples)}\n")

    return skills


def display_results(user_skills: list[str],
                    recommendations: list[tuple[str, float]]) -> None:
    """Prints the Top-N recommendations in a clean, readable format."""

    print("\n" + "═" * 60)
    print("   📊  YOUR RESULTS")
    print("═" * 60)
    print(f"\n  Your Skills : {', '.join(user_skills)}\n")

    if not recommendations:
        print("  ⚠  No matches found. Your skills don't overlap with any")
        print("     role in the dataset. Try different or more specific skills.")
        return

    print("  Top Recommended Job Roles:\n")
    medals = ["🥇", "🥈", "🥉"]

    for rank, (role, score) in enumerate(recommendations, start=1):
        medal      = medals[rank - 1] if rank <= 3 else f"  {rank}."
        bar_filled = int(score * 30)
        bar        = "█" * bar_filled + "░" * (30 - bar_filled)
        pct        = score * 100
        print(f"  {medal}  {role}")
        print(f"       [{bar}] {pct:.1f}% match")
        print()

    print("═" * 60)
    print("  💡 Tip: The match % is your cosine similarity score —")
    print("     how closely your skill profile aligns with each role.")
    print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    DATASET_PATH = "raw_skills.csv"   # must be in the same folder
    TOP_N        = 3                  # how many roles to recommend

    # --- Build the recommendation engine ---
    dataset    = load_dataset(DATASET_PATH)
    vocabulary = build_vocabulary(dataset)
    idf        = compute_idf(dataset)

    print(f"\n  ✅  Dataset loaded: {len(dataset)} job roles, "
        f"{len(vocabulary)} unique skills in vocabulary.")

    # --- Run the interactive loop ---
    while True:
        user_skills      = get_user_skills(min_skills=3)
        recommendations  = recommend(user_skills, dataset, vocabulary,
                                    idf, top_n=TOP_N)
        display_results(user_skills, recommendations)

        again = input("  Would you like to try again? (yes / no): ").strip().lower()
        if again not in ("yes", "y"):
            print("\n  👋  Thanks for using the Tech Stack Recommender!\n")
            break


if __name__ == "__main__":
    main()

