# Tech Stack Recommender
### DecodeLabs Industrial Training Kit — AI Project 3
**Batch 2026 | Content-Based Filtering using TF-IDF + Cosine Similarity**

---

## What This Project Does

The Tech Stack Recommender is a command-line AI tool that takes a user's skills as input and recommends the **Top 3 most relevant job roles** from a dataset of 20 tech careers. It works entirely through mathematical similarity logic — no external AI APIs, no pre-trained models, just pure algorithmic reasoning built from scratch.

**Example:**
```
Input skills  → Python, SQL, Machine_Learning
Output        → 1. Data Scientist (41.7%)
                2. ML Engineer (27.9%)
                3. AI Researcher (25.7%)
```

---

## Project Structure

```
project3/
├── tech_stack_recommender.py   ← main script
├── raw_skills.csv              ← dataset (20 job roles, 98 unique skills)
└── README.md                   ← this file
```

---

## How to Run

**Requirements:** Python 3.10 or above. No third-party libraries needed — only the built-in `csv` and `math` modules are used.

**Step 1 — Place both files in the same folder:**
```
raw_skills.csv
tech_stack_recommender.py
```

**Step 2 — Run the script:**
```bash
python tech_stack_recommender.py
```

**Step 3 — Follow the prompts:**
```
Skill 1: Python
  ✔  Added: Python
Skill 2: Docker
  ✔  Added: Docker
Skill 3: Kubernetes
  ✔  Added: Kubernetes
  (Minimum met! Keep adding or type 'done'.)
Skill 4: done
```

**Step 4 — View your recommendations:**
```
  🥇  DevOps Engineer
       [██████████████████████░░░░░░░░] 73.4% match

  🥈  Cloud Architect
       [████████████████████░░░░░░░░░░] 66.1% match

  🥉  Site Reliability Engineer
       [████████████████░░░░░░░░░░░░░░] 54.2% match
```

---

## Skill Input Tips

- Enter one skill per line
- Use underscores for multi-word skills: `Machine_Learning`, `Deep_Learning`, `Cloud_Computing`
- Skills are case-insensitive — `python`, `Python`, and `PYTHON` all work
- Type `done` when finished (minimum 3 skills required)
- Type `yes` after results to try again with different skills

**Sample skills to try:**
| Domain | Skills |
|---|---|
| Data Science | Python, SQL, Machine_Learning, Statistics, TensorFlow |
| DevOps | AWS, Docker, Kubernetes, CI_CD, Linux |
| Frontend | JavaScript, React, HTML, CSS, TypeScript |
| Backend | Java, Python, REST_APIs, PostgreSQL, Django |
| Security | Networking, Linux, Ethical_Hacking, Cryptography, Python |

---

## How It Works — The Algorithm

The script implements a **Content-Based Filtering** recommendation engine using a 4-step pipeline:

```
Step 1 — Ingestion   : User skills → TF-IDF weighted vector (98 dimensions)
Step 2 — Scoring     : Cosine similarity computed against every job role
Step 3 — Sorting     : All 20 roles ranked highest score → lowest
Step 4 — Filtering   : Top 3 returned to prevent choice overload
```

### Key Concepts Used

**Vector Mapping**
Every unique skill across all 20 roles forms a shared vocabulary of 98 skills. Both user profiles and job roles are converted into numerical arrays (vectors) aligned to this vocabulary. Machines compare numbers, not words.

**TF-IDF Weighting**
Binary 1/0 vectors treat "Git" (used by everyone) the same as "TensorFlow" (used by few). TF-IDF solves this:
- `TF (Term Frequency)` = how often a skill appears in one role
- `IDF (Inverse Document Frequency)` = how rare the skill is across all roles
- `Weight = TF × IDF` — rare, specific skills get high weights; common skills get low weights

**Cosine Similarity**
Measures the *angle* between two vectors, not their length. This makes it magnitude-invariant — a user with 3 skills and a role with 15 skills can still score 1.0 if their skill directions perfectly align.

```
cos(θ) = (A · B) / (‖A‖ × ‖B‖)

Score 1.0 → perfect match
Score 0.5 → partial match
Score 0.0 → no shared characteristics
```

**Cold Start Guard**
If all user-entered skills are unrecognised by the vocabulary, the user vector becomes all zeros. A zero vector has no direction, making cosine similarity undefined. The script detects this early and returns an empty list with a helpful message.

---

## Dataset — raw_skills.csv

The dataset contains 20 job roles with their required skills:

| Job Role | Key Skills |
|---|---|
| Data Scientist | Python, SQL, ML, Statistics, TensorFlow, Pandas |
| ML Engineer | Python, TensorFlow, PyTorch, Docker, MLOps |
| Backend Developer | Python, Java, SQL, REST_APIs, Django, PostgreSQL |
| Frontend Developer | JavaScript, React, CSS, TypeScript, Figma |
| Full Stack Developer | JavaScript, Python, React, Node.js, MongoDB |
| DevOps Engineer | AWS, Docker, Kubernetes, CI_CD, Terraform |
| Cloud Architect | AWS, Azure, GCP, Kubernetes, Networking |
| Data Engineer | Python, Spark, Kafka, Airflow, ETL |
| Cybersecurity Analyst | Networking, Linux, Ethical_Hacking, Cryptography |
| Mobile Developer | Swift, Kotlin, React_Native, Flutter, iOS |
| AI Researcher | Python, Deep_Learning, NLP, Mathematics, Research |
| Database Administrator | SQL, PostgreSQL, Oracle, Performance_Tuning |
| Blockchain Developer | Solidity, Ethereum, Web3, Smart_Contracts |
| Systems Programmer | C, C++, Linux, Assembly, OS_Development |
| QA Engineer | Selenium, Testing, Automation, CI_CD, Bug_Tracking |
| Data Analyst | SQL, Power_BI, Tableau, Excel, Visualization |
| Site Reliability Engineer | Linux, Python, Kubernetes, Monitoring, AWS |
| NLP Engineer | Python, NLP, BERT, Transformers, Text_Processing |
| Computer Vision Engineer | Python, OpenCV, CUDA, Image_Processing, C++ |
| Product Manager | Agile, Roadmapping, Jira, Data_Analysis, Leadership |

You can extend this dataset by adding more rows to `raw_skills.csv` following the same format:
```
job_role,skills
Your_Role,Skill1 Skill2 Skill3 Skill4 Skill5
```

---

## Code Structure

The script is organised into 7 clearly commented sections:

```
Section 1 — load_dataset()          Reads raw_skills.csv into a dictionary
Section 2 — build_vocabulary()      Builds the shared 98-skill vocabulary
Section 3 — compute_tf/idf/tfidf()  TF-IDF feature extraction (3 functions)
Section 4 — cosine_similarity()     Similarity engine — core math
Section 5 — recommend()             Full 4-step ranking pipeline
Section 6 — get_user_skills()       User input with validation
            display_results()       Formatted output with progress bars
Section 7 — main()                  Entry point and run loop
```

---

## Extending the Project

Here are ways to build on this foundation:

**Add more job roles** — simply append rows to `raw_skills.csv`. The vocabulary and IDF weights rebuild automatically at runtime.

**Change the number of recommendations** — edit the `TOP_N` constant in `main()`:
```python
TOP_N = 5   # show top 5 instead of top 3
```

**Save results to a file** — add this after `display_results()` in `main()`:
```python
with open("results.txt", "w") as f:
    for role, score in recommendations:
        f.write(f"{role}: {score*100:.1f}%\n")
```

**Use a different dataset domain** — the same algorithm works for any item-attribute matching problem: movie recommendations by genre tags, course recommendations by topic, product matching by features. Just change the CSV.

---

## Concepts Demonstrated

This project fulfils all requirements from the DecodeLabs AI Project 3 specification:

| Requirement | Implementation |
|---|---|
| Take user input (choices or interests) | `get_user_skills()` with validation loop |
| Match preferences using logic or similarity | TF-IDF vectorisation + Cosine similarity |
| Display recommended items | `display_results()` with visual bars |
| Logic building | 4-step pipeline in `recommend()` |
| Pattern matching | Vector space model with shared vocabulary |
| Recommendation concepts | Content-based filtering, cold start handling |

---

## Author

DecodeLabs Industrial Training — Batch 2026
