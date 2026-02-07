---
description: Apply code changes safely by ensuring a git backup exists before editing, allowing for precise diff comparison if issues arise.
---

1.  **Context Analysis**:
    -   Identify the specific files valid for modification based on the user's request.
    -   Determine the nature of the change (Refactor, Fix, Feature).

2.  **Pre-Edit Backup (CRITICAL)**:
    -   Run `git status` to see if there are uncommitted changes.
    -   Run `git add .` to stage current files.
    -   Run `git commit -m "Backup: Snapshot before applying [Task Name]"` to save the current state.
    -   *Note: This ensures that we can exacty comparisons using `git diff HEAD~1` later, eliminating guess work.*

3.  **Apply Edits**:
    -   Use `replace_file_content` or `multi_replace_file_content` to modify the code.
    -   Ensure explicit line numbers and content matching are used meticulously.

4.  **Post-Edit Verification**:
    -   Read the modified file (or relevant sections) to ensure the edit was applied correctly.
    -   (Optional) Run a quick syntax check or test provided by the user (e.g., `python -m py_compile [file]`).

5.  **Diff Review**:
    -   Run `git diff` to see exactly what changed in the working directory compared to the backup commit.
    -   Explain the changes to the user based on this actual diff, not memory.

6.  **Final Commit**:
    -   If the changes are correct, run `git add [file]` and `git commit -m "[Task Name]: [Brief Description]"`
    -   Push the changes using `git push`.

7.  **Recovery (If needed)**:
    -   If the result is unexpected, run `git reset --hard HEAD` to revert to the backup snapshot immediately.
