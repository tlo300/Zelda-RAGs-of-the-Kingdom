# Session start

Read CLAUDE.md fully before doing anything else.

Then do the following and present the results clearly:

1. Read the memory file at:
   C:\Users\twanv\.claude\projects\c--Users-twanv-Zelda-RAGs-of-the-Kingdom\memory\MEMORY.md
   Note any relevant preferences or constraints that apply to this session.

2. Show the Current state section from CLAUDE.md.

3. Run: git log --oneline -10
   Summarise what was last worked on in one sentence.

4. Run: gh issue list --repo tlo300/Zelda-RAGs-of-the-Kingdom --label in-progress --json number,title
   List anything currently in progress.

5. Run: gh issue list --repo tlo300/Zelda-RAGs-of-the-Kingdom --milestone "1 - Data pipeline and knowledge base" --state open --json number,title,labels
   List open issues in the current milestone.
   Replace the milestone name with the active one from CLAUDE.md if it has changed.

6. Based on the above, recommend the single best next issue to work on and why.

7. Before starting any issue, invoke the superpowers:brainstorming skill to think through
   the approach. Do not skip this step.

Then wait for confirmation before starting any work.
