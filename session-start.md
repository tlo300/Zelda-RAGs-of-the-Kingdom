# Session start

Read CLAUDE.md fully before doing anything else.

Then do the following and present the results clearly:

1. Show the Current state section from CLAUDE.md.

2. Run: git log --oneline -10
   Summarise what was last worked on in one sentence.

3. Run: gh issue list --repo tlo300/zelda-totk-guide --label in-progress --json number,title
   List anything currently in progress.

4. Run: gh issue list --repo tlo300/zelda-totk-guide --milestone "1 - Data pipeline and knowledge base" --state open --json number,title,labels
   List open issues in the current milestone.
   Replace the milestone name with the active one from CLAUDE.md if it has changed.

5. Based on the above, recommend the single best next issue to work on and why.

Then wait for confirmation before starting any work.
